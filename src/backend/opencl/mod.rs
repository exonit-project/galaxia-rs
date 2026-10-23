// SPDX-License-Identifier: AGPL-3.0-or-later
//! OpenCL Accelerator Backend using dynamic OpenCL ICD loader.
//!
//! **EXPERIMENTAL - NOT VALIDATED - NOT FOR SCIENCE.** Compiled only with the non-default
//! `opencl` cargo feature. Known open issues: rejection-loop cap behaviour, kernel handling,
//! resource leaks on error paths, and thread-safety assumptions of the raw driver handles.
//!
//! Supports Intel Arc iGPU (FP64 via `cl_khr_fp64`), AMD Vega iGPU (FP32 via Mesa rusticl),
//! AMD dGPU, and NVIDIA via OpenCL 1.2+.

use crate::backend::{
    AcceleratorBackend, ActiveLeaf, CudaActiveLeaf, CudaOutputStar, CudaPopParams, CudaSurveyParams,
};
use crate::imf::Sampler;
use crate::interp::VCirc;
use crate::isochrone::IsochroneDB;
use crate::params::Params;
use crate::population::{Population, StarParticle};
use std::ffi::{c_void, CStr, CString};
use std::sync::{Arc, Mutex};

// OpenCL 1.2 Type Definitions
pub type ClPlatformId = *mut c_void;
pub type ClDeviceId = *mut c_void;
pub type ClContext = *mut c_void;
pub type ClCommandQueue = *mut c_void;
pub type ClProgram = *mut c_void;
pub type ClKernel = *mut c_void;
pub type ClMem = *mut c_void;
pub type ClInt = i32;
pub type ClUint = u32;
pub type ClBool = u32;
pub type ClDeviceType = u64;
pub type ClPlatformInfo = u32;
pub type ClDeviceInfo = u32;
pub type ClProgramBuildInfo = u32;
pub type ClMemFlags = u64;

pub const CL_SUCCESS: ClInt = 0;
pub const CL_DEVICE_TYPE_GPU: ClDeviceType = 1 << 2;
pub const CL_DEVICE_TYPE_ALL: ClDeviceType = 0xFFFFFFFF;

pub const CL_PLATFORM_NAME: ClPlatformInfo = 0x0902;
pub const CL_DEVICE_NAME: ClDeviceInfo = 0x102B;
pub const CL_DEVICE_EXTENSIONS: ClDeviceInfo = 0x1030;
pub const CL_PROGRAM_BUILD_LOG: ClProgramBuildInfo = 0x1183;

pub const CL_MEM_READ_WRITE: ClMemFlags = 1 << 0;
pub const CL_MEM_WRITE_ONLY: ClMemFlags = 1 << 1;
pub const CL_MEM_READ_ONLY: ClMemFlags = 1 << 2;
pub const CL_MEM_COPY_HOST_PTR: ClMemFlags = 1 << 5;

pub const CL_TRUE: ClBool = 1;
pub const CL_FALSE: ClBool = 0;

/// Dynamically loaded OpenCL library functions.
#[allow(dead_code)]
pub struct OpenClDriver {
    _lib: libloading::Library,
    cl_get_platform_ids: unsafe extern "C" fn(
        num_entries: ClUint,
        platforms: *mut ClPlatformId,
        num_platforms: *mut ClUint,
    ) -> ClInt,
    cl_get_platform_info: unsafe extern "C" fn(
        platform: ClPlatformId,
        param_name: ClPlatformInfo,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> ClInt,
    cl_get_device_ids: unsafe extern "C" fn(
        platform: ClPlatformId,
        device_type: ClDeviceType,
        num_entries: ClUint,
        devices: *mut ClDeviceId,
        num_devices: *mut ClUint,
    ) -> ClInt,
    cl_get_device_info: unsafe extern "C" fn(
        device: ClDeviceId,
        param_name: ClDeviceInfo,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> ClInt,
    cl_create_context: unsafe extern "C" fn(
        properties: *const isize,
        num_devices: ClUint,
        devices: *const ClDeviceId,
        pfn_notify: *const c_void,
        user_data: *mut c_void,
        errcode_ret: *mut ClInt,
    ) -> ClContext,
    cl_release_context: unsafe extern "C" fn(context: ClContext) -> ClInt,
    cl_create_command_queue: unsafe extern "C" fn(
        context: ClContext,
        device: ClDeviceId,
        properties: u64,
        errcode_ret: *mut ClInt,
    ) -> ClCommandQueue,
    cl_release_command_queue: unsafe extern "C" fn(command_queue: ClCommandQueue) -> ClInt,
    cl_create_program_with_source: unsafe extern "C" fn(
        context: ClContext,
        count: ClUint,
        strings: *const *const i8,
        lengths: *const usize,
        errcode_ret: *mut ClInt,
    ) -> ClProgram,
    cl_build_program: unsafe extern "C" fn(
        program: ClProgram,
        num_devices: ClUint,
        device_list: *const ClDeviceId,
        options: *const i8,
        pfn_notify: *const c_void,
        user_data: *mut c_void,
    ) -> ClInt,
    cl_get_program_build_info: unsafe extern "C" fn(
        program: ClProgram,
        device: ClDeviceId,
        param_name: ClProgramBuildInfo,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> ClInt,
    cl_create_kernel: unsafe extern "C" fn(
        program: ClProgram,
        kernel_name: *const i8,
        errcode_ret: *mut ClInt,
    ) -> ClKernel,
    cl_release_kernel: unsafe extern "C" fn(kernel: ClKernel) -> ClInt,
    cl_release_program: unsafe extern "C" fn(program: ClProgram) -> ClInt,
    cl_create_buffer: unsafe extern "C" fn(
        context: ClContext,
        flags: ClMemFlags,
        size: usize,
        host_ptr: *mut c_void,
        errcode_ret: *mut ClInt,
    ) -> ClMem,
    cl_release_mem_object: unsafe extern "C" fn(memobj: ClMem) -> ClInt,
    cl_enqueue_write_buffer: unsafe extern "C" fn(
        command_queue: ClCommandQueue,
        buffer: ClMem,
        blocking_write: ClBool,
        offset: usize,
        size: usize,
        ptr: *const c_void,
        num_events_in_wait_list: ClUint,
        event_wait_list: *const c_void,
        event: *mut c_void,
    ) -> ClInt,
    cl_enqueue_read_buffer: unsafe extern "C" fn(
        command_queue: ClCommandQueue,
        buffer: ClMem,
        blocking_read: ClBool,
        offset: usize,
        size: usize,
        ptr: *mut c_void,
        num_events_in_wait_list: ClUint,
        event_wait_list: *const c_void,
        event: *mut c_void,
    ) -> ClInt,
    cl_set_kernel_arg: unsafe extern "C" fn(
        kernel: ClKernel,
        arg_index: ClUint,
        arg_size: usize,
        arg_value: *const c_void,
    ) -> ClInt,
    cl_enqueue_nd_range_kernel: unsafe extern "C" fn(
        command_queue: ClCommandQueue,
        kernel: ClKernel,
        work_dim: ClUint,
        global_work_offset: *const usize,
        global_work_size: *const usize,
        local_work_size: *const usize,
        num_events_in_wait_list: ClUint,
        event_wait_list: *const c_void,
        event: *mut c_void,
    ) -> ClInt,
    cl_finish: unsafe extern "C" fn(command_queue: ClCommandQueue) -> ClInt,
}

impl OpenClDriver {
    pub fn load() -> anyhow::Result<Self> {
        // Automatically enable rusticl for AMD / Intel if Mesa rusticl is present
        if std::env::var("RUSTICL_ENABLE").is_err() {
            std::env::set_var("RUSTICL_ENABLE", "radeonsi,iris,all");
        }

        let candidates = [
            "libOpenCL.so.1",
            "libOpenCL.so",
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
            "OpenCL.dll",
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
        ];
        let mut loaded = None;
        for c in &candidates {
            if let Ok(lib) = unsafe { libloading::Library::new(*c) } {
                loaded = Some(lib);
                break;
            }
        }

        let lib = loaded
            .ok_or_else(|| anyhow::anyhow!("OpenCL dynamic library (libOpenCL.so.1) not found"))?;

        unsafe {
            let cl_get_platform_ids = *lib.get(b"clGetPlatformIDs\0")?;
            let cl_get_platform_info = *lib.get(b"clGetPlatformInfo\0")?;
            let cl_get_device_ids = *lib.get(b"clGetDeviceIDs\0")?;
            let cl_get_device_info = *lib.get(b"clGetDeviceInfo\0")?;
            let cl_create_context = *lib.get(b"clCreateContext\0")?;
            let cl_release_context = *lib.get(b"clReleaseContext\0")?;
            let cl_create_command_queue = *lib.get(b"clCreateCommandQueue\0")?;
            let cl_release_command_queue = *lib.get(b"clReleaseCommandQueue\0")?;
            let cl_create_program_with_source = *lib.get(b"clCreateProgramWithSource\0")?;
            let cl_build_program = *lib.get(b"clBuildProgram\0")?;
            let cl_get_program_build_info = *lib.get(b"clGetProgramBuildInfo\0")?;
            let cl_create_kernel = *lib.get(b"clCreateKernel\0")?;
            let cl_release_kernel = *lib.get(b"clReleaseKernel\0")?;
            let cl_release_program = *lib.get(b"clReleaseProgram\0")?;
            let cl_create_buffer = *lib.get(b"clCreateBuffer\0")?;
            let cl_release_mem_object = *lib.get(b"clReleaseMemObject\0")?;
            let cl_enqueue_write_buffer = *lib.get(b"clEnqueueWriteBuffer\0")?;
            let cl_enqueue_read_buffer = *lib.get(b"clEnqueueReadBuffer\0")?;
            let cl_set_kernel_arg = *lib.get(b"clSetKernelArg\0")?;
            let cl_enqueue_nd_range_kernel = *lib.get(b"clEnqueueNDRangeKernel\0")?;
            let cl_finish = *lib.get(b"clFinish\0")?;

            Ok(Self {
                _lib: lib,
                cl_get_platform_ids,
                cl_get_platform_info,
                cl_get_device_ids,
                cl_get_device_info,
                cl_create_context,
                cl_release_context,
                cl_create_command_queue,
                cl_release_command_queue,
                cl_create_program_with_source,
                cl_build_program,
                cl_get_program_build_info,
                cl_create_kernel,
                cl_release_kernel,
                cl_release_program,
                cl_create_buffer,
                cl_release_mem_object,
                cl_enqueue_write_buffer,
                cl_enqueue_read_buffer,
                cl_set_kernel_arg,
                cl_enqueue_nd_range_kernel,
                cl_finish,
            })
        }
    }
}

const OPENCL_KERNEL_SOURCE: &str = include_str!("kernels/spawn.cl");

/// Precision mode for OpenCL accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    Fp64,
    Fp32,
}

// Pod structs for FP32 mode (e.g. AMD Vega iGPU)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ClActiveLeafF32 {
    pub pos: [f32; 4],
    pub l: f32,
    pub lage: f32,
    pub rho_max: f32,
    pub rho_min: f32,
    pub dmod: f32,
    pub box_mmin: f32,
    pub box_mmax: f32,
    pub cpd_min: f32,
    pub cpd_max: f32,
    pub orig_leaf_idx: u32,
    pub n_stars: u32,
    pub star_offset: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ClPopParamsF32 {
    pub pop_id: i32,
    pub option_e: i32,
    pub warp_flare_on: i32,
    pub pad0: i32,
    pub age: f32,
    pub dage: f32,
    pub feh: f32,
    pub dfeh: f32,
    pub sigma_v: [f32; 3],
    pub pad1: f32,
    pub rho0: f32,
    pub hr1: f32,
    pub hr2: f32,
    pub xl: f32,
    pub r_warp: f32,
    pub r_flare: f32,
    pub g_warp: f32,
    pub g_flare: f32,
    pub th_max: f32,
    pub qparam: f32,
    pub bparam: f32,
    pub ac: f32,
    pub rhoc: f32,
    pub eps: f32,
    pub x0: f32,
    pub y0: f32,
    pub z0: f32,
    pub rc_bulge: f32,
    pub rot_matrix: [[f32; 3]; 3],
    pub agevec: [f32; 7],
    pub fehvec: [f32; 7],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ClSurveyParamsF32 {
    pub pos_c: [f32; 6],
    pub cone_dir: [f32; 3],
    pub cone_dth: f32,
    pub app_mag_limits: [f32; 2],
    pub color_limits: [f32; 2],
    pub abs_mag_limits: [f32; 2],
    pub r_max: f32,
    pub seed: u64,
    pub field_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ClOutputStarF32 {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub age: f32,
    pub feh: f32,
    pub smass: f32,
    pub mags: [f32; 3],
    pub abs_mags: [f32; 3],
    pub rad: f32,
    pub dmod: f32,
    pub popid: i32,
    pub accepted: i32,
}

/// Resident OpenCL buffers for shared tables (Isochrones and VCirc).
struct ResidentTables {
    d_isochrone_m: ClMem,
    d_isochrone_mag0: ClMem,
    d_isochrone_mag1: ClMem,
    d_isochrone_mag2: ClMem,
    d_track_offsets: ClMem,
    d_track_m_min: ClMem,
    d_track_m_max: ClMem,
    d_feh_grid: ClMem,
    num_ages: u32,
    num_feh: u32,
    age_grid_min_f64: f64,
    d_age_f64: f64,
    age_grid_min_f32: f32,
    d_age_f32: f32,
    d_vcirc_r: ClMem,
    d_vcirc_v: ClMem,
    vcirc_size: u32,
}

/// OpenCL backend implementing `AcceleratorBackend`.
pub struct OpenClBackend {
    driver: Arc<OpenClDriver>,
    context: ClContext,
    queue: ClCommandQueue,
    program: ClProgram,
    kernel: ClKernel,
    device_name: String,
    precision: PrecisionMode,
    resident: Mutex<Option<ResidentTables>>,
}

unsafe impl Send for OpenClBackend {}
unsafe impl Sync for OpenClBackend {}

impl OpenClBackend {
    /// Initialize the OpenCL accelerator backend.
    /// Detects GPU device, checks precision capabilities (FP64 or FP32 fallback for AMD Vega),
    /// compiles the kernel with matching options, and creates the execution context.
    pub fn new() -> anyhow::Result<Self> {
        let driver = Arc::new(OpenClDriver::load()?);

        unsafe {
            // 1. Enumerate platforms
            let mut num_platforms: ClUint = 0;
            let res = (driver.cl_get_platform_ids)(0, std::ptr::null_mut(), &mut num_platforms);
            if res != CL_SUCCESS || num_platforms == 0 {
                return Err(anyhow::anyhow!("No OpenCL platforms found (code {})", res));
            }

            let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
            (driver.cl_get_platform_ids)(
                num_platforms,
                platforms.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            // 2. Find suitable GPU device
            let mut selected_device: Option<(ClPlatformId, ClDeviceId, String, bool)> = None;

            for platform in platforms {
                let mut num_devices: ClUint = 0;
                let dev_res = (driver.cl_get_device_ids)(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    0,
                    std::ptr::null_mut(),
                    &mut num_devices,
                );
                if dev_res != CL_SUCCESS || num_devices == 0 {
                    continue;
                }

                let mut devices = vec![std::ptr::null_mut(); num_devices as usize];
                (driver.cl_get_device_ids)(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    num_devices,
                    devices.as_mut_ptr(),
                    std::ptr::null_mut(),
                );

                for dev in devices {
                    let mut name_buf = [0i8; 256];
                    let mut ext_buf = [0i8; 4096];
                    (driver.cl_get_device_info)(
                        dev,
                        CL_DEVICE_NAME,
                        256,
                        name_buf.as_mut_ptr() as *mut c_void,
                        std::ptr::null_mut(),
                    );
                    (driver.cl_get_device_info)(
                        dev,
                        CL_DEVICE_EXTENSIONS,
                        4096,
                        ext_buf.as_mut_ptr() as *mut c_void,
                        std::ptr::null_mut(),
                    );

                    let dev_name = CStr::from_ptr(name_buf.as_ptr())
                        .to_string_lossy()
                        .into_owned();
                    let dev_ext = CStr::from_ptr(ext_buf.as_ptr())
                        .to_string_lossy()
                        .into_owned();

                    let has_fp64 =
                        dev_ext.contains("cl_khr_fp64") || dev_ext.contains("cl_amd_fp64");
                    selected_device = Some((platform, dev, dev_name, has_fp64));
                    break;
                }

                if selected_device.is_some() {
                    break;
                }
            }

            let (_platform, device, device_name, has_fp64) = selected_device
                .ok_or_else(|| anyhow::anyhow!("No OpenCL GPU device found on this system"))?;

            let precision = if has_fp64 {
                PrecisionMode::Fp64
            } else {
                PrecisionMode::Fp32
            };

            // 3. Create context & command queue
            let mut err: ClInt = 0;
            let context = (driver.cl_create_context)(
                std::ptr::null(),
                1,
                &device,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut err,
            );
            if err != CL_SUCCESS || context.is_null() {
                return Err(anyhow::anyhow!("clCreateContext failed with code {}", err));
            }

            let queue = (driver.cl_create_command_queue)(context, device, 0, &mut err);
            if err != CL_SUCCESS || queue.is_null() {
                (driver.cl_release_context)(context);
                return Err(anyhow::anyhow!(
                    "clCreateCommandQueue failed with code {}",
                    err
                ));
            }

            // 4. Build program from embedded source with appropriate precision options
            let source_bytes = OPENCL_KERNEL_SOURCE.as_bytes();
            let source_ptr = source_bytes.as_ptr() as *const i8;
            let source_len = source_bytes.len();

            let program = (driver.cl_create_program_with_source)(
                context,
                1,
                &source_ptr,
                &source_len,
                &mut err,
            );
            if err != CL_SUCCESS || program.is_null() {
                (driver.cl_release_command_queue)(queue);
                (driver.cl_release_context)(context);
                return Err(anyhow::anyhow!(
                    "clCreateProgramWithSource failed with code {}",
                    err
                ));
            }

            let build_opts_str = if precision == PrecisionMode::Fp64 {
                "-cl-std=CL1.2".to_string()
            } else {
                "-cl-std=CL1.2 -DREAL_IS_FP32 -cl-single-precision-constant".to_string()
            };
            let build_opts = CString::new(build_opts_str).unwrap();

            let build_res = (driver.cl_build_program)(
                program,
                1,
                &device,
                build_opts.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            );

            if build_res != CL_SUCCESS {
                let mut log_len: usize = 0;
                (driver.cl_get_program_build_info)(
                    program,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    0,
                    std::ptr::null_mut(),
                    &mut log_len,
                );
                let mut log_buf = vec![0i8; log_len.max(1)];
                (driver.cl_get_program_build_info)(
                    program,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    log_len,
                    log_buf.as_mut_ptr() as *mut c_void,
                    std::ptr::null_mut(),
                );
                let build_log = CStr::from_ptr(log_buf.as_ptr()).to_string_lossy();

                (driver.cl_release_program)(program);
                (driver.cl_release_command_queue)(queue);
                (driver.cl_release_context)(context);
                return Err(anyhow::anyhow!(
                    "clBuildProgram failed (code {}):\n{}",
                    build_res,
                    build_log
                ));
            }

            // 5. Create kernel
            let kernel_name = CString::new("spawn_kernel").unwrap();
            let kernel = (driver.cl_create_kernel)(program, kernel_name.as_ptr(), &mut err);
            if err != CL_SUCCESS || kernel.is_null() {
                (driver.cl_release_program)(program);
                (driver.cl_release_command_queue)(queue);
                (driver.cl_release_context)(context);
                return Err(anyhow::anyhow!(
                    "clCreateKernel(spawn_kernel) failed with code {}",
                    err
                ));
            }

            Ok(Self {
                driver,
                context,
                queue,
                program,
                kernel,
                device_name,
                precision,
                resident: Mutex::new(None),
            })
        }
    }

    /// Get current precision mode.
    pub fn precision(&self) -> PrecisionMode {
        self.precision
    }

    /// Allocate OpenCL device buffer and copy host data into it.
    unsafe fn upload_slice<T>(&self, data: &[T], flags: ClMemFlags) -> anyhow::Result<ClMem> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let mut err: ClInt = 0;
        let buf = (self.driver.cl_create_buffer)(
            self.context,
            flags | CL_MEM_COPY_HOST_PTR,
            bytes,
            data.as_ptr() as *mut c_void,
            &mut err,
        );
        if err != CL_SUCCESS || buf.is_null() {
            return Err(anyhow::anyhow!("clCreateBuffer failed with code {}", err));
        }
        Ok(buf)
    }

    /// Free OpenCL memory object.
    unsafe fn free_mem(&self, mem: ClMem) {
        if !mem.is_null() {
            (self.driver.cl_release_mem_object)(mem);
        }
    }

    /// Ensure resident tables (Isochrones and VCirc) are uploaded to device memory.
    fn ensure_resident_tables(
        &self,
        isochrones: &IsochroneDB,
        vcirc: &VCirc,
    ) -> anyhow::Result<()> {
        let mut guard = self.resident.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }

        let num_ages = isochrones.age_grid.len() as u32;
        let num_feh = isochrones.feh_grid.len() as u32;
        let total_tracks = (num_ages * num_feh) as usize;

        let mut track_offsets = Vec::with_capacity(total_tracks + 1);
        let mut cur_offset = 0u32;
        track_offsets.push(cur_offset);

        let mut isochrone_m_f64 = Vec::new();
        let mut isochrone_mag0_f64 = Vec::new();
        let mut isochrone_mag1_f64 = Vec::new();
        let mut isochrone_mag2_f64 = Vec::new();
        let mut track_m_min_f64 = Vec::with_capacity(total_tracks);
        let mut track_m_max_f64 = Vec::with_capacity(total_tracks);

        for j in 0..num_feh as usize {
            for i in 0..num_ages as usize {
                let track_idx = j * (num_ages as usize) + i;
                if track_idx < isochrones.tracks.len() {
                    let track = &isochrones.tracks[track_idx];
                    let count = track.m.len() as u32;
                    cur_offset += count;
                    track_offsets.push(cur_offset);
                    track_m_min_f64.push(track.m_min);
                    track_m_max_f64.push(*track.m.last().unwrap_or(&100.0));

                    isochrone_m_f64.extend_from_slice(&track.m);
                    isochrone_mag0_f64.extend_from_slice(&track.mag0);
                    isochrone_mag1_f64.extend_from_slice(&track.mag1);
                    isochrone_mag2_f64.extend_from_slice(&track.mag2);
                } else {
                    track_offsets.push(cur_offset);
                    track_m_min_f64.push(100.0);
                    track_m_max_f64.push(0.0);
                }
            }
        }

        let age_grid_min_f64 = isochrones.age_grid[0];
        let d_age_f64 = isochrones.d_age;
        let age_grid_min_f32 = age_grid_min_f64 as f32;
        let d_age_f32 = d_age_f64 as f32;

        unsafe {
            let (
                d_isochrone_m,
                d_isochrone_mag0,
                d_isochrone_mag1,
                d_isochrone_mag2,
                d_track_m_min,
                d_track_m_max,
                d_feh_grid,
                d_vcirc_r,
                d_vcirc_v,
            ) = if self.precision == PrecisionMode::Fp64 {
                (
                    self.upload_slice(&isochrone_m_f64, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&isochrone_mag0_f64, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&isochrone_mag1_f64, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&isochrone_mag2_f64, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&track_m_min_f64, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&track_m_max_f64, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&isochrones.feh_grid, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&vcirc.r, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&vcirc.v, CL_MEM_READ_ONLY)?,
                )
            } else {
                let m_f32: Vec<f32> = isochrone_m_f64.iter().map(|&x| x as f32).collect();
                let mag0_f32: Vec<f32> = isochrone_mag0_f64.iter().map(|&x| x as f32).collect();
                let mag1_f32: Vec<f32> = isochrone_mag1_f64.iter().map(|&x| x as f32).collect();
                let mag2_f32: Vec<f32> = isochrone_mag2_f64.iter().map(|&x| x as f32).collect();
                let m_min_f32: Vec<f32> = track_m_min_f64.iter().map(|&x| x as f32).collect();
                let m_max_f32: Vec<f32> = track_m_max_f64.iter().map(|&x| x as f32).collect();
                let feh_f32: Vec<f32> = isochrones.feh_grid.iter().map(|&x| x as f32).collect();
                let vr_f32: Vec<f32> = vcirc.r.iter().map(|&x| x as f32).collect();
                let vv_f32: Vec<f32> = vcirc.v.iter().map(|&x| x as f32).collect();
                (
                    self.upload_slice(&m_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&mag0_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&mag1_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&mag2_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&m_min_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&m_max_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&feh_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&vr_f32, CL_MEM_READ_ONLY)?,
                    self.upload_slice(&vv_f32, CL_MEM_READ_ONLY)?,
                )
            };

            let d_track_offsets = self.upload_slice(&track_offsets, CL_MEM_READ_ONLY)?;

            *guard = Some(ResidentTables {
                d_isochrone_m,
                d_isochrone_mag0,
                d_isochrone_mag1,
                d_isochrone_mag2,
                d_track_offsets,
                d_track_m_min,
                d_track_m_max,
                d_feh_grid,
                num_ages,
                num_feh,
                age_grid_min_f64,
                d_age_f64,
                age_grid_min_f32,
                d_age_f32,
                d_vcirc_r,
                d_vcirc_v,
                vcirc_size: vcirc.r.len() as u32,
            });
        }

        Ok(())
    }
}

impl Drop for OpenClBackend {
    fn drop(&mut self) {
        unsafe {
            if let Some(res) = self.resident.lock().unwrap().take() {
                self.free_mem(res.d_isochrone_m);
                self.free_mem(res.d_isochrone_mag0);
                self.free_mem(res.d_isochrone_mag1);
                self.free_mem(res.d_isochrone_mag2);
                self.free_mem(res.d_track_offsets);
                self.free_mem(res.d_track_m_min);
                self.free_mem(res.d_track_m_max);
                self.free_mem(res.d_feh_grid);
                self.free_mem(res.d_vcirc_r);
                self.free_mem(res.d_vcirc_v);
            }
            if !self.kernel.is_null() {
                (self.driver.cl_release_kernel)(self.kernel);
            }
            if !self.program.is_null() {
                (self.driver.cl_release_program)(self.program);
            }
            if !self.queue.is_null() {
                (self.driver.cl_release_command_queue)(self.queue);
            }
            if !self.context.is_null() {
                (self.driver.cl_release_context)(self.context);
            }
        }
    }
}

impl AcceleratorBackend for OpenClBackend {
    fn name(&self) -> &str {
        &self.device_name
    }

    fn platform_tag(&self) -> String {
        match self.precision {
            PrecisionMode::Fp64 => format!("galaxia-rs-{}-opencl-fp64", env!("CARGO_PKG_VERSION")),
            PrecisionMode::Fp32 => format!("galaxia-rs-{}-opencl-fp32", env!("CARGO_PKG_VERSION")),
        }
    }

    fn is_production_grade(&self) -> bool {
        self.precision == PrecisionMode::Fp64
    }

    fn spawn_stars(
        &self,
        pop_id: usize,
        pop: &dyn Population,
        params: &Params,
        active_leaves: &[ActiveLeaf],
        imf: &Sampler,
        isochrones: &IsochroneDB,
        vcirc: &VCirc,
    ) -> anyhow::Result<Vec<StarParticle>> {
        let total_stars: usize = active_leaves.iter().map(|l| l.n_tot).sum();
        if total_stars == 0 {
            return Ok(Vec::new());
        }

        self.ensure_resident_tables(isochrones, vcirc)?;
        let res_guard = self.resident.lock().unwrap();
        let res = res_guard.as_ref().unwrap();

        // Flatten star mapping arrays
        let mut star_to_leaf = Vec::with_capacity(total_stars);
        let mut star_in_leaf = Vec::with_capacity(total_stars);

        for (l_idx, leaf) in active_leaves.iter().enumerate() {
            for s_idx in 0..leaf.n_tot {
                star_to_leaf.push(l_idx as u32);
                star_in_leaf.push(s_idx as u32);
            }
        }

        let imf_size = imf.x.len() as u32;
        let num_ages = res.num_ages;
        let num_feh = res.num_feh;
        let vcirc_size = res.vcirc_size;
        let total_stars_u32 = total_stars as u32;

        unsafe {
            let d_star_to_leaf = self.upload_slice(&star_to_leaf, CL_MEM_READ_ONLY)?;
            let d_star_in_leaf = self.upload_slice(&star_in_leaf, CL_MEM_READ_ONLY)?;

            let set_arg = |idx: u32, size: usize, ptr: *const c_void| -> anyhow::Result<()> {
                let code = (self.driver.cl_set_kernel_arg)(self.kernel, idx, size, ptr);
                if code != CL_SUCCESS {
                    Err(anyhow::anyhow!(
                        "clSetKernelArg({}) failed with code {}",
                        idx,
                        code
                    ))
                } else {
                    Ok(())
                }
            };

            let (d_leaves, d_imf_x, d_imf_cpd, d_out_stars, out_bytes) =
                if self.precision == PrecisionMode::Fp64 {
                    let mut leaves_gpu = Vec::with_capacity(active_leaves.len());
                    for leaf in active_leaves {
                        leaves_gpu.push(CudaActiveLeaf {
                            pos: leaf.pos,
                            l: leaf.l,
                            lage: leaf.lage,
                            rho_max: leaf.rho_max,
                            rho_min: leaf.rho_min,
                            dmod: leaf.dmod,
                            box_mmin: leaf.box_mmin,
                            box_mmax: leaf.box_mmax,
                            cpd_min: leaf.cpd_min,
                            cpd_max: leaf.cpd_max,
                            orig_leaf_idx: leaf.orig_leaf_idx as u32,
                            n_stars: leaf.n_tot as u32,
                            star_offset: leaf.star_offset as u32,
                            pad: 0,
                        });
                    }
                    let pop_params = make_pop_params_f64(pop_id, pop, params);
                    let survey_params = make_survey_params_f64(params);
                    let d_leaves = self.upload_slice(&leaves_gpu, CL_MEM_READ_ONLY)?;
                    let d_imf_x = self.upload_slice(&imf.x, CL_MEM_READ_ONLY)?;
                    let d_imf_cpd = self.upload_slice(&imf.cpd, CL_MEM_READ_ONLY)?;

                    let out_bytes = total_stars * std::mem::size_of::<CudaOutputStar>();
                    let mut err: ClInt = 0;
                    let d_out_stars = (self.driver.cl_create_buffer)(
                        self.context,
                        CL_MEM_READ_WRITE,
                        out_bytes,
                        std::ptr::null_mut(),
                        &mut err,
                    );
                    if err != CL_SUCCESS || d_out_stars.is_null() {
                        return Err(anyhow::anyhow!(
                            "clCreateBuffer output stars failed: {}",
                            err
                        ));
                    }

                    set_arg(
                        0,
                        std::mem::size_of::<CudaPopParams>(),
                        &pop_params as *const _ as *const c_void,
                    )?;
                    set_arg(
                        1,
                        std::mem::size_of::<CudaSurveyParams>(),
                        &survey_params as *const _ as *const c_void,
                    )?;
                    set_arg(
                        2,
                        std::mem::size_of::<ClMem>(),
                        &d_leaves as *const _ as *const c_void,
                    )?;
                    set_arg(
                        3,
                        std::mem::size_of::<ClMem>(),
                        &d_star_to_leaf as *const _ as *const c_void,
                    )?;
                    set_arg(
                        4,
                        std::mem::size_of::<ClMem>(),
                        &d_star_in_leaf as *const _ as *const c_void,
                    )?;
                    set_arg(
                        5,
                        std::mem::size_of::<ClMem>(),
                        &d_imf_x as *const _ as *const c_void,
                    )?;
                    set_arg(
                        6,
                        std::mem::size_of::<ClMem>(),
                        &d_imf_cpd as *const _ as *const c_void,
                    )?;
                    set_arg(
                        7,
                        std::mem::size_of::<u32>(),
                        &imf_size as *const _ as *const c_void,
                    )?;
                    set_arg(
                        8,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_m as *const _ as *const c_void,
                    )?;
                    set_arg(
                        9,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_mag0 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        10,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_mag1 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        11,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_mag2 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        12,
                        std::mem::size_of::<ClMem>(),
                        &res.d_track_offsets as *const _ as *const c_void,
                    )?;
                    set_arg(
                        13,
                        std::mem::size_of::<ClMem>(),
                        &res.d_track_m_min as *const _ as *const c_void,
                    )?;
                    set_arg(
                        14,
                        std::mem::size_of::<ClMem>(),
                        &res.d_track_m_max as *const _ as *const c_void,
                    )?;
                    set_arg(
                        15,
                        std::mem::size_of::<u32>(),
                        &num_ages as *const _ as *const c_void,
                    )?;
                    set_arg(
                        16,
                        std::mem::size_of::<u32>(),
                        &num_feh as *const _ as *const c_void,
                    )?;
                    set_arg(
                        17,
                        std::mem::size_of::<f64>(),
                        &res.age_grid_min_f64 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        18,
                        std::mem::size_of::<f64>(),
                        &res.d_age_f64 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        19,
                        std::mem::size_of::<ClMem>(),
                        &res.d_feh_grid as *const _ as *const c_void,
                    )?;
                    set_arg(
                        20,
                        std::mem::size_of::<ClMem>(),
                        &res.d_vcirc_r as *const _ as *const c_void,
                    )?;
                    set_arg(
                        21,
                        std::mem::size_of::<ClMem>(),
                        &res.d_vcirc_v as *const _ as *const c_void,
                    )?;
                    set_arg(
                        22,
                        std::mem::size_of::<u32>(),
                        &vcirc_size as *const _ as *const c_void,
                    )?;
                    set_arg(
                        23,
                        std::mem::size_of::<ClMem>(),
                        &d_out_stars as *const _ as *const c_void,
                    )?;
                    set_arg(
                        24,
                        std::mem::size_of::<u32>(),
                        &total_stars_u32 as *const _ as *const c_void,
                    )?;

                    (d_leaves, d_imf_x, d_imf_cpd, d_out_stars, out_bytes)
                } else {
                    let mut leaves_gpu_f32 = Vec::with_capacity(active_leaves.len());
                    for leaf in active_leaves {
                        leaves_gpu_f32.push(ClActiveLeafF32 {
                            pos: [
                                leaf.pos[0] as f32,
                                leaf.pos[1] as f32,
                                leaf.pos[2] as f32,
                                leaf.pos[3] as f32,
                            ],
                            l: leaf.l as f32,
                            lage: leaf.lage as f32,
                            rho_max: leaf.rho_max as f32,
                            rho_min: leaf.rho_min as f32,
                            dmod: leaf.dmod as f32,
                            box_mmin: leaf.box_mmin as f32,
                            box_mmax: leaf.box_mmax as f32,
                            cpd_min: leaf.cpd_min as f32,
                            cpd_max: leaf.cpd_max as f32,
                            orig_leaf_idx: leaf.orig_leaf_idx as u32,
                            n_stars: leaf.n_tot as u32,
                            star_offset: leaf.star_offset as u32,
                            pad: 0,
                        });
                    }
                    let pop_params = make_pop_params_f32(pop_id, pop, params);
                    let survey_params = make_survey_params_f32(params);
                    let imf_x_f32: Vec<f32> = imf.x.iter().map(|&x| x as f32).collect();
                    let imf_cpd_f32: Vec<f32> = imf.cpd.iter().map(|&x| x as f32).collect();

                    let d_leaves = self.upload_slice(&leaves_gpu_f32, CL_MEM_READ_ONLY)?;
                    let d_imf_x = self.upload_slice(&imf_x_f32, CL_MEM_READ_ONLY)?;
                    let d_imf_cpd = self.upload_slice(&imf_cpd_f32, CL_MEM_READ_ONLY)?;

                    let out_bytes = total_stars * std::mem::size_of::<ClOutputStarF32>();
                    let mut err: ClInt = 0;
                    let d_out_stars = (self.driver.cl_create_buffer)(
                        self.context,
                        CL_MEM_READ_WRITE,
                        out_bytes,
                        std::ptr::null_mut(),
                        &mut err,
                    );
                    if err != CL_SUCCESS || d_out_stars.is_null() {
                        return Err(anyhow::anyhow!(
                            "clCreateBuffer output stars failed: {}",
                            err
                        ));
                    }

                    set_arg(
                        0,
                        std::mem::size_of::<ClPopParamsF32>(),
                        &pop_params as *const _ as *const c_void,
                    )?;
                    set_arg(
                        1,
                        std::mem::size_of::<ClSurveyParamsF32>(),
                        &survey_params as *const _ as *const c_void,
                    )?;
                    set_arg(
                        2,
                        std::mem::size_of::<ClMem>(),
                        &d_leaves as *const _ as *const c_void,
                    )?;
                    set_arg(
                        3,
                        std::mem::size_of::<ClMem>(),
                        &d_star_to_leaf as *const _ as *const c_void,
                    )?;
                    set_arg(
                        4,
                        std::mem::size_of::<ClMem>(),
                        &d_star_in_leaf as *const _ as *const c_void,
                    )?;
                    set_arg(
                        5,
                        std::mem::size_of::<ClMem>(),
                        &d_imf_x as *const _ as *const c_void,
                    )?;
                    set_arg(
                        6,
                        std::mem::size_of::<ClMem>(),
                        &d_imf_cpd as *const _ as *const c_void,
                    )?;
                    set_arg(
                        7,
                        std::mem::size_of::<u32>(),
                        &imf_size as *const _ as *const c_void,
                    )?;
                    set_arg(
                        8,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_m as *const _ as *const c_void,
                    )?;
                    set_arg(
                        9,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_mag0 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        10,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_mag1 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        11,
                        std::mem::size_of::<ClMem>(),
                        &res.d_isochrone_mag2 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        12,
                        std::mem::size_of::<ClMem>(),
                        &res.d_track_offsets as *const _ as *const c_void,
                    )?;
                    set_arg(
                        13,
                        std::mem::size_of::<ClMem>(),
                        &res.d_track_m_min as *const _ as *const c_void,
                    )?;
                    set_arg(
                        14,
                        std::mem::size_of::<ClMem>(),
                        &res.d_track_m_max as *const _ as *const c_void,
                    )?;
                    set_arg(
                        15,
                        std::mem::size_of::<u32>(),
                        &num_ages as *const _ as *const c_void,
                    )?;
                    set_arg(
                        16,
                        std::mem::size_of::<u32>(),
                        &num_feh as *const _ as *const c_void,
                    )?;
                    set_arg(
                        17,
                        std::mem::size_of::<f32>(),
                        &res.age_grid_min_f32 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        18,
                        std::mem::size_of::<f32>(),
                        &res.d_age_f32 as *const _ as *const c_void,
                    )?;
                    set_arg(
                        19,
                        std::mem::size_of::<ClMem>(),
                        &res.d_feh_grid as *const _ as *const c_void,
                    )?;
                    set_arg(
                        20,
                        std::mem::size_of::<ClMem>(),
                        &res.d_vcirc_r as *const _ as *const c_void,
                    )?;
                    set_arg(
                        21,
                        std::mem::size_of::<ClMem>(),
                        &res.d_vcirc_v as *const _ as *const c_void,
                    )?;
                    set_arg(
                        22,
                        std::mem::size_of::<u32>(),
                        &vcirc_size as *const _ as *const c_void,
                    )?;
                    set_arg(
                        23,
                        std::mem::size_of::<ClMem>(),
                        &d_out_stars as *const _ as *const c_void,
                    )?;
                    set_arg(
                        24,
                        std::mem::size_of::<u32>(),
                        &total_stars_u32 as *const _ as *const c_void,
                    )?;

                    (d_leaves, d_imf_x, d_imf_cpd, d_out_stars, out_bytes)
                };

            let local_work_size: usize = 256;
            let global_work_size: usize =
                ((total_stars + local_work_size - 1) / local_work_size) * local_work_size;

            let launch_res = (self.driver.cl_enqueue_nd_range_kernel)(
                self.queue,
                self.kernel,
                1,
                std::ptr::null(),
                &global_work_size,
                &local_work_size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );

            if launch_res != CL_SUCCESS {
                self.free_mem(d_leaves);
                self.free_mem(d_star_to_leaf);
                self.free_mem(d_star_in_leaf);
                self.free_mem(d_imf_x);
                self.free_mem(d_imf_cpd);
                self.free_mem(d_out_stars);
                return Err(anyhow::anyhow!(
                    "clEnqueueNDRangeKernel failed with code {}",
                    launch_res
                ));
            }

            let sync_res = (self.driver.cl_finish)(self.queue);
            if sync_res != CL_SUCCESS {
                self.free_mem(d_leaves);
                self.free_mem(d_star_to_leaf);
                self.free_mem(d_star_in_leaf);
                self.free_mem(d_imf_x);
                self.free_mem(d_imf_cpd);
                self.free_mem(d_out_stars);
                return Err(anyhow::anyhow!("clFinish failed with code {}", sync_res));
            }

            let mut accepted_stars = Vec::new();

            if self.precision == PrecisionMode::Fp64 {
                let mut out_stars_host: Vec<CudaOutputStar> =
                    vec![CudaOutputStar::default(); total_stars];
                let read_res = (self.driver.cl_enqueue_read_buffer)(
                    self.queue,
                    d_out_stars,
                    CL_TRUE,
                    0,
                    out_bytes,
                    out_stars_host.as_mut_ptr() as *mut c_void,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                );
                if read_res != CL_SUCCESS {
                    return Err(anyhow::anyhow!(
                        "clEnqueueReadBuffer failed with code {}",
                        read_res
                    ));
                }

                for s in out_stars_host {
                    if s.accepted == 1 {
                        accepted_stars.push(StarParticle {
                            pos: s.pos,
                            vel: s.vel,
                            age: s.age,
                            feh: s.feh,
                            alpha: s.feh,
                            smass: s.smass,
                            rad: s.rad,
                            mags: s.mags,
                            abs_mags: s.abs_mags,
                            lum: 0.0,
                            teff: 0.0,
                            grav: 0.0,
                            dmod: s.dmod,
                            mact: 0.0,
                            mtip: 0.0,
                            popid: s.popid,
                            satid: 0,
                            partid: 0,
                        });
                    }
                }
            } else {
                let mut out_stars_host_f32: Vec<ClOutputStarF32> =
                    vec![ClOutputStarF32::default(); total_stars];
                let read_res = (self.driver.cl_enqueue_read_buffer)(
                    self.queue,
                    d_out_stars,
                    CL_TRUE,
                    0,
                    out_bytes,
                    out_stars_host_f32.as_mut_ptr() as *mut c_void,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                );
                if read_res != CL_SUCCESS {
                    return Err(anyhow::anyhow!(
                        "clEnqueueReadBuffer failed with code {}",
                        read_res
                    ));
                }

                for s in out_stars_host_f32 {
                    if s.accepted == 1 {
                        accepted_stars.push(StarParticle {
                            pos: [s.pos[0] as f64, s.pos[1] as f64, s.pos[2] as f64],
                            vel: [s.vel[0] as f64, s.vel[1] as f64, s.vel[2] as f64],
                            age: s.age as f64,
                            feh: s.feh as f64,
                            alpha: s.feh as f64,
                            smass: s.smass as f64,
                            rad: s.rad as f64,
                            mags: [s.mags[0] as f64, s.mags[1] as f64, s.mags[2] as f64],
                            abs_mags: [
                                s.abs_mags[0] as f64,
                                s.abs_mags[1] as f64,
                                s.abs_mags[2] as f64,
                            ],
                            lum: 0.0,
                            teff: 0.0,
                            grav: 0.0,
                            dmod: s.dmod as f64,
                            mact: 0.0,
                            mtip: 0.0,
                            popid: s.popid,
                            satid: 0,
                            partid: 0,
                        });
                    }
                }
            }

            // Cleanup ephemeral device buffers
            self.free_mem(d_leaves);
            self.free_mem(d_star_to_leaf);
            self.free_mem(d_star_in_leaf);
            self.free_mem(d_imf_x);
            self.free_mem(d_imf_cpd);
            self.free_mem(d_out_stars);

            Ok(accepted_stars)
        }
    }
}

// Helpers for FP64
fn make_pop_params_f64(pop_id: usize, pop: &dyn Population, params: &Params) -> CudaPopParams {
    let mut p = CudaPopParams::default();
    p.pop_id = pop_id as i32;
    p.option_e = pop.option_e() as i32;
    p.warp_flare_on = params.warp_flare_on as i32;
    p.age = pop.age();
    p.dage = pop.dage();
    p.feh = pop.feh();
    p.dfeh = pop.dfeh();
    p.sigma_v = pop.sigma_v();

    let wf = params.warp_flare_on as usize;
    match pop_id {
        0 => {
            let td = crate::population::ThinDisk0::new(wf, pop.option_e());
            p.rho0 = td.rho0;
            p.hr1 = td.hr1;
            p.hr2 = td.hr2;
            p.r_warp = td.base.r_warp;
            p.r_flare = td.base.r_flare;
            p.g_warp = td.base.g_warp;
            p.g_flare = td.base.g_flare;
            p.th_max = td.base.th_max;
            p.qparam = td.base.qparam;
            p.bparam = td.base.bparam;
            p.agevec = td.base.agevec;
            p.fehvec = td.base.fehvec;
        }
        1..=6 => {
            let td = crate::population::ThinDisk1::new(pop_id, wf, pop.option_e());
            p.rho0 = td.rho0;
            p.hr1 = td.hr1;
            p.hr2 = td.hr2;
            p.r_warp = td.base.r_warp;
            p.r_flare = td.base.r_flare;
            p.g_warp = td.base.g_warp;
            p.g_flare = td.base.g_flare;
            p.th_max = td.base.th_max;
            p.qparam = td.base.qparam;
            p.bparam = td.base.bparam;
            p.agevec = td.base.agevec;
            p.fehvec = td.base.fehvec;
        }
        7 => {
            let td = crate::population::ThickDisk::new(wf);
            p.rho0 = td.rho0;
            p.hr1 = td.hr1;
            p.hr2 = td.hr2;
            p.xl = td.xl;
            p.r_warp = td.r_warp;
            p.r_flare = td.r_flare;
            p.g_warp = td.g_warp;
            p.g_flare = td.g_flare;
            p.th_max = td.th_max;
            p.qparam = td.qparam;
        }
        8 => {
            let sph = crate::population::Spheroid::new();
            p.rho0 = sph.rho0;
            p.rhoc = sph.rhoc;
            p.ac = sph.ac;
            p.eps = sph.eps;
        }
        9 => {
            let blg = crate::population::Bulge::new();
            p.rho0 = blg.rho0;
            p.x0 = blg.x0;
            p.y0 = blg.y0;
            p.z0 = blg.z0;
            p.rc_bulge = blg.rc;
            p.rot_matrix = blg.rot_matrix.m;
        }
        _ => {}
    }
    p
}

fn make_survey_params_f64(params: &Params) -> CudaSurveyParams {
    let survey = crate::survey::SurveyDesign::from_params(params);
    CudaSurveyParams {
        pos_c: params.pos_c,
        cone_dir: survey.patch.center_dir,
        cone_dth: survey.patch.d_th,
        app_mag_limits: params.app_mag_limits,
        color_limits: params.color_limits,
        abs_mag_limits: params.abs_mag_limits,
        r_max: params.r_max,
        seed: params.seed as u64,
        field_id: 0,
        pad: 0,
    }
}

// Helpers for FP32
fn make_pop_params_f32(pop_id: usize, pop: &dyn Population, params: &Params) -> ClPopParamsF32 {
    let p_f64 = make_pop_params_f64(pop_id, pop, params);
    let mut rot = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            rot[r][c] = p_f64.rot_matrix[r][c] as f32;
        }
    }
    let mut agevec = [0.0f32; 7];
    let mut fehvec = [0.0f32; 7];
    for k in 0..7 {
        agevec[k] = p_f64.agevec[k] as f32;
        fehvec[k] = p_f64.fehvec[k] as f32;
    }

    ClPopParamsF32 {
        pop_id: p_f64.pop_id,
        option_e: p_f64.option_e,
        warp_flare_on: p_f64.warp_flare_on,
        pad0: 0,
        age: p_f64.age as f32,
        dage: p_f64.dage as f32,
        feh: p_f64.feh as f32,
        dfeh: p_f64.dfeh as f32,
        sigma_v: [
            p_f64.sigma_v[0] as f32,
            p_f64.sigma_v[1] as f32,
            p_f64.sigma_v[2] as f32,
        ],
        pad1: 0.0,
        rho0: p_f64.rho0 as f32,
        hr1: p_f64.hr1 as f32,
        hr2: p_f64.hr2 as f32,
        xl: p_f64.xl as f32,
        r_warp: p_f64.r_warp as f32,
        r_flare: p_f64.r_flare as f32,
        g_warp: p_f64.g_warp as f32,
        g_flare: p_f64.g_flare as f32,
        th_max: p_f64.th_max as f32,
        qparam: p_f64.qparam as f32,
        bparam: p_f64.bparam as f32,
        ac: p_f64.ac as f32,
        rhoc: p_f64.rhoc as f32,
        eps: p_f64.eps as f32,
        x0: p_f64.x0 as f32,
        y0: p_f64.y0 as f32,
        z0: p_f64.z0 as f32,
        rc_bulge: p_f64.rc_bulge as f32,
        rot_matrix: rot,
        agevec,
        fehvec,
    }
}

fn make_survey_params_f32(params: &Params) -> ClSurveyParamsF32 {
    let s_f64 = make_survey_params_f64(params);
    ClSurveyParamsF32 {
        pos_c: [
            s_f64.pos_c[0] as f32,
            s_f64.pos_c[1] as f32,
            s_f64.pos_c[2] as f32,
            s_f64.pos_c[3] as f32,
            s_f64.pos_c[4] as f32,
            s_f64.pos_c[5] as f32,
        ],
        cone_dir: [
            s_f64.cone_dir[0] as f32,
            s_f64.cone_dir[1] as f32,
            s_f64.cone_dir[2] as f32,
        ],
        cone_dth: s_f64.cone_dth as f32,
        app_mag_limits: [
            s_f64.app_mag_limits[0] as f32,
            s_f64.app_mag_limits[1] as f32,
        ],
        color_limits: [s_f64.color_limits[0] as f32, s_f64.color_limits[1] as f32],
        abs_mag_limits: [
            s_f64.abs_mag_limits[0] as f32,
            s_f64.abs_mag_limits[1] as f32,
        ],
        r_max: s_f64.r_max as f32,
        seed: s_f64.seed,
        field_id: s_f64.field_id,
        pad: 0,
    }
}
