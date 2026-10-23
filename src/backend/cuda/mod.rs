// SPDX-License-Identifier: AGPL-3.0-or-later
//! CUDA Accelerator Backend using dynamic CUDA Driver API loading.
//!
//! **EXPERIMENTAL - NOT VALIDATED - NOT FOR SCIENCE.** Compiled only with the non-default
//! `cuda` cargo feature. Known open issues: rejection-loop cap behaviour, PTX/kernel handling,
//! resource leaks on error paths, and thread-safety assumptions of the raw driver handles.
//!
//! Executes `spawn_kernel` on NVIDIA GPU (sm_89 / sm_120 Ada/Blackwell).

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

// CUDA Driver API type definitions
type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;

/// Low-level dynamically loaded CUDA driver bindings.
pub struct CudaDriver {
    _lib: libloading::Library,
    cu_init: unsafe extern "C" fn(flags: u32) -> CUresult,
    cu_device_get: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    cu_device_get_name: unsafe extern "C" fn(name: *mut i8, len: i32, dev: CUdevice) -> CUresult,
    cu_ctx_create_v2:
        unsafe extern "C" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult,
    cu_ctx_destroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cu_module_load_data:
        unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    cu_module_unload: unsafe extern "C" fn(hmod: CUmodule) -> CUresult,
    cu_module_get_function:
        unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> CUresult,
    cu_mem_alloc_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    cu_mem_free_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    cu_memcpy_htod_v2:
        unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult,
    cu_memcpy_dtoh_v2:
        unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult,
    cu_launch_kernel: unsafe extern "C" fn(
        f: CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        h_stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,
}

impl CudaDriver {
    pub fn load() -> anyhow::Result<Self> {
        let candidates = [
            "nvcuda.dll",
            "libcuda.so.1",
            "libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        ];
        let mut loaded = None;
        for c in &candidates {
            if let Ok(lib) = unsafe { libloading::Library::new(*c) } {
                loaded = Some(lib);
                break;
            }
        }

        let lib =
            loaded.ok_or_else(|| anyhow::anyhow!("libcuda.so.1 dynamic library not found"))?;

        unsafe {
            let cu_init = *lib.get(b"cuInit\0")?;
            let cu_device_get = *lib.get(b"cuDeviceGet\0")?;
            let cu_device_get_name = *lib.get(b"cuDeviceGetName\0")?;
            let cu_ctx_create_v2 = *lib.get(b"cuCtxCreate_v2\0")?;
            let cu_ctx_destroy_v2 = *lib.get(b"cuCtxDestroy_v2\0")?;
            let cu_module_load_data = *lib.get(b"cuModuleLoadData\0")?;
            let cu_module_unload = *lib.get(b"cuModuleUnload\0")?;
            let cu_module_get_function = *lib.get(b"cuModuleGetFunction\0")?;
            let cu_mem_alloc_v2 = *lib.get(b"cuMemAlloc_v2\0")?;
            let cu_mem_free_v2 = *lib.get(b"cuMemFree_v2\0")?;
            let cu_memcpy_htod_v2 = *lib.get(b"cuMemcpyHtoD_v2\0")?;
            let cu_memcpy_dtoh_v2 = *lib.get(b"cuMemcpyDtoH_v2\0")?;
            let cu_launch_kernel = *lib.get(b"cuLaunchKernel\0")?;
            let cu_ctx_synchronize = *lib.get(b"cuCtxSynchronize\0")?;

            Ok(Self {
                _lib: lib,
                cu_init,
                cu_device_get,
                cu_device_get_name,
                cu_ctx_create_v2,
                cu_ctx_destroy_v2,
                cu_module_load_data,
                cu_module_unload,
                cu_module_get_function,
                cu_mem_alloc_v2,
                cu_mem_free_v2,
                cu_memcpy_htod_v2,
                cu_memcpy_dtoh_v2,
                cu_launch_kernel,
                cu_ctx_synchronize,
            })
        }
    }
}

// Embedded PTX fallback (pre-compiled PTX for CUDA devices)
const EMBEDDED_PTX: &str = include_str!("kernels/spawn.ptx");

/// Resident GPU memory buffers for shared tables (Isochrones and VCirc).
struct ResidentTables {
    d_isochrone_m: CUdeviceptr,
    d_isochrone_mag0: CUdeviceptr,
    d_isochrone_mag1: CUdeviceptr,
    d_isochrone_mag2: CUdeviceptr,
    d_track_offsets: CUdeviceptr,
    d_track_m_min: CUdeviceptr,
    d_track_m_max: CUdeviceptr,
    d_feh_grid: CUdeviceptr,
    num_ages: u32,
    num_feh: u32,
    age_grid_min: f64,
    d_age: f64,
    d_vcirc_r: CUdeviceptr,
    d_vcirc_v: CUdeviceptr,
    vcirc_size: u32,
}

/// CUDA backend implementing `AcceleratorBackend`.
pub struct CudaBackend {
    driver: Arc<CudaDriver>,
    ctx: CUcontext,
    module: CUmodule,
    kernel: CUfunction,
    device_name: String,
    resident: Mutex<Option<ResidentTables>>,
}

// Safety: CUcontext and CUmodule are managed safely across threads with synchronization.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// Initialize the CUDA driver, device 0, create a context, and load `spawn_kernel`.
    pub fn new() -> anyhow::Result<Self> {
        let driver = Arc::new(CudaDriver::load()?);

        unsafe {
            let res = (driver.cu_init)(0);
            if res != CUDA_SUCCESS {
                return Err(anyhow::anyhow!("cuInit failed with code {}", res));
            }

            let mut dev: CUdevice = 0;
            let res = (driver.cu_device_get)(&mut dev, 0);
            if res != CUDA_SUCCESS {
                return Err(anyhow::anyhow!("cuDeviceGet(0) failed with code {}", res));
            }

            let mut name_buf = [0i8; 256];
            (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev);
            let device_name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let mut ctx: CUcontext = std::ptr::null_mut();
            let res = (driver.cu_ctx_create_v2)(&mut ctx, 0, dev);
            if res != CUDA_SUCCESS {
                return Err(anyhow::anyhow!("cuCtxCreate failed with code {}", res));
            }

            // Load PTX: try external file first, fallback to embedded
            let mut ptx_string = if let Ok(s) = std::fs::read_to_string("target/ptx/spawn.ptx") {
                s
            } else {
                EMBEDDED_PTX.to_string()
            };

            // Patch PTX version if needed for driver compatibility (e.g. 9.3 -> 9.2)
            if ptx_string.contains(".version 9.3") {
                ptx_string = ptx_string.replace(".version 9.3", ".version 9.2");
            }

            let mut ptx_bytes = ptx_string.into_bytes();
            ptx_bytes.push(0); // null terminate

            let mut module: CUmodule = std::ptr::null_mut();
            let res =
                (driver.cu_module_load_data)(&mut module, ptx_bytes.as_ptr() as *const c_void);
            if res != CUDA_SUCCESS {
                (driver.cu_ctx_destroy_v2)(ctx);
                return Err(anyhow::anyhow!("cuModuleLoadData failed with code {}", res));
            }

            let kernel_name = CString::new("spawn_kernel").unwrap();
            let mut kernel: CUfunction = std::ptr::null_mut();
            let res = (driver.cu_module_get_function)(&mut kernel, module, kernel_name.as_ptr());
            if res != CUDA_SUCCESS {
                (driver.cu_module_unload)(module);
                (driver.cu_ctx_destroy_v2)(ctx);
                return Err(anyhow::anyhow!(
                    "cuModuleGetFunction(spawn_kernel) failed with code {}",
                    res
                ));
            }

            Ok(Self {
                driver,
                ctx,
                module,
                kernel,
                device_name,
                resident: Mutex::new(None),
            })
        }
    }

    /// Allocate GPU device memory and copy host slice to device.
    unsafe fn upload_slice<T>(&self, data: &[T]) -> anyhow::Result<CUdeviceptr> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(0);
        }
        let mut dptr: CUdeviceptr = 0;
        let res = (self.driver.cu_mem_alloc_v2)(&mut dptr, bytes);
        if res != CUDA_SUCCESS {
            return Err(anyhow::anyhow!("cuMemAlloc failed: {}", res));
        }
        let res = (self.driver.cu_memcpy_htod_v2)(dptr, data.as_ptr() as *const c_void, bytes);
        if res != CUDA_SUCCESS {
            (self.driver.cu_mem_free_v2)(dptr);
            return Err(anyhow::anyhow!("cuMemcpyHtoD failed: {}", res));
        }
        Ok(dptr)
    }

    /// Free allocated GPU memory.
    unsafe fn free_device(&self, dptr: CUdeviceptr) {
        if dptr != 0 {
            (self.driver.cu_mem_free_v2)(dptr);
        }
    }

    /// Ensure resident database tables (Isochrones and VCirc) are resident in GPU memory.
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
        let mut track_m_min = Vec::with_capacity(total_tracks);
        let mut track_m_max = Vec::with_capacity(total_tracks);

        let mut isochrone_m = Vec::new();
        let mut isochrone_mag0 = Vec::new();
        let mut isochrone_mag1 = Vec::new();
        let mut isochrone_mag2 = Vec::new();

        let mut cur_offset = 0u32;
        track_offsets.push(cur_offset);

        for j in 0..num_feh as usize {
            for i in 0..num_ages as usize {
                let track_idx = j * (num_ages as usize) + i;
                if track_idx < isochrones.tracks.len() {
                    let track = &isochrones.tracks[track_idx];
                    let count = track.m.len() as u32;
                    cur_offset += count;
                    track_offsets.push(cur_offset);
                    track_m_min.push(track.m_min);
                    track_m_max.push(*track.m.last().unwrap_or(&100.0));

                    isochrone_m.extend_from_slice(&track.m);
                    isochrone_mag0.extend_from_slice(&track.mag0);
                    isochrone_mag1.extend_from_slice(&track.mag1);
                    isochrone_mag2.extend_from_slice(&track.mag2);
                } else {
                    track_offsets.push(cur_offset);
                    track_m_min.push(100.0);
                    track_m_max.push(0.0);
                }
            }
        }

        let age_grid_min = isochrones.age_grid[0];
        let d_age = isochrones.d_age;

        unsafe {
            let d_isochrone_m = self.upload_slice(&isochrone_m)?;
            let d_isochrone_mag0 = self.upload_slice(&isochrone_mag0)?;
            let d_isochrone_mag1 = self.upload_slice(&isochrone_mag1)?;
            let d_isochrone_mag2 = self.upload_slice(&isochrone_mag2)?;
            let d_track_offsets = self.upload_slice(&track_offsets)?;
            let d_track_m_min = self.upload_slice(&track_m_min)?;
            let d_track_m_max = self.upload_slice(&track_m_max)?;
            let d_feh_grid = self.upload_slice(&isochrones.feh_grid)?;

            let d_vcirc_r = self.upload_slice(&vcirc.r)?;
            let d_vcirc_v = self.upload_slice(&vcirc.v)?;

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
                age_grid_min,
                d_age,
                d_vcirc_r,
                d_vcirc_v,
                vcirc_size: vcirc.r.len() as u32,
            });
        }

        Ok(())
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            if let Some(res) = self.resident.lock().unwrap().take() {
                self.free_device(res.d_isochrone_m);
                self.free_device(res.d_isochrone_mag0);
                self.free_device(res.d_isochrone_mag1);
                self.free_device(res.d_isochrone_mag2);
                self.free_device(res.d_track_offsets);
                self.free_device(res.d_track_m_min);
                self.free_device(res.d_track_m_max);
                self.free_device(res.d_feh_grid);
                self.free_device(res.d_vcirc_r);
                self.free_device(res.d_vcirc_v);
            }
            if !self.module.is_null() {
                (self.driver.cu_module_unload)(self.module);
            }
            if !self.ctx.is_null() {
                (self.driver.cu_ctx_destroy_v2)(self.ctx);
            }
        }
    }
}

impl AcceleratorBackend for CudaBackend {
    fn name(&self) -> &str {
        &self.device_name
    }

    fn platform_tag(&self) -> String {
        format!("galaxia-rs-{}-cuda", env!("CARGO_PKG_VERSION"))
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

        // Flatten leaf and star mapping arrays
        let mut leaves_cuda = Vec::with_capacity(active_leaves.len());
        let mut star_to_leaf = Vec::with_capacity(total_stars);
        let mut star_in_leaf = Vec::with_capacity(total_stars);

        for (l_idx, leaf) in active_leaves.iter().enumerate() {
            leaves_cuda.push(CudaActiveLeaf {
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

            for s_idx in 0..leaf.n_tot {
                star_to_leaf.push(l_idx as u32);
                star_in_leaf.push(s_idx as u32);
            }
        }

        let pop_params = make_pop_params(pop_id, pop, params);
        let survey_params = make_survey_params(params);

        unsafe {
            let d_leaves = self.upload_slice(&leaves_cuda)?;
            let d_star_to_leaf = self.upload_slice(&star_to_leaf)?;
            let d_star_in_leaf = self.upload_slice(&star_in_leaf)?;
            let d_imf_x = self.upload_slice(&imf.x)?;
            let d_imf_cpd = self.upload_slice(&imf.cpd)?;

            let out_bytes = total_stars * std::mem::size_of::<CudaOutputStar>();
            let mut d_out_stars: CUdeviceptr = 0;
            let status = (self.driver.cu_mem_alloc_v2)(&mut d_out_stars, out_bytes);
            if status != CUDA_SUCCESS {
                self.free_device(d_leaves);
                self.free_device(d_star_to_leaf);
                self.free_device(d_star_in_leaf);
                self.free_device(d_imf_x);
                self.free_device(d_imf_cpd);
                return Err(anyhow::anyhow!(
                    "cuMemAlloc output stars failed: {}",
                    status
                ));
            }

            let imf_size = imf.x.len() as u32;
            let num_ages = res.num_ages;
            let num_feh = res.num_feh;
            let age_grid_min = res.age_grid_min;
            let d_age = res.d_age;
            let vcirc_size = res.vcirc_size;
            let total_stars_u32 = total_stars as u32;

            // Prepare kernel arguments
            let mut kernel_args: [*mut c_void; 25] = [
                &pop_params as *const _ as *mut c_void,
                &survey_params as *const _ as *mut c_void,
                &d_leaves as *const _ as *mut c_void,
                &d_star_to_leaf as *const _ as *mut c_void,
                &d_star_in_leaf as *const _ as *mut c_void,
                &d_imf_x as *const _ as *mut c_void,
                &d_imf_cpd as *const _ as *mut c_void,
                &imf_size as *const _ as *mut c_void,
                &res.d_isochrone_m as *const _ as *mut c_void,
                &res.d_isochrone_mag0 as *const _ as *mut c_void,
                &res.d_isochrone_mag1 as *const _ as *mut c_void,
                &res.d_isochrone_mag2 as *const _ as *mut c_void,
                &res.d_track_offsets as *const _ as *mut c_void,
                &res.d_track_m_min as *const _ as *mut c_void,
                &res.d_track_m_max as *const _ as *mut c_void,
                &num_ages as *const _ as *mut c_void,
                &num_feh as *const _ as *mut c_void,
                &age_grid_min as *const _ as *mut c_void,
                &d_age as *const _ as *mut c_void,
                &res.d_feh_grid as *const _ as *mut c_void,
                &res.d_vcirc_r as *const _ as *mut c_void,
                &res.d_vcirc_v as *const _ as *mut c_void,
                &vcirc_size as *const _ as *mut c_void,
                &d_out_stars as *const _ as *mut c_void,
                &total_stars_u32 as *const _ as *mut c_void,
            ];

            let block_size = 256u32;
            let grid_size = ((total_stars as u32) + block_size - 1) / block_size;

            let launch_res = (self.driver.cu_launch_kernel)(
                self.kernel,
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                kernel_args.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if launch_res != CUDA_SUCCESS {
                self.free_device(d_leaves);
                self.free_device(d_star_to_leaf);
                self.free_device(d_star_in_leaf);
                self.free_device(d_imf_x);
                self.free_device(d_imf_cpd);
                self.free_device(d_out_stars);
                return Err(anyhow::anyhow!(
                    "cuLaunchKernel failed with code {}",
                    launch_res
                ));
            }

            let sync_res = (self.driver.cu_ctx_synchronize)();
            if sync_res != CUDA_SUCCESS {
                self.free_device(d_leaves);
                self.free_device(d_star_to_leaf);
                self.free_device(d_star_in_leaf);
                self.free_device(d_imf_x);
                self.free_device(d_imf_cpd);
                self.free_device(d_out_stars);
                return Err(anyhow::anyhow!(
                    "cuCtxSynchronize failed with code {}",
                    sync_res
                ));
            }

            let mut out_stars_host: Vec<CudaOutputStar> =
                vec![CudaOutputStar::default(); total_stars];
            let copy_res = (self.driver.cu_memcpy_dtoh_v2)(
                out_stars_host.as_mut_ptr() as *mut c_void,
                d_out_stars,
                out_bytes,
            );

            // Cleanup ephemeral device memory
            self.free_device(d_leaves);
            self.free_device(d_star_to_leaf);
            self.free_device(d_star_in_leaf);
            self.free_device(d_imf_x);
            self.free_device(d_imf_cpd);
            self.free_device(d_out_stars);

            if copy_res != CUDA_SUCCESS {
                return Err(anyhow::anyhow!(
                    "cuMemcpyDtoH failed with code {}",
                    copy_res
                ));
            }

            // Filter accepted stars
            let mut accepted_stars = Vec::new();
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

            Ok(accepted_stars)
        }
    }
}

fn make_pop_params(pop_id: usize, pop: &dyn Population, params: &Params) -> CudaPopParams {
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

fn make_survey_params(params: &Params) -> CudaSurveyParams {
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
