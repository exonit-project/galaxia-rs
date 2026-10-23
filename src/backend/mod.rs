// SPDX-License-Identifier: AGPL-3.0-or-later
//! Accelerator backends for Galaxia star generation.
//!
//! Only the CPU fp64 backend is validated and supported. The CUDA and OpenCL
//! backends are EXPERIMENTAL (single-precision or unvalidated, not for science)
//! and are compiled only with the non-default `cuda` / `opencl` cargo features.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;

use crate::imf::Sampler;
use crate::interp::VCirc;
use crate::isochrone::IsochroneDB;
use crate::params::Params;
use crate::population::{Population, StarParticle};

pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
#[cfg(feature = "opencl")]
pub use opencl::OpenClBackend;

/// Information for an active BHTree leaf node passing survey cone cuts.
#[derive(Debug, Clone, Default)]
pub struct ActiveLeaf {
    pub orig_leaf_idx: usize,
    pub pos: [f64; 4],
    pub l: f64,
    pub lage: f64,
    pub rho_max: f64,
    pub rho_min: f64,
    pub dmod: f64,
    pub mass: f64,
    pub box_mmin: f64,
    pub box_mmax: f64,
    pub cpd_min: f64,
    pub cpd_max: f64,
    pub n_tot: usize,
    pub star_offset: usize,
}

/// GPU-compatible POD representation of an active leaf.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaActiveLeaf {
    pub pos: [f64; 4],
    pub l: f64,
    pub lage: f64,
    pub rho_max: f64,
    pub rho_min: f64,
    pub dmod: f64,
    pub box_mmin: f64,
    pub box_mmax: f64,
    pub cpd_min: f64,
    pub cpd_max: f64,
    pub orig_leaf_idx: u32,
    pub n_stars: u32,
    pub star_offset: u32,
    pub pad: u32,
}

/// GPU-compatible POD representation of population parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaPopParams {
    pub pop_id: i32,
    pub option_e: i32,
    pub warp_flare_on: i32,
    pub pad0: i32,
    pub age: f64,
    pub dage: f64,
    pub feh: f64,
    pub dfeh: f64,
    pub sigma_v: [f64; 3],
    pub pad1: f64,
    pub rho0: f64,
    pub hr1: f64,
    pub hr2: f64,
    pub xl: f64,
    pub r_warp: f64,
    pub r_flare: f64,
    pub g_warp: f64,
    pub g_flare: f64,
    pub th_max: f64,
    pub qparam: f64,
    pub bparam: f64,
    pub ac: f64,
    pub rhoc: f64,
    pub eps: f64,
    pub x0: f64,
    pub y0: f64,
    pub z0: f64,
    pub rc_bulge: f64,
    pub rot_matrix: [[f64; 3]; 3],
    pub agevec: [f64; 7],
    pub fehvec: [f64; 7],
}

impl Default for CudaPopParams {
    fn default() -> Self {
        Self {
            pop_id: 0,
            option_e: 0,
            warp_flare_on: 1,
            pad0: 0,
            age: 0.0,
            dage: 0.0,
            feh: 0.0,
            dfeh: 0.0,
            sigma_v: [0.0; 3],
            pad1: 0.0,
            rho0: 0.0,
            hr1: 0.0,
            hr2: 0.0,
            xl: 0.0,
            r_warp: 0.0,
            r_flare: 0.0,
            g_warp: 0.0,
            g_flare: 0.0,
            th_max: 0.0,
            qparam: 0.0,
            bparam: 0.0,
            ac: 0.0,
            rhoc: 0.0,
            eps: 0.0,
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            rc_bulge: 0.0,
            rot_matrix: [[0.0; 3]; 3],
            agevec: [0.0; 7],
            fehvec: [0.0; 7],
        }
    }
}

/// GPU-compatible POD representation of survey design and observational cuts.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaSurveyParams {
    pub pos_c: [f64; 6],
    pub cone_dir: [f64; 3],
    pub cone_dth: f64,
    pub app_mag_limits: [f64; 2],
    pub color_limits: [f64; 2],
    pub abs_mag_limits: [f64; 2],
    pub r_max: f64,
    pub seed: u64,
    pub field_id: u32,
    pub pad: u32,
}

/// GPU-compatible POD output structure for generated star particles.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaOutputStar {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub age: f64,
    pub feh: f64,
    pub smass: f64,
    pub mags: [f64; 3],
    pub abs_mags: [f64; 3],
    pub rad: f64,
    pub dmod: f64,
    pub popid: i32,
    pub accepted: i32,
}

/// Abstract accelerator backend trait implemented by `CpuBackend` (and, experimentally, the GPU backends).
pub trait AcceleratorBackend: Send + Sync {
    /// Human-readable backend description.
    fn name(&self) -> &str;

    /// Platform/version tag for reproducibility and database metadata.
    fn platform_tag(&self) -> String;

    /// Whether this backend produces production-grade (FP64 bit-compatible) catalogs.
    fn is_production_grade(&self) -> bool {
        true
    }

    /// Spawn stars for a given population and set of active leaves.
    fn spawn_stars(
        &self,
        pop_id: usize,
        pop: &dyn Population,
        params: &Params,
        active_leaves: &[ActiveLeaf],
        imf: &Sampler,
        isochrones: &IsochroneDB,
        vcirc: &VCirc,
    ) -> anyhow::Result<Vec<StarParticle>>;
}

/// Accelerator backend kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    OpenCl,
    Cpu,
}

impl std::str::FromStr for BackendKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cuda" | "nvidia" => Ok(BackendKind::Cuda),
            "opencl" | "cl" | "intel" | "igpu" => Ok(BackendKind::OpenCl),
            "cpu" => Ok(BackendKind::Cpu),
            "gpu" => {
                // If generic 'gpu' requested, try CUDA first, then OpenCL
                if std::path::Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so.1").exists()
                    || std::path::Path::new("/usr/lib/wsl/lib/libcuda.so.1").exists()
                    || std::path::Path::new("C:\\Windows\\System32\\nvcuda.dll").exists()
                {
                    Ok(BackendKind::Cuda)
                } else {
                    Ok(BackendKind::OpenCl)
                }
            }
            other => Err(anyhow::anyhow!(
                "Unknown backend '{}', expected 'cuda', 'opencl' (or 'intel'), or 'cpu'",
                other
            )),
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Cuda => write!(f, "cuda"),
            BackendKind::OpenCl => write!(f, "opencl"),
            BackendKind::Cpu => write!(f, "cpu"),
        }
    }
}

/// Version string with crate version, git commit, backend tag, and tmag ceiling.
pub fn version_string(backend: BackendKind, tmag_ceiling: Option<f64>) -> String {
    let backend_str = match backend {
        BackendKind::Cpu => "cpu-fp64",
        BackendKind::Cuda => "cuda",
        BackendKind::OpenCl => "opencl",
    };
    let ceiling_str = match tmag_ceiling {
        Some(c) => format!("{:.6}", c),
        None => "none".to_string(),
    };
    format!(
        "galaxia-rs {} (git {}) backend={} tmag_ceiling={}",
        env!("CARGO_PKG_VERSION"),
        env!("GIT_COMMIT"),
        backend_str,
        ceiling_str
    )
}

/// Initialize the requested accelerator backend.
///
/// No silent fallback from a GPU backend to the CPU is performed: if a GPU
/// backend is requested but unavailable (not compiled in, or initialization
/// fails), this returns an error. The CPU backend is used only when requested.
pub fn create_backend(kind: BackendKind) -> anyhow::Result<Box<dyn AcceleratorBackend>> {
    match kind {
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let cuda = CudaBackend::new().map_err(|e| {
                    anyhow::anyhow!(
                        "EXPERIMENTAL CUDA backend initialization failed: {}. \
                         No silent fallback to CPU is performed; \
                         use '--backend cpu' if CPU execution is desired.",
                        e
                    )
                })?;
                println!(
                    "Accelerator backend: {} (EXPERIMENTAL, not for science)",
                    cuda.name()
                );
                Ok(Box::new(cuda))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow::anyhow!(
                    "CUDA backend was not compiled in (build with `--features cuda`; EXPERIMENTAL). \
                     No silent fallback to CPU is performed; use '--backend cpu'."
                ))
            }
        }
        BackendKind::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                let cl = OpenClBackend::new().map_err(|e| {
                    anyhow::anyhow!(
                        "EXPERIMENTAL OpenCL backend initialization failed: {}. \
                         No silent fallback to CPU is performed; \
                         use '--backend cpu' if CPU execution is desired.",
                        e
                    )
                })?;
                println!(
                    "Accelerator backend: {} (EXPERIMENTAL, not for science)",
                    cl.name()
                );
                Ok(Box::new(cl))
            }
            #[cfg(not(feature = "opencl"))]
            {
                Err(anyhow::anyhow!(
                    "OpenCL backend was not compiled in (build with `--features opencl`; EXPERIMENTAL). \
                     No silent fallback to CPU is performed; use '--backend cpu'."
                ))
            }
        }
        BackendKind::Cpu => {
            let cpu = CpuBackend::new();
            println!("Accelerator backend: {}", cpu.name());
            Ok(Box::new(cpu))
        }
    }
}
