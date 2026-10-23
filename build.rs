// SPDX-License-Identifier: AGPL-3.0-or-later
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate `nvcc`: the `NVCC` environment variable if set, otherwise `nvcc` from `PATH`.
fn find_nvcc() -> Option<PathBuf> {
    if let Ok(path) = env::var("NVCC") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let ok = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if ok {
        Some(PathBuf::from("nvcc"))
    } else {
        None
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/backend/cuda/kernels/spawn.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/");

    // Capture git commit hash and dirty status for output provenance
    let git_commit = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        });

    let is_dirty = Command::new("git")
        .args(&["status", "--porcelain"])
        .output()
        .ok()
        .map(|out| out.status.success() && !out.stdout.is_empty())
        .unwrap_or(false);

    let full_commit = match git_commit {
        Some(hash) => {
            if is_dirty {
                format!("{}-dirty", hash)
            } else {
                hash
            }
        }
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=GIT_COMMIT={}", full_commit);

    let cuda_feature = env::var("CARGO_FEATURE_CUDA").is_ok();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cu_path = Path::new(&manifest_dir).join("src/backend/cuda/kernels/spawn.cu");
    let out_dir = Path::new(&manifest_dir).join("target/ptx");
    let ptx_path = out_dir.join("spawn.ptx");

    if !cu_path.exists() {
        return;
    }

    if cuda_feature {
        if let Some(nvcc) = find_nvcc() {
            std::fs::create_dir_all(&out_dir).ok();
            println!(
                "cargo:warning=Compiling {} to {} with {:?}",
                cu_path.display(),
                ptx_path.display(),
                nvcc
            );
            let arch_flag = format!(
                "--gpu-architecture={}",
                env::var("GALAXIA_CUDA_ARCH").unwrap_or_else(|_| "compute_89".to_string())
            );
            let status = Command::new(&nvcc)
                .args(&[
                    "-ptx",
                    arch_flag.as_str(),
                    "-o",
                    ptx_path.to_str().unwrap(),
                    cu_path.to_str().unwrap(),
                ])
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!(
                        "cargo:warning=Successfully compiled CUDA PTX to {}",
                        ptx_path.display()
                    );
                    // Patch .version 9.3 -> .version 9.2 for driver compatibility with some CUDA 13.x drivers
                    if let Ok(content) = std::fs::read_to_string(&ptx_path) {
                        if content.contains(".version 9.3") {
                            let patched = content.replace(".version 9.3", ".version 9.2");
                            let _ = std::fs::write(&ptx_path, patched);
                        }
                    }
                }
                Ok(s) => {
                    println!("cargo:warning=nvcc compilation exited with status: {}", s);
                }
                Err(e) => {
                    println!("cargo:warning=Failed to execute nvcc: {}", e);
                }
            }
        } else {
            if ptx_path.exists() {
                println!(
                    "cargo:warning=nvcc not found; using existing precompiled PTX at {}",
                    ptx_path.display()
                );
            } else {
                println!("cargo:warning=nvcc not found and no precompiled PTX present.");
            }
        }
    }
}
