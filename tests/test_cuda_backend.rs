#![cfg(feature = "cuda")]
// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::backend::{AcceleratorBackend, CpuBackend, CudaBackend};
use galaxia_rs::generator::Generator;
use galaxia_rs::params::Params;
use std::path::Path;

#[test]
fn test_cuda_backend_initialization_and_fallback() {
    match CudaBackend::new() {
        Ok(cuda) => {
            println!("CUDA GPU initialized successfully: {}", cuda.name());
            assert!(!cuda.name().is_empty());
        }
        Err(e) => {
            println!(
                "CUDA not available on this host: {}, testing CPU fallback",
                e
            );
            let cpu = CpuBackend::new();
            assert_eq!(cpu.name(), "CPU Backend");
        }
    }
}

#[test]
fn test_cuda_execution_if_available() {
    let cuda_res = CudaBackend::new();
    if let Err(e) = cuda_res {
        eprintln!("CUDA not available: {}, skipping GPU execution test", e);
        return;
    }
    let cuda = cuda_res.unwrap();
    println!("Running GPU test on {}", cuda.name());

    let valid_data_dir = galaxia_rs::require_data_dir!();
    let par_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/t.par");

    let mut params = Params::from_file(&par_file).expect("Parse t.par");
    params.code_data_dir = valid_data_dir.to_str().unwrap().to_string();
    params.pop_id = 0; // Test pop 0 (ThinDisk0) on GPU
    let temp_dir = std::env::temp_dir();
    params.output_dir = format!("{}/", temp_dir.display());
    params.output_file = "test_cuda_pop0".to_string();

    let generator = Generator::with_backend(valid_data_dir, Box::new(cuda))
        .expect("Create generator with CUDA backend");

    let stats = generator.run_model(&params).expect("Run model on CUDA GPU");
    println!("CUDA GPU generated {} stars for pop 0", stats.total_stars);
    assert!(
        stats.total_stars > 0,
        "GPU should have generated stars for pop 0"
    );

    let out_file = params.output_ebf_path();
    let _ = std::fs::remove_file(out_file);
}
