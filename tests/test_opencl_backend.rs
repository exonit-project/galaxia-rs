#![cfg(feature = "opencl")]
// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::backend::{AcceleratorBackend, BackendKind, OpenClBackend};
use galaxia_rs::generator::run_model;
use galaxia_rs::params::Params;
use std::path::Path;

#[test]
fn test_opencl_backend_initialization() {
    match OpenClBackend::new() {
        Ok(cl) => {
            println!("OpenCL GPU initialized successfully: {}", cl.name());
            assert!(!cl.name().is_empty());
        }
        Err(e) => {
            println!("OpenCL GPU not available on this host: {}", e);
        }
    }
}

#[test]
fn test_opencl_execution_if_available() {
    let cl_res = OpenClBackend::new();
    if let Err(e) = cl_res {
        eprintln!("OpenCL GPU not available: {}, skipping execution test", e);
        return;
    }
    let cl = cl_res.unwrap();
    println!("Running OpenCL GPU test on {}", cl.name());

    let valid_data_dir = galaxia_rs::require_data_dir!();
    let par_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/t.par");

    let mut params = Params::from_file(&par_file).expect("Parse t.par");
    params.code_data_dir = format!("{}/", valid_data_dir.display());
    let temp_dir = std::env::temp_dir();
    params.output_dir = format!("{}/", temp_dir.display());
    params.output_file = "test_opencl_output".to_string();

    let stats = run_model(&params, BackendKind::OpenCl).expect("Run model on OpenCL GPU");
    println!("OpenCL GPU generated {} stars", stats.total_stars);
    assert!(
        stats.total_stars > 30000 && stats.total_stars < 35000,
        "Expected ~32,900 stars, got {}",
        stats.total_stars
    );

    let out_file = params.output_ebf_path();
    let _ = std::fs::remove_file(out_file);
}
