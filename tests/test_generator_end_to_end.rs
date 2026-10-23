// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::backend::BackendKind;
use galaxia_rs::ebf::EbfFile;
use galaxia_rs::generator::run_model;
use galaxia_rs::params::Params;
use std::path::Path;

#[test]
fn test_generator_end_to_end_on_t_par() {
    let data_dir = galaxia_rs::require_data_dir!();
    let par_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/t.par");
    assert!(
        par_file.exists(),
        "Parameter file not found: {:?}",
        par_file
    );

    let mut params = Params::from_file(&par_file).expect("Failed to parse tests/fixtures/t.par");
    params.code_data_dir = format!("{}/", data_dir.display());
    // Write test output to dedicated temporary path
    let temp_dir = std::env::temp_dir();
    params.output_dir = format!("{}/", temp_dir.display());
    params.output_file = "gx_test_e2e".to_string();

    let stats = run_model(&params, BackendKind::Cpu).expect("Failed to run generation pipeline");

    println!("E2E test generated {} stars", stats.total_stars);
    assert!(
        stats.total_stars > 30000 && stats.total_stars < 35000,
        "Expected ~32,900 stars, got {}",
        stats.total_stars
    );

    // Verify written EBF file
    let out_file = params.output_ebf_path();
    let mut reader = EbfFile::open(&out_file).expect("Failed to open generated EBF file");

    let expected_keys = [
        "/px",
        "/py",
        "/pz",
        "/vx",
        "/vy",
        "/vz",
        "/feh",
        "/alpha",
        "/smass",
        "/age",
        "/rad",
        "/mag0",
        "/mag1",
        "/mag2",
        "/popid",
        "/satid",
        "/fieldid",
        "/partid",
        "/center",
        "/lum",
        "/teff",
        "/grav",
        "/mact",
        "/mtip",
        "/dcmc_i",
        "/dcmc_j",
        "/dcmc_h",
        "/dcmc_ks",
        "/exbv_schlegel",
        "/exbv_solar",
        "/exbv_schlegel_inf",
        "/glon",
        "/glat",
    ];

    for key in &expected_keys {
        assert!(
            reader.contains_key(key),
            "Generated EBF missing required key: {}",
            key
        );
    }

    let px = reader.read_f32("/px").expect("Read /px");
    assert_eq!(px.len(), stats.total_stars);

    let popid = reader.read_i32("/popid").expect("Read /popid");
    assert_eq!(popid.len(), stats.total_stars);

    let center = reader.read_f64("/center").expect("Read /center");
    assert_eq!(center.len(), 6);
    assert_eq!(center, params.pos_c);

    // Clean up
    let _ = std::fs::remove_file(out_file);
}
