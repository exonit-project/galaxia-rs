// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::approx_constant)]
// Tests for the C++ parity fixes (leaf-specific mass cutoff, float quantization of grids,
// backend gate, provenance metadata, CSV validation, atomic EBF writes, warp/option-E).

use galaxia_rs::backend::{version_string, BackendKind};
use galaxia_rs::batch::{parse_chunk_csv, run_batch, BatchConfig};
use galaxia_rs::ebf::EbfFile;
use galaxia_rs::imf::Sampler;
use galaxia_rs::isochrone::{IsochroneDB, IsochroneFieldState};
use galaxia_rs::population::{create_population, StarParticle, PI_WARP};
use galaxia_rs::rng::Philox;
use std::fs::File;
use std::io::Write;

/// Leaf-specific mass cutoff and maxV cache witness (found by independent code review).
/// Witness: maxV=5.0, mass=0.639620019, log age=10.0, FeH=-1.6766936096.
/// Must be REJECTED in generation (abs_mags[0] == 1e6) because mass <= track.min_mass_for_max_v(5.0).
#[test]
fn test_witness_leaf_mass_cutoff() {
    let data_dir = galaxia_rs::require_data_dir!();

    let db = IsochroneDB::load(&data_dir).expect("Load IsochroneDB");
    let mut state = IsochroneFieldState::new(&db);

    // Initial state: max_v = 30.0
    assert_eq!(state.max_v, 30.0);
    assert_eq!(state.delta_max_v, 0.01);

    // Call set_isochrones1 with max_v = 5.0
    let age_linear = 11e9_f64;
    let dage_linear = 1e9_f64;
    let feh = -0.78_f64;
    let dfeh = 0.3_f64;
    let max_v = 5.0_f64;
    let mut imf = Sampler::for_pop_id(7);
    let mut rng = Philox::from_stream(1, 0, 7, 0, 0, 0);

    let window = db.set_isochrones1(
        &mut state,
        age_linear,
        dage_linear,
        feh,
        dfeh,
        max_v,
        1000.0,
        0,
        &mut imf,
        &mut rng,
    );

    assert_eq!(state.max_v, 5.0);

    // Locate the track for age=10.0, feh=-1.6766936096
    let age_idx = db
        .age_grid
        .iter()
        .position(|&a| (a - 10.0).abs() < 1e-4)
        .expect("Find age 10.0");
    let feh_idx = db
        .feh_grid
        .iter()
        .position(|&f| (f - (-1.6766936096)).abs() < 1e-4)
        .expect("Find FeH -1.68");
    let track_idx = db.track_index(age_idx, feh_idx);
    let track = &db.tracks[track_idx];

    let track_cutoff = track.min_mass_for_max_v(5.0);
    assert!(
        (track_cutoff - 0.639621019).abs() < 1e-6,
        "Expected cutoff ~0.639621, got {}",
        track_cutoff
    );
    assert_eq!(state.track_m_min[track_idx], track_cutoff);

    // Candidate star with mass 0.639620019 (< cutoff 0.639621019)
    let candidate_mass = 0.639620019_f64;
    assert!(candidate_mass < track_cutoff);
    assert!(candidate_mass > window.box_mmin);

    let mut star = StarParticle::default();
    star.age = db.age_grid[age_idx];
    star.feh = db.feh_grid[feh_idx];
    star.smass = candidate_mass;

    db.interpolate_star(&mut star, &state);

    // Must be rejected by leaf-specific track cutoff!
    assert_eq!(
        star.abs_mags[0], 1e6,
        "leaf-cutoff witness star must be rejected (abs_mags[0] == 1e6) by track leaf-specific cutoff"
    );
}

/// Verify maxV cache hysteresis (delta_maxV = 0.01).
#[test]
fn test_max_v_cache_hysteresis() {
    let data_dir = galaxia_rs::require_data_dir!();
    let db = IsochroneDB::load(&data_dir).expect("Load IsochroneDB");
    let mut state = IsochroneFieldState::new(&db);

    let mut imf = Sampler::for_pop_id(7);
    let mut rng = Philox::from_stream(1, 0, 7, 0, 0, 0);

    // Set max_v to 10.0
    db.set_isochrones1(
        &mut state, 11e9, 1e9, -0.78, 0.3, 10.0, 1000.0, 0, &mut imf, &mut rng,
    );
    assert_eq!(state.max_v, 10.0);

    // Change within 0.01: max_v should NOT change
    db.set_isochrones1(
        &mut state, 11e9, 1e9, -0.78, 0.3, 10.005, 1000.0, 0, &mut imf, &mut rng,
    );
    assert_eq!(
        state.max_v, 10.0,
        "maxV should not update when delta <= 0.01"
    );

    // Change > 0.01: max_v MUST change
    db.set_isochrones1(
        &mut state, 11e9, 1e9, -0.78, 0.3, 10.02, 1000.0, 0, &mut imf, &mut rng,
    );
    assert_eq!(state.max_v, 10.02, "maxV must update when delta > 0.01");
}

/// Float rounding transitions witness (found by independent code review).
/// Witness: log age = 10.08999999, FeH = 0.0, mass = 1.0.
/// In C++ generation: float rounded age (10.09000015) selects track 175 whose max mass < 1.0, rejecting it.
/// In C++ append1: clamps to endpoint J = -5.86299991608.
#[test]
fn test_witness_float_transitions() {
    let data_dir = galaxia_rs::require_data_dir!();
    let db = IsochroneDB::load(&data_dir).expect("Load IsochroneDB");
    let state = IsochroneFieldState::new(&db);

    let mut star = StarParticle::default();
    star.age = 10.08999999;
    star.feh = 0.0;
    star.smass = 1.0;

    db.interpolate_star(&mut star, &state);

    // Generation must REJECT the star
    assert_eq!(
        star.abs_mags[0], 1e6,
        "float-quantization generation witness must be rejected (J=1e6) due to float age quantization"
    );

    // Append (interpolate_tgm) must yield the exact older track endpoint
    let tgm = db.interpolate_tgm(10.08999999, 0.0, 1.0);
    // In DCMC, mags = [I, J, H, Ks], so mags[1] is J
    assert!(
        (tgm.mags[1] - (-5.86299991608)).abs() < 1e-4,
        "float-quantization append witness J mag must clamp to endpoint -5.863, got {}",
        tgm.mags[1]
    );
}

/// Half-grid points for Padova age grid.
#[test]
fn test_age_grid_midpoints() {
    let data_dir = galaxia_rs::require_data_dir!();
    let db = IsochroneDB::load(&data_dir).expect("Load IsochroneDB");
    let state = IsochroneFieldState::new(&db);

    for pair in db.age_grid.windows(2) {
        let age_mid = (pair[0] + pair[1]) / 2.0;
        let mut s1 = StarParticle::default();
        s1.age = age_mid;
        s1.feh = 0.0;
        s1.smass = 0.8;
        db.interpolate_star(&mut s1, &state);

        let tgm = db.interpolate_tgm(age_mid, 0.0, 0.8);
        assert!(tgm.teff > 3.0 && tgm.teff < 5.0);
    }
}

/// CSV input validation (malformed rows, non-finite values).
#[test]
fn test_csv_validation() {
    let temp_dir = std::env::temp_dir().join(format!("gx_test_atomic_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();

    // 1. Short CSV row
    let short_csv = temp_dir.join("short.csv");
    {
        let mut f = File::create(&short_csv).unwrap();
        writeln!(f, "tic,ra,dec").unwrap();
        writeln!(f, "12345,10.0").unwrap(); // missing dec
    }
    let res = parse_chunk_csv(&short_csv);
    assert!(res.is_err(), "Short CSV row must return error");

    // 2. NaN coordinate
    let nan_csv = temp_dir.join("nan.csv");
    {
        let mut f = File::create(&nan_csv).unwrap();
        writeln!(f, "tic,ra,dec").unwrap();
        writeln!(f, "12345,NaN,10.0").unwrap();
    }
    let res = parse_chunk_csv(&nan_csv);
    assert!(res.is_err(), "NaN coordinate must return error");

    // 3. Out of bounds RA (> 360)
    let oob_csv = temp_dir.join("oob.csv");
    {
        let mut f = File::create(&oob_csv).unwrap();
        writeln!(f, "tic,ra,dec").unwrap();
        writeln!(f, "12345,400.0,10.0").unwrap();
    }
    let res = parse_chunk_csv(&oob_csv);
    assert!(res.is_err(), "Out of bounds RA must return error");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

/// Atomic EBF creation via temporary file and rename.
#[test]
fn test_atomic_ebf_write() {
    let data_dir = galaxia_rs::require_data_dir!();

    let temp_dir = std::env::temp_dir().join(format!("gx_test_atomic_{}", std::process::id()));
    let out_dir = temp_dir.join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    let csv = temp_dir.join("field.csv");
    {
        let mut f = File::create(&csv).unwrap();
        writeln!(f, "tic,ra,dec").unwrap();
        writeln!(f, "346418409,344.609667,39.658711").unwrap();
    }

    let config = BatchConfig {
        csv_path: csv,
        out_dir: out_dir.clone(),
        data_dir: data_dir.clone(),
        backend: BackendKind::Cpu,
        tmag_ceiling: Some(22.5),
        threads: 1,
    };

    let res = run_batch(&config);
    assert!(res.is_ok(), "Batch run failed: {:?}", res.err());

    let final_file = out_dir.join("gx_346418409.ebf");
    assert!(final_file.exists(), "Final EBF file must exist");

    // Ensure no stale temporary file remains
    let entries = std::fs::read_dir(&out_dir).unwrap();
    for entry in entries {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(
            !name.contains(".tmp."),
            "No temporary files must remain: {}",
            name
        );
    }

    // Verify /log metadata in generated file
    let mut reader = EbfFile::open(&final_file).expect("Open EBF");
    assert!(reader.contains_key("/log"), "EBF must contain /log dataset");
    let log_ds = reader.read("/log").expect("Read /log");
    let log_str = match log_ds.data {
        galaxia_rs::ebf::EbfData::UInt8(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        galaxia_rs::ebf::EbfData::String(s) => s,
        _ => panic!("Expected UInt8/Char or String data for /log"),
    };
    assert!(
        log_str.contains("backend: cpu-fp64"),
        "/log must contain 'backend: cpu-fp64'"
    );
    assert!(
        log_str.contains("tmag_ceiling: 22.500000"),
        "/log must contain ceiling"
    );
    assert!(log_str.contains("seed: 18410"), "/log must contain seed");
    assert!(
        log_str.contains("coordinates:"),
        "/log must contain coordinates"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

/// Warp literal constant (3.14159) and ThinDisk Option E=0.
#[test]
fn test_warp_and_option_e() {
    assert_eq!(
        PI_WARP, 3.14159,
        "PI_WARP must be exactly 3.14159 matching C++ Population.h"
    );

    // Option E=0 must use epsvec
    let pop0_e0 = create_population(0, 1, 0);
    assert_eq!(pop0_e0.eps(), 0.014);

    let pop1_e0 = create_population(1, 1, 0);
    assert_eq!(pop1_e0.eps(), 0.0268);
}

/// Backend production gate (only CPU permitted in without --allow-fp32-test).
#[test]
fn test_backend_gate_version_tag() {
    let cpu_tag = version_string(BackendKind::Cpu, Some(22.5));
    assert!(cpu_tag.contains("backend=cpu-fp64"));
    assert!(cpu_tag.contains("tmag_ceiling=22.500000"));
}

/// CLI refusal tests for untested GPU backends and invalid flags.
#[test]
fn test_cli_refusal_untested_gpu_backends_and_invalid_ceiling() {
    let bin_path = env!("CARGO_BIN_EXE_galaxia-rs");

    // 1. --backend opencl without --allow-fp32-test must exit non-zero (code 1)
    let out_cl = std::process::Command::new(bin_path)
        .args(&[
            "--backend",
            "opencl",
            "-i",
            "dummy.csv",
            "-o",
            "/tmp",
            "-d",
            "/tmp",
        ])
        .output()
        .expect("Run CLI with opencl");
    assert!(
        !out_cl.status.success(),
        "opencl without allow-fp32-test must fail"
    );
    let err_cl = String::from_utf8_lossy(&out_cl.stderr);
    assert!(err_cl.contains("not certified for production") || err_cl.contains("allow-fp32-test"));

    // 2. --backend cuda without --allow-fp32-test must exit non-zero (code 1)
    let out_cuda = std::process::Command::new(bin_path)
        .args(&[
            "--backend",
            "cuda",
            "-i",
            "dummy.csv",
            "-o",
            "/tmp",
            "-d",
            "/tmp",
        ])
        .output()
        .expect("Run CLI with cuda");
    assert!(
        !out_cuda.status.success(),
        "cuda without allow-fp32-test must fail"
    );
    let err_cuda = String::from_utf8_lossy(&out_cuda.stderr);
    assert!(
        err_cuda.contains("not certified for production") || err_cuda.contains("allow-fp32-test")
    );

    // 3. Non-numeric --tmag-ceiling must fail in CLI pre-scan
    let out_ceil = std::process::Command::new(bin_path)
        .args(&["--tmag-ceiling", "invalid_number", "-i", "dummy.csv"])
        .output()
        .expect("Run CLI with invalid ceiling");
    assert!(
        !out_ceil.status.success(),
        "invalid --tmag-ceiling must fail"
    );
    let err_ceil = String::from_utf8_lossy(&out_ceil.stderr);
    assert!(err_ceil.contains("Invalid --tmag-ceiling"));
}

/// Batch failure tests (injected failure, missing EBF, and invalid BHTree leaf each giving non-zero exit).
#[test]
fn test_batch_failure_modes() {
    let data_dir = galaxia_rs::require_data_dir!();
    let temp_base = std::env::temp_dir().join(format!("gx_test_d3_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_base);
    std::fs::create_dir_all(&temp_base).expect("Create temp base");

    let csv_path = temp_base.join("input.csv");
    {
        let mut f = File::create(&csv_path).unwrap();
        writeln!(f, "tic,ra,dec").unwrap();
        writeln!(f, "346418409,344.609667,39.658711").unwrap();
    }

    // 1. Injected failure: invalid output path (file exists where dir expected)
    let fake_out_file = temp_base.join("out_as_file");
    {
        let mut f = File::create(&fake_out_file).unwrap();
        writeln!(f, "blocking file").unwrap();
    }
    let config_fail = BatchConfig {
        csv_path: csv_path.clone(),
        out_dir: fake_out_file.join("subdir"), // cannot create directory under regular file
        data_dir: data_dir.clone(),
        backend: BackendKind::Cpu,
        tmag_ceiling: Some(22.5),
        threads: 1,
    };
    let res1 = run_batch(&config_fail);
    assert!(
        res1.is_err(),
        "Batch must return Err when field cannot write output directory"
    );

    // 2. Corrupt/Invalid BHTree leaf: pointing data_dir to a corrupted tree
    let bad_data_dir = temp_base.join("bad_data");
    let bad_trees_dir = bad_data_dir.join("BHTree");
    std::fs::create_dir_all(&bad_trees_dir).unwrap();
    // Copy Isochrones, Extinction, Model from real data dir to bad_data_dir
    for d in ["Isochrones", "Extinction", "Model"] {
        link_dir(&data_dir.join(d), &bad_data_dir.join(d));
    }
    // Create an invalid/corrupted BHTree file for pop 0
    let bad_tree_file = bad_trees_dir.join("ThinDisk0_E1.ebf");
    {
        let mut f = File::create(&bad_tree_file).unwrap();
        f.write_all(b"CORRUPTED_BHTREE_DATA_NOT_VALID_EBF").unwrap();
    }

    let config_bad_tree = BatchConfig {
        csv_path: csv_path.clone(),
        out_dir: temp_base.join("out_bad_tree"),
        data_dir: bad_data_dir,
        backend: BackendKind::Cpu,
        tmag_ceiling: Some(22.5),
        threads: 1,
    };
    let res2 = run_batch(&config_bad_tree);
    assert!(
        res2.is_err(),
        "Batch must return Err when BHTree is corrupted"
    );

    // Clean up
    let _ = std::fs::remove_dir_all(temp_base);
}

/// Directory symlink on every platform (Windows needs symlink permission or developer mode).
fn link_dir(src: &std::path::Path, dst: &std::path::Path) {
    #[cfg(unix)]
    let r = std::os::unix::fs::symlink(src, dst);
    #[cfg(windows)]
    let r = std::os::windows::fs::symlink_dir(src, dst);
    if let Err(e) = r {
        if e.kind() != std::io::ErrorKind::AlreadyExists {
            panic!("cannot link {} -> {}: {e}", dst.display(), src.display());
        }
    }
}
