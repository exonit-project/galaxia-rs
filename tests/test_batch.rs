// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::backend::{create_backend, version_string, BackendKind};
use galaxia_rs::batch::{parse_chunk_csv, run_batch, BatchConfig};
use galaxia_rs::coords::{icrs_to_galactic, seed_for_tic};
use galaxia_rs::ebf::EbfFile;
use galaxia_rs::generator::{extincted_tmag, stassun_tmag};
use std::fs::File;
use std::io::Write;

#[test]
fn test_seed_for_tic_deterministic() {
    assert_eq!(seed_for_tic(0), 1);
    assert_eq!(seed_for_tic(99999), 100000);
    assert_eq!(seed_for_tic(100000), 1);
    assert_eq!(seed_for_tic(346418409), 18410);
    assert_eq!(seed_for_tic(83653976), 53977);
}

#[test]
fn test_icrs_to_galactic_hipparcos() {
    // Known test point (180.0, 30.0)
    let (l, b) = icrs_to_galactic(180.0, 30.0);
    assert!(
        (l - 195.639488).abs() < 1e-4,
        "Expected l ~ 195.6395, got {}",
        l
    );
    assert!(
        (b - 78.353806).abs() < 1e-4,
        "Expected b ~ 78.3538, got {}",
        b
    );

    // Galactic center region: RA ~ 266.4 deg, Dec ~ -29.0 deg -> l ~ 0 deg, b ~ 0 deg
    let (gc_l, gc_b) = icrs_to_galactic(266.405, -28.936);
    assert!((gc_l < 1.0 || gc_l > 359.0), "Expected l ~ 0, got {}", gc_l);
    assert!(gc_b.abs() < 1.0, "Expected b ~ 0, got {}", gc_b);
}

#[test]
fn test_stassun_tmag_branches() {
    // Branch 1: c < -0.1
    let t1 = stassun_tmag(12.0, 12.2); // c = -0.2
    assert!((t1 - 12.5).abs() < 1e-6);

    // Branch 2: -0.1 <= c <= 0.70
    let t2 = stassun_tmag(15.0, 14.6); // c = 0.4
    let expected_t2 =
        15.0 + 1.22163 * 0.4_f64.powi(3) - 1.74299 * 0.4_f64.powi(2) + 1.89115 * 0.4 + 0.0563;
    assert!((t2 - expected_t2).abs() < 1e-6);

    // Branch 3: 0.70 < c <= 1.0
    let t3 = stassun_tmag(14.0, 13.15); // c = 0.85
    let expected_t3 =
        14.0 - 269.372 * 0.85_f64.powi(3) + 668.453 * 0.85_f64.powi(2) - 545.64 * 0.85 + 147.811;
    assert!((t3 - expected_t3).abs() < 1e-6);

    // Branch 4: c > 1.0
    let t4 = stassun_tmag(16.0, 14.5); // c = 1.5
    assert!((t4 - 17.75).abs() < 1e-6);
}

#[test]
fn test_extincted_tmag() {
    // Distance 1 kpc (dm = 10.0), J_abs = 2.0, Ks_abs = 1.5, ebv = 0.1
    // j_ext = 2.0 + 10.0 + 0.282 * 3.1 * 0.1 = 12.08742
    // ks_ext = 1.5 + 10.0 + 0.112 * 3.1 * 0.1 = 11.53472
    let tmag = extincted_tmag(1.0, 2.0, 1.5, 0.1);
    assert!(tmag > 12.0 && tmag < 14.0);
}

#[test]
fn test_parse_chunk_csv() {
    let temp_dir = std::env::temp_dir();
    let csv_file = temp_dir.join("test_chunk_parser.csv");
    {
        let mut f = File::create(&csv_file).expect("Create test CSV");
        writeln!(f, "tic,ra,dec,extra_col").unwrap();
        writeln!(f, "12345,100.5,-20.25,some_value").unwrap();
        writeln!(f, "67890,200.123,45.678,another_value").unwrap();
    }

    let records = parse_chunk_csv(&csv_file).expect("Parse CSV");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].tic, 12345);
    assert!((records[0].ra - 100.5).abs() < 1e-6);
    assert!((records[0].dec - -20.25).abs() < 1e-6);
    assert_eq!(records[1].tic, 67890);

    let _ = std::fs::remove_file(csv_file);
}

#[test]
fn test_version_strings() {
    let v_cuda = version_string(BackendKind::Cuda, None);
    let v_cpu = version_string(BackendKind::Cpu, Some(22.5));
    assert!(v_cuda.contains("backend=cuda") && v_cuda.contains("tmag_ceiling=none"));
    assert!(v_cpu.contains("backend=cpu-fp64") && v_cpu.contains("tmag_ceiling=22.500000"));
}

#[test]
fn test_no_silent_fallback_on_unavailable_cuda() {
    // If CUDA is unavailable on this host, create_backend(BackendKind::Cuda) MUST fail.
    // Silent fallback to CPU is strictly forbidden.
    match create_backend(BackendKind::Cuda) {
        Ok(backend) => {
            // If running on a machine with a working NVIDIA GPU
            assert!(backend.name().to_lowercase().contains("nvidia") || !backend.name().is_empty());
        }
        Err(e) => {
            // On non-GPU host, it must return an error mentioning policy/forbidden fallback
            let err_str = e.to_string();
            assert!(
                err_str.contains("No silent fallback") || err_str.contains("failed"),
                "Error message should clearly state policy: {}",
                err_str
            );
        }
    }
}

#[test]
fn test_batch_mode_small_3_field_csv() {
    let data_dir = galaxia_rs::require_data_dir!();

    let temp_base = std::env::temp_dir().join("gx_batch_test");
    let out_dir = temp_base.join("output");
    let _ = std::fs::remove_dir_all(&temp_base);
    std::fs::create_dir_all(&temp_base).expect("Create temp base dir");

    let csv_path = temp_base.join("chunk_test.csv");
    {
        let mut f = File::create(&csv_path).expect("Create test chunk CSV");
        writeln!(f, "tic,ra,dec").unwrap();
        // 3 high-latitude test fields (fast spawning, low density)
        writeln!(f, "346418409,344.609667,39.658711").unwrap();
        writeln!(f, "160074939,346.319167,-36.945639").unwrap();
        writeln!(f, "316937670,356.492458,-56.344047").unwrap();
    }

    let config = BatchConfig {
        csv_path: csv_path.clone(),
        out_dir: out_dir.clone(),
        data_dir: data_dir.to_path_buf(),
        backend: BackendKind::Cpu,
        tmag_ceiling: Some(22.5),
        threads: 1,
    };

    let batch_res = run_batch(&config);
    assert!(batch_res.is_ok(), "run_batch failed: {:?}", batch_res.err());

    // Verify all 3 output EBF files were written
    let expected_files = [
        out_dir.join("gx_346418409.ebf"),
        out_dir.join("gx_160074939.ebf"),
        out_dir.join("gx_316937670.ebf"),
    ];

    for file_path in &expected_files {
        assert!(
            file_path.exists(),
            "Expected output file '{}' was not created",
            file_path.display()
        );

        let reader = EbfFile::open(file_path).expect("Failed to read output EBF file");
        assert!(reader.contains_key("/rad"), "EBF missing /rad");
        assert!(reader.contains_key("/dcmc_j"), "EBF missing /dcmc_j");
        assert!(
            reader.contains_key("/exbv_solar"),
            "EBF missing /exbv_solar"
        );
    }

    // Clean up
    let _ = std::fs::remove_dir_all(temp_base);
}

#[test]
fn test_batch_multithreaded_determinism() {
    let data_dir = galaxia_rs::require_data_dir!();

    let temp_base = std::env::temp_dir().join(format!("gx_test_batch_mt_{}", std::process::id()));
    let out_dir_t1 = temp_base.join("out_t1");
    let out_dir_t2 = temp_base.join("out_t2");
    let _ = std::fs::create_dir_all(&out_dir_t1);
    let _ = std::fs::create_dir_all(&out_dir_t2);

    let csv_path = temp_base.join("chunk_test.csv");
    {
        let mut f = File::create(&csv_path).expect("Create test chunk CSV");
        writeln!(f, "tic,ra,dec").unwrap();
        // 2 high-latitude test fields
        writeln!(f, "346418409,344.609667,39.658711").unwrap();
        writeln!(f, "160074939,346.319167,-36.945639").unwrap();
    }

    let config_t1 = BatchConfig {
        csv_path: csv_path.clone(),
        out_dir: out_dir_t1.clone(),
        data_dir: data_dir.to_path_buf(),
        backend: BackendKind::Cpu,
        tmag_ceiling: Some(22.5),
        threads: 1,
    };
    let res_t1 = run_batch(&config_t1);
    assert!(
        res_t1.is_ok(),
        "Single-threaded batch failed: {:?}",
        res_t1.err()
    );

    let config_t2 = BatchConfig {
        csv_path: csv_path.clone(),
        out_dir: out_dir_t2.clone(),
        data_dir: data_dir.to_path_buf(),
        backend: BackendKind::Cpu,
        tmag_ceiling: Some(22.5),
        threads: 2,
    };
    let res_t2 = run_batch(&config_t2);
    assert!(
        res_t2.is_ok(),
        "Multi-threaded batch failed: {:?}",
        res_t2.err()
    );

    for tic in &[346418409u64, 160074939u64] {
        let f1 = out_dir_t1.join(format!("gx_{}.ebf", tic));
        let f2 = out_dir_t2.join(format!("gx_{}.ebf", tic));
        assert!(f1.exists(), "f1 does not exist: {}", f1.display());
        assert!(f2.exists(), "f2 does not exist: {}", f2.display());

        let bytes1 = std::fs::read(&f1).expect("read f1");
        let bytes2 = std::fs::read(&f2).expect("read f2");
        assert_eq!(
            bytes1, bytes2,
            "EBF outputs differ between threads=1 and threads=2 for field {}",
            tic
        );
    }

    let _ = std::fs::remove_dir_all(temp_base);
}
