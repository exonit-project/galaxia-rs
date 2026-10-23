// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::ebf::EbfFile;
use std::path::Path;
use std::process::Command;

#[test]
fn test_read_gx_test_ebf_and_compare_with_python() {
    let _data_dir = galaxia_rs::require_data_dir!();
    let ref_dir = galaxia_rs::testutil::cpp_ref_dir();
    let gx_file_cpp = ref_dir.join("gx_test_cpp.ebf");
    let gx_file_cpp = gx_file_cpp.as_path();
    let gx_file = if gx_file_cpp.exists() {
        gx_file_cpp
    } else {
        Path::new("gx_test.ebf")
    };
    assert!(
        gx_file.exists(),
        "C++ reference file gx_test_cpp.ebf not found in {}",
        ref_dir.display()
    );

    let mut reader = EbfFile::open(gx_file).expect("Failed to open C++ reference EBF");
    let keys = reader.keys();
    println!("Found {} keys in gx_test.ebf", keys.len());
    assert!(keys.contains(&"/px".to_string()));
    assert!(keys.contains(&"/rad".to_string()));
    assert!(keys.contains(&"/teff".to_string()));
    assert!(keys.contains(&"/grav".to_string()));
    assert!(keys.contains(&"/feh".to_string()));
    assert!(keys.contains(&"/popid".to_string()));

    let px_rust = reader.read_f32("/px").expect("Failed to read /px in Rust");
    let rad_rust = reader
        .read_f32("/rad")
        .expect("Failed to read /rad in Rust");
    let teff_rust = reader
        .read_f32("/teff")
        .expect("Failed to read /teff in Rust");
    let popid_rust = reader
        .read_i32("/popid")
        .expect("Failed to read /popid in Rust");

    if gx_file == gx_file_cpp {
        assert_eq!(px_rust.len(), 32911);
        assert_eq!(rad_rust.len(), 32911);
        assert_eq!(teff_rust.len(), 32911);
        assert_eq!(popid_rust.len(), 32911);
    } else {
        assert!(px_rust.len() > 30000 && px_rust.len() < 35000);
        assert_eq!(rad_rust.len(), px_rust.len());
        assert_eq!(teff_rust.len(), px_rust.len());
        assert_eq!(popid_rust.len(), px_rust.len());
    }

    // If GALAXIA_PYTHON (python with the `ebf` module) is set, compare first 100 values bit-for-bit
    if let Some(python_bin) = galaxia_rs::testutil::python_bin() {
        let py_cmd = format!(
            r#"
import ebf, json
px = ebf.read("{path}", "/px")[:100].tolist()
rad = ebf.read("{path}", "/rad")[:100].tolist()
teff = ebf.read("{path}", "/teff")[:100].tolist()
popid = ebf.read("{path}", "/popid")[:100].tolist()
print(json.dumps({{"px": px, "rad": rad, "teff": teff, "popid": popid}}))
"#,
            path = gx_file.display()
        );
        let output = Command::new(&python_bin)
            .arg("-c")
            .arg(py_cmd)
            .output()
            .expect("Failed to run python check");

        if output.status.success() {
            let json_str = String::from_utf8_lossy(&output.stdout);
            let parsed: serde_json::Value =
                serde_json::from_str(&json_str).expect("Failed to parse json");

            let py_px: Vec<f32> = parsed["px"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let py_rad: Vec<f32> = parsed["rad"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let py_teff: Vec<f32> = parsed["teff"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let py_popid: Vec<i32> = parsed["popid"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();

            assert_eq!(&px_rust[..100], &py_px[..]);
            assert_eq!(&rad_rust[..100], &py_rad[..]);
            assert_eq!(&teff_rust[..100], &py_teff[..]);
            assert_eq!(&popid_rust[..100], &py_popid[..]);
            println!("gx_test.ebf verified against python ebf!");
        }
    }
}

#[test]
fn test_read_exmap3d_ebf() {
    let data_dir = galaxia_rs::require_data_dir!();
    let exmap_file = data_dir.join("Extinction/ExMap3d_1024.ebf");
    assert!(
        exmap_file.exists(),
        "Extinction map file {} not found",
        exmap_file.display()
    );

    let reader = EbfFile::open(&exmap_file).expect("Failed to open Extinction/ExMap3d_1024.ebf");
    let keys = reader.keys();
    println!("Found {} keys in ExMap3d_1024.ebf: {:?}", keys.len(), keys);

    assert!(
        reader.contains_key("/exmap3d.data") || reader.contains_key("/ExMap3d"),
        "Missing exmap3d key"
    );
    assert!(
        reader.contains_key("/exmap2d.data") || reader.contains_key("/ExMap2d"),
        "Missing exmap2d key"
    );

    let h3d = reader
        .header("/exmap3d.data")
        .expect("Missing header for /exmap3d.data");
    assert_eq!(h3d.dims, vec![1024, 1024, 100]);
    assert_eq!(h3d.elements(), 1024 * 1024 * 100);

    let h2d = reader
        .header("/exmap2d.data")
        .expect("Missing header for /exmap2d.data");
    assert_eq!(h2d.dims, vec![1024, 1024]);
    assert_eq!(h2d.elements(), 1024 * 1024);

    let h3d_xmms = reader
        .header("/exmap3d.xmms")
        .expect("Missing header for /exmap3d.xmms");
    assert_eq!(h3d_xmms.dims, vec![3, 3]);

    println!("ExMap3d_1024.ebf (v1.0) verified successfully!");
}

#[test]
fn test_read_all_bhtree_ebf_files() {
    let data_dir = galaxia_rs::require_data_dir!();
    let bhtree_dir = data_dir.join("BHTree-2.2/bhtree_with_wf");
    assert!(
        bhtree_dir.exists(),
        "BHTree directory {} not found",
        bhtree_dir.display()
    );

    let entries = std::fs::read_dir(&bhtree_dir).expect("Failed to read bhtree dir");
    let mut count = 0;
    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("ebf") {
            let reader = EbfFile::open(&path)
                .unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e));

            assert!(
                reader.contains_key("/nodes"),
                "File {} missing /nodes",
                path.display()
            );
            assert!(
                reader.contains_key("/leafs"),
                "File {} missing /leafs",
                path.display()
            );

            let nodes_header = reader.header("/nodes").unwrap();
            let leafs_header = reader.header("/leafs").unwrap();

            assert_eq!(nodes_header.dims.len(), 2);
            assert_eq!(nodes_header.dims[1], 12);
            assert_eq!(leafs_header.dims.len(), 1);

            count += 1;
            println!(
                "Verified {}: {} nodes, {} leafs",
                path.file_name().unwrap().to_string_lossy(),
                nodes_header.dims[0],
                leafs_header.dims[0]
            );
        }
    }

    assert!(
        count >= 10,
        "Expected at least 10 bhtree files, found {}",
        count
    );
    println!("All {} BHTree EBF files verified successfully!", count);
}
