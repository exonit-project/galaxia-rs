// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::ebf::{self, EbfWriter};
use std::process::Command;

#[test]
fn test_rust_ebf_writer_python_reader_interop() {
    let python_bin = match galaxia_rs::testutil::python_bin() {
        Some(p) => p,
        None => return,
    };

    let temp_dir = std::env::temp_dir();
    let test_file = temp_dir.join("galaxia_rs_interop_test.ebf");

    let f32_vals: Vec<f32> = vec![1.25, 2.5, 3.75, 5.0];
    let f64_vals: Vec<f64> = vec![100.1, 200.2, 300.3];
    let i32_vals: Vec<i32> = vec![10, 20, 30, 40, 50];
    let i64_vals: Vec<i64> = vec![1000, 2000, 3000];

    {
        let mut writer = EbfWriter::create(&test_file).expect("Failed to create EBF writer");
        writer
            .write_f32("/data/f32_arr", "km/s", &[4], &f32_vals)
            .expect("Failed to write f32");
        writer
            .write_f64("/data/f64_arr", "Msun", &[3], &f64_vals)
            .expect("Failed to write f64");
        writer
            .write_i32("/data/i32_arr", "count", &[5], &i32_vals)
            .expect("Failed to write i32");
        writer
            .write_i64("/data/i64_arr", "id", &[3], &i64_vals)
            .expect("Failed to write i64");
    }

    // Verify python ebf.read can read all datasets and check values
    let py_script = format!(
        r#"
import ebf, numpy as np, sys

fname = r"{path}"
f32_data = ebf.read(fname, "/data/f32_arr")
assert f32_data.dtype == np.float32, f"f32 dtype mismatch: {{f32_data.dtype}}"
assert np.allclose(f32_data, [1.25, 2.5, 3.75, 5.0]), f"f32 values mismatch: {{f32_data}}"

f64_data = ebf.read(fname, "/data/f64_arr")
assert f64_data.dtype == np.float64, f"f64 dtype mismatch: {{f64_data.dtype}}"
assert np.allclose(f64_data, [100.1, 200.2, 300.3]), f"f64 values mismatch: {{f64_data}}"

i32_data = ebf.read(fname, "/data/i32_arr")
assert i32_data.dtype == np.int32, f"i32 dtype mismatch: {{i32_data.dtype}}"
assert np.array_equal(i32_data, [10, 20, 30, 40, 50]), f"i32 values mismatch: {{i32_data}}"

i64_data = ebf.read(fname, "/data/i64_arr")
assert i64_data.dtype == np.int64, f"i64 dtype mismatch: {{i64_data.dtype}}"
assert np.array_equal(i64_data, [1000, 2000, 3000]), f"i64 values mismatch: {{i64_data}}"

print("PYTHON_INTEROP_OK")
"#,
        path = test_file.display()
    );

    let output = Command::new(&python_bin)
        .arg("-c")
        .arg(&py_script)
        .output()
        .expect("Failed to run python");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Python interop failed!\nStdout: {}\nStderr: {}",
        stdout,
        stderr
    );
    assert!(stdout.contains("PYTHON_INTEROP_OK"));

    // Also verify reading back in Rust
    let mut reader = ebf::EbfFile::open(&test_file).expect("Failed to read back in Rust");
    assert_eq!(reader.read_f32("/data/f32_arr").unwrap(), f32_vals);
    assert_eq!(reader.read_f64("/data/f64_arr").unwrap(), f64_vals);
    assert_eq!(reader.read_i32("/data/i32_arr").unwrap(), i32_vals);
    assert_eq!(reader.read_i64("/data/i64_arr").unwrap(), i64_vals);

    let _ = std::fs::remove_file(test_file);
}
