// SPDX-License-Identifier: AGPL-3.0-or-later
//! Test-only helpers for locating external test inputs (not part of the public API).
//!
//! Data-dependent tests need the original `GalaxiaData` directory, which is not
//! redistributed with this crate. Selection is explicit:
//!
//! * `GALAXIA_DATA_DIR` unset: data-dependent tests are *not selected*; each prints a
//!   notice (visible with `-- --nocapture`) and returns. This is an explicit opt-out.
//! * `GALAXIA_DATA_DIR` set: data-dependent tests are selected and FAIL if the
//!   directory (or any other required input) is missing.
//!
//! Further selectors:
//! * `GALAXIA_CPP_REF_DIR` - directory with `gx_test_cpp.ebf`, the output of the original
//!   C++ `galaxia -r tests/fixtures/t.par` (default: `$TMPDIR/gxt`).
//! * `GALAXIA_PYTHON` - Python interpreter with the `ebf` module (EBF interop tests).

use std::path::PathBuf;

/// `GALAXIA_DATA_DIR` if set (panics if it is not a directory), else `None` (tests not selected).
pub fn data_dir() -> Option<PathBuf> {
    match std::env::var_os("GALAXIA_DATA_DIR") {
        Some(v) if !v.is_empty() => {
            let p = PathBuf::from(v);
            assert!(
                p.is_dir(),
                "GALAXIA_DATA_DIR is set but {:?} is not a directory",
                p
            );
            Some(p)
        }
        _ => {
            eprintln!(
                "NOT SELECTED: data-dependent test skipped (set GALAXIA_DATA_DIR to the GalaxiaData directory to run it)"
            );
            None
        }
    }
}

/// Directory with C++ reference output (`gx_test_cpp.ebf`). Only call after `data_dir()` returned `Some`.
pub fn cpp_ref_dir() -> PathBuf {
    let p = match std::env::var_os("GALAXIA_CPP_REF_DIR") {
        Some(v) => PathBuf::from(v),
        None => std::env::temp_dir().join("gxt"),
    };
    assert!(
        p.is_dir(),
        "C++ reference directory {:?} not found (set GALAXIA_CPP_REF_DIR; see README)",
        p
    );
    p
}

/// `GALAXIA_PYTHON` if set (panics if it does not exist), else `None` (tests not selected).
pub fn python_bin() -> Option<String> {
    match std::env::var("GALAXIA_PYTHON") {
        Ok(v) if !v.is_empty() => {
            assert!(
                std::path::Path::new(&v).exists(),
                "GALAXIA_PYTHON is set but {} does not exist",
                v
            );
            Some(v)
        }
        _ => {
            eprintln!(
                "NOT SELECTED: python interop test skipped (set GALAXIA_PYTHON to a python with the `ebf` module)"
            );
            None
        }
    }
}

/// Early-return from a test when `GALAXIA_DATA_DIR` is not set; otherwise yields the path.
#[macro_export]
macro_rules! require_data_dir {
    () => {
        match $crate::testutil::data_dir() {
            Some(d) => d,
            None => return,
        }
    };
}
