// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::ebf::EbfFile;
use galaxia_rs::extinction::ExtinctionMap;

#[test]
fn test_extinction_parity_with_cpp_gx_test() {
    let data_dir = galaxia_rs::require_data_dir!();
    let ref_dir = galaxia_rs::testutil::cpp_ref_dir();
    let gx_file = ref_dir.join("gx_test_cpp.ebf");

    assert!(gx_file.exists(), "Test file {:?} not found", gx_file);

    let mut reader = EbfFile::open(&gx_file).expect("Failed to open C++ reference EBF");

    let px = reader.read_f32("/px").expect("Read /px");
    let py = reader.read_f32("/py").expect("Read /py");
    let pz = reader.read_f32("/pz").expect("Read /pz");

    let cpp_ex_sch = reader
        .read_f32("/exbv_schlegel")
        .expect("Read /exbv_schlegel");
    let cpp_ex_sol = reader.read_f32("/exbv_solar").expect("Read /exbv_solar");
    let cpp_ex_inf = reader
        .read_f32("/exbv_schlegel_inf")
        .expect("Read /exbv_schlegel_inf");
    let cpp_glon = reader.read_f32("/glon").expect("Read /glon");
    let cpp_glat = reader.read_f32("/glat").expect("Read /glat");

    let n = px.len();
    assert_eq!(py.len(), n);
    assert_eq!(pz.len(), n);
    assert_eq!(cpp_ex_sch.len(), n);
    assert_eq!(cpp_ex_sol.len(), n);
    assert_eq!(cpp_ex_inf.len(), n);

    let ext = ExtinctionMap::load_from_dir(&data_dir).expect("Load extinction maps");

    // Check first 1000 stars
    let check_n = n.min(1000);
    let mut max_diff_glon = 0.0f32;
    let mut max_diff_glat = 0.0f32;
    let mut max_diff_sch = 0.0f32;
    let mut max_diff_sol = 0.0f32;
    let mut max_diff_inf = 0.0f32;

    for i in 0..check_n {
        let pos = [px[i] as f64, py[i] as f64, pz[i] as f64];
        let values = ext.compute(&pos);

        let d_glon = (values.glon - cpp_glon[i]).abs();
        let d_glat = (values.glat - cpp_glat[i]).abs();
        let d_sch = (values.exbv_schlegel - cpp_ex_sch[i]).abs();
        let d_sol = (values.exbv_solar - cpp_ex_sol[i]).abs();
        let d_inf = (values.exbv_schlegel_inf - cpp_ex_inf[i]).abs();

        if d_glon > max_diff_glon {
            max_diff_glon = d_glon;
        }
        if d_glat > max_diff_glat {
            max_diff_glat = d_glat;
        }
        if d_sch > max_diff_sch {
            max_diff_sch = d_sch;
        }
        if d_sol > max_diff_sol {
            max_diff_sol = d_sol;
        }
        if d_inf > max_diff_inf {
            max_diff_inf = d_inf;
        }

        assert!(
            d_glon < 1e-4,
            "Star {}: glon mismatch computed={} cpp={}",
            i,
            values.glon,
            cpp_glon[i]
        );
        assert!(
            d_glat < 1e-4,
            "Star {}: glat mismatch computed={} cpp={}",
            i,
            values.glat,
            cpp_glat[i]
        );
        assert!(
            d_sch < 1e-4,
            "Star {}: E(B-V)_schlegel mismatch computed={} cpp={}",
            i,
            values.exbv_schlegel,
            cpp_ex_sch[i]
        );
        assert!(
            d_sol < 1e-4,
            "Star {}: E(B-V)_solar mismatch computed={} cpp={}",
            i,
            values.exbv_solar,
            cpp_ex_sol[i]
        );
        assert!(
            d_inf < 1e-4,
            "Star {}: E(B-V)_schlegel_inf mismatch computed={} cpp={}",
            i,
            values.exbv_schlegel_inf,
            cpp_ex_inf[i]
        );
    }

    println!(
        "Verified parity on {} stars against C++ gx_test.ebf:",
        check_n
    );
    println!("  max diff glon:               {:.2e} deg", max_diff_glon);
    println!("  max diff glat:               {:.2e} deg", max_diff_glat);
    println!("  max diff E(B-V)_schlegel:    {:.2e}", max_diff_sch);
    println!("  max diff E(B-V)_solar:       {:.2e}", max_diff_sol);
    println!("  max diff E(B-V)_schlegel_inf:{:.2e}", max_diff_inf);
}
