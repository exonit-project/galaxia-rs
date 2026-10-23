// SPDX-License-Identifier: AGPL-3.0-or-later
// C++ parity test harness linked against original Galaxia 0.7.2 objects.
// Generates test fixtures covering leaf-cutoff and float-quantization witnesses, age/FeH grids,
// Vmon cache hysteresis, multi-leaf order and reset, and ThinDisk0 density (E=0).

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>

#include "IsochroneDB.h"
#include "Population.h"
#include "Sampler.h"
#include "Interp.h"

// Standard Chabrier IMF function used across Galaxia
static double chabrier_imf(double x) {
    return 22.8978 * exp(-pow(716.4 / x, 0.25)) * pow(x, -3.3);
}

// Subclass to access protected members of IsochroneBase and IsochroneDB
class ParityIsochroneDB : public IsochroneDB {
public:
    ParityIsochroneDB(const std::string& inputDir, const std::string& dirname,
                      const std::string& photoSys, const std::string& magcolorNames,
                      int extraFieldsOn)
        : IsochroneDB(inputDir, dirname, photoSys, magcolorNames, extraFieldsOn) {}

    const std::vector<double>& getAgeGrid() const { return Age; }
    const std::vector<double>& getFeHGrid() const { return FeH; }
    int getTrackIndex(int i, int j) { return index(i, j, 0); }
    double getTrackMMin(int track_idx) const { return icv[track_idx].m_min; }
    double getTrackMMax(int track_idx) const { return icv[track_idx].m_max; }
    const std::vector<double>& getTrackVmon(int track_idx) const { return icv[track_idx].Vmon; }
    const std::vector<double>& getTrackM(int track_idx) const { return icv[track_idx].m; }
    double getCurrentMaxV() const {
        // Return maxV from first track that was updated or 30.0
        return icv[0].maxV;
    }
};

int main(int argc, char** argv) {
    std::string data_dir;
    if (const char* env_p = std::getenv("GALAXIA_DATA_DIR")) {
        data_dir = env_p;
        if (!data_dir.empty() && data_dir.back() != '/') {
            data_dir += '/';
        }
    }
    if (data_dir.empty()) {
        std::cerr << "GALAXIA_DATA_DIR must point to the original GalaxiaData directory" << std::endl;
        return 2;
    }

    std::string out_path = "tests/fixtures/cpp_boundaries.json";
    if (argc > 1) {
        out_path = argv[1];
    }

    std::cout << "Loading IsochroneDB from " << data_dir << "..." << std::endl;
    Sampler imf(10000, 0.07, 100.0, chabrier_imf, 1);
    ParityIsochroneDB db(data_dir + "Isochrones/", "padova/", "DCMC", "J,J-Ks", 1);
    db.setimf(&imf);

    std::ofstream out(out_path.c_str());
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << out_path << std::endl;
        return 1;
    }

    out << std::setprecision(17);
    out << "{\n";

    // ------------------------------------------------------------------------
    // 1. Leaf-cutoff witnesses
    // ------------------------------------------------------------------------
    std::cout << "Generating leaf-cutoff witnesses..." << std::endl;
    out << "  \"leaf_cutoff_witnesses\": [\n";
    {
        // Setup setIsochrones1 with maxV = 5.0 (scenario)
        db.setIsochrones1(11e9, 1e9, -0.78, 0.3, 5.0, 1000.0, 0);

        // Find track for age=10.0, feh=-1.6766936096
        const auto& age_grid = db.getAgeGrid();
        const auto& feh_grid = db.getFeHGrid();
        int age_i = -1;
        for (size_t i = 0; i < age_grid.size(); ++i) {
            if (std::abs(age_grid[i] - 10.0) < 1e-4) { age_i = i; break; }
        }
        int feh_j = -1;
        for (size_t j = 0; j < feh_grid.size(); ++j) {
            if (std::abs(feh_grid[j] - (-1.6766936096)) < 1e-4) { feh_j = j; break; }
        }
        int track_idx = db.getTrackIndex(age_i, feh_j);
        double cutoff = db.getTrackMMin(track_idx);

        // Sub-test 1: candidate mass 0.639620019 (below cutoff ~0.639621 -> rejected)
        {
            StarParticle s1;
            s1.age() = age_grid[age_i];
            s1.feh() = feh_grid[feh_j];
            s1.smass() = 0.639620019;
            db.interpolateStar(s1);
            bool accepted = (s1.mag(0) < 1e5);

            out << "    {\n"
                << "      \"name\": \"leaf_cutoff_below_cutoff_rejected\",\n"
                << "      \"max_v\": 5.0,\n"
                << "      \"age\": " << s1.age() << ",\n"
                << "      \"feh\": " << s1.feh() << ",\n"
                << "      \"smass\": " << s1.smass() << ",\n"
                << "      \"track_idx\": " << track_idx << ",\n"
                << "      \"track_cutoff\": " << cutoff << ",\n"
                << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                << "      \"mags\": [" << s1.mag(0) << ", " << s1.mag(1) << ", " << s1.mag(2) << "],\n"
                << "      \"lum\": " << s1.lum() << ",\n"
                << "      \"teff\": " << s1.teff() << ",\n"
                << "      \"grav\": " << s1.grav() << "\n"
                << "    },\n";
        }
        // Sub-test 2: candidate mass 0.639622 (above cutoff -> accepted)
        {
            StarParticle s2;
            s2.age() = age_grid[age_i];
            s2.feh() = feh_grid[feh_j];
            s2.smass() = 0.639622;
            db.interpolateStar(s2);
            bool accepted = (s2.mag(0) < 1e5);

            out << "    {\n"
                << "      \"name\": \"leaf_cutoff_above_cutoff_accepted\",\n"
                << "      \"max_v\": 5.0,\n"
                << "      \"age\": " << s2.age() << ",\n"
                << "      \"feh\": " << s2.feh() << ",\n"
                << "      \"smass\": " << s2.smass() << ",\n"
                << "      \"track_idx\": " << track_idx << ",\n"
                << "      \"track_cutoff\": " << cutoff << ",\n"
                << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                << "      \"mags\": [" << s2.mag(0) << ", " << s2.mag(1) << ", " << s2.mag(2) << "],\n"
                << "      \"lum\": " << s2.lum() << ",\n"
                << "      \"teff\": " << s2.teff() << ",\n"
                << "      \"grav\": " << s2.grav() << "\n"
                << "    }\n";
        }
    }
    out << "  ],\n";

    // ------------------------------------------------------------------------
    // 2. Float-quantization witnesses
    // ------------------------------------------------------------------------
    std::cout << "Generating float-quantization witnesses..." << std::endl;
    out << "  \"float_transition_witnesses\": [\n";
    {
        double test_ages[2] = { 10.08999999, 10.09000001 };
        for (int k = 0; k < 2; ++k) {
            double age = test_ages[k];
            double feh = 0.0;
            double smass = 1.0;

            std::vector<double> appended(9, 0.0);
            db.interpolateTGM(static_cast<float>(age), static_cast<float>(feh), static_cast<float>(smass), appended);

            StarParticle star;
            star.age() = age;
            star.feh() = feh;
            star.smass() = smass;
            db.interpolateStar(star);

            out << "    {\n"
                << "      \"name\": \"" << (k == 0 ? "float_rounding_just_below_10_09" : "float_rounding_just_above_10_09") << "\",\n"
                << "      \"log_age\": " << age << ",\n"
                << "      \"feh\": " << feh << ",\n"
                << "      \"smass\": " << smass << ",\n"
                << "      \"tgm\": [";
            for (size_t i = 0; i < appended.size(); ++i) {
                out << appended[i] << (i + 1 < appended.size() ? ", " : "");
            }
            out << "],\n"
                << "      \"star_mags\": [" << star.mag(0) << ", " << star.mag(1) << ", " << star.mag(2) << "],\n"
                << "      \"lum\": " << star.lum() << ",\n"
                << "      \"teff\": " << star.teff() << ",\n"
                << "      \"grav\": " << star.grav() << "\n"
                << "    }" << (k + 1 < 2 ? ",\n" : "\n");
        }
    }
    out << "  ],\n";

    // ------------------------------------------------------------------------
    // 3. Every Padova age grid point ± 1e-7 and half-grid points
    // ------------------------------------------------------------------------
    std::cout << "Generating Padova age grid ± 1e-7 and half-grid points..." << std::endl;
    out << "  \"padova_age_grid\": [\n";
    {
        const auto& age_grid = db.getAgeGrid();
        bool first_entry = true;
        double feh_samples[3] = { -1.0, 0.0, 0.2 };

        // Grid points ± 1e-7
        for (size_t i = 0; i < age_grid.size(); ++i) {
            double base_age = age_grid[i];
            double offsets[3] = { -1e-7, 0.0, 1e-7 };
            const char* tags[3] = { "minus_1e7", "exact", "plus_1e7" };

            for (int o = 0; o < 3; ++o) {
                double age = base_age + offsets[o];
                for (int f = 0; f < 3; ++f) {
                    double feh = feh_samples[f];
                    double smass = 1.0;

                    std::vector<double> tgm(9, 0.0);
                    db.interpolateTGM(static_cast<float>(age), static_cast<float>(feh), static_cast<float>(smass), tgm);

                    StarParticle star;
                    star.age() = age;
                    star.feh() = feh;
                    star.smass() = smass;
                    db.interpolateStar(star);
                    bool accepted = (star.mag(0) < 1e5);

                    if (!first_entry) out << ",\n";
                    first_entry = false;

                    out << "    {\n"
                        << "      \"grid_idx\": " << i << ",\n"
                        << "      \"tag\": \"" << tags[o] << "\",\n"
                        << "      \"age\": " << age << ",\n"
                        << "      \"feh\": " << feh << ",\n"
                        << "      \"smass\": " << smass << ",\n"
                        << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                        << "      \"mags\": [" << star.mag(0) << ", " << star.mag(1) << ", " << star.mag(2) << "],\n"
                        << "      \"lum\": " << star.lum() << ",\n"
                        << "      \"teff\": " << star.teff() << ",\n"
                        << "      \"grav\": " << star.grav() << ",\n"
                        << "      \"tgm\": [";
                    for (size_t t = 0; t < tgm.size(); ++t) {
                        out << tgm[t] << (t + 1 < tgm.size() ? ", " : "");
                    }
                    out << "]\n    }";
                }
            }
        }

        // Half-grid points
        for (size_t i = 0; i + 1 < age_grid.size(); ++i) {
            double mid_age = 0.5 * (age_grid[i] + age_grid[i + 1]);
            for (int f = 0; f < 3; ++f) {
                double feh = feh_samples[f];
                double smass = 1.0;

                std::vector<double> tgm(9, 0.0);
                db.interpolateTGM(static_cast<float>(mid_age), static_cast<float>(feh), static_cast<float>(smass), tgm);

                StarParticle star;
                star.age() = mid_age;
                star.feh() = feh;
                star.smass() = smass;
                db.interpolateStar(star);
                bool accepted = (star.mag(0) < 1e5);

                if (!first_entry) out << ",\n";
                first_entry = false;

                out << "    {\n"
                    << "      \"grid_idx\": " << i << ",\n"
                    << "      \"tag\": \"midpoint\",\n"
                    << "      \"age\": " << mid_age << ",\n"
                    << "      \"feh\": " << feh << ",\n"
                    << "      \"smass\": " << smass << ",\n"
                    << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                    << "      \"mags\": [" << star.mag(0) << ", " << star.mag(1) << ", " << star.mag(2) << "],\n"
                    << "      \"lum\": " << star.lum() << ",\n"
                    << "      \"teff\": " << star.teff() << ",\n"
                    << "      \"grav\": " << star.grav() << ",\n"
                    << "      \"tgm\": [";
                for (size_t t = 0; t < tgm.size(); ++t) {
                    out << tgm[t] << (t + 1 < tgm.size() ? ", " : "");
                }
                out << "]\n    }";
            }
        }
    }
    out << "\n  ],\n";

    // ------------------------------------------------------------------------
    // 4. Every FeH grid point ± 1e-7
    // ------------------------------------------------------------------------
    std::cout << "Generating FeH grid points ± 1e-7..." << std::endl;
    out << "  \"padova_feh_grid\": [\n";
    {
        const auto& feh_grid = db.getFeHGrid();
        bool first_entry = true;
        double age = 9.0;
        double smass = 1.0;

        for (size_t j = 0; j < feh_grid.size(); ++j) {
            double base_feh = feh_grid[j];
            double offsets[3] = { -1e-7, 0.0, 1e-7 };
            const char* tags[3] = { "minus_1e7", "exact", "plus_1e7" };

            for (int o = 0; o < 3; ++o) {
                double feh = base_feh + offsets[o];

                std::vector<double> tgm(9, 0.0);
                db.interpolateTGM(static_cast<float>(age), static_cast<float>(feh), static_cast<float>(smass), tgm);

                StarParticle star;
                star.age() = age;
                star.feh() = feh;
                star.smass() = smass;
                db.interpolateStar(star);
                bool accepted = (star.mag(0) < 1e5);

                if (!first_entry) out << ",\n";
                first_entry = false;

                out << "    {\n"
                    << "      \"grid_idx\": " << j << ",\n"
                    << "      \"tag\": \"" << tags[o] << "\",\n"
                    << "      \"age\": " << age << ",\n"
                    << "      \"feh\": " << feh << ",\n"
                    << "      \"smass\": " << smass << ",\n"
                    << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                    << "      \"mags\": [" << star.mag(0) << ", " << star.mag(1) << ", " << star.mag(2) << "],\n"
                    << "      \"lum\": " << star.lum() << ",\n"
                    << "      \"teff\": " << star.teff() << ",\n"
                    << "      \"grav\": " << star.grav() << ",\n"
                    << "      \"tgm\": [";
                for (size_t t = 0; t < tgm.size(); ++t) {
                    out << tgm[t] << (t + 1 < tgm.size() ? ", " : "");
                }
                out << "]\n    }";
            }
        }
    }
    out << "\n  ],\n";

    // ------------------------------------------------------------------------
    // 5. Vmon table values ± 0.005 and ± 0.02 for maxV cache hysteresis
    // ------------------------------------------------------------------------
    std::cout << "Generating Vmon cache hysteresis fixtures..." << std::endl;
    out << "  \"max_v_cache\": [\n";
    {
        // Use track age_idx = 100, feh_idx = 15
        int target_track = db.getTrackIndex(100, 15);
        const auto& vmon = db.getTrackVmon(target_track);
        double age_val = db.getAgeGrid()[100];
        double feh_val = db.getFeHGrid()[15];

        // Pick 3 representative Vmon values from the track
        size_t sample_indices[3] = { 10, vmon.size() / 2, vmon.size() - 10 };
        bool first_entry = true;

        for (int s = 0; s < 3; ++s) {
            size_t vi = sample_indices[s];
            if (vi >= vmon.size()) vi = vmon.size() - 1;
            double base_v = vmon[vi];

            // Sequence:
            // 1. base_v (updates)
            // 2. base_v + 0.005 (hysteresis <= 0.01: NO update)
            // 3. base_v - 0.005 (hysteresis <= 0.01: NO update)
            // 4. base_v + 0.02  (hysteresis > 0.01: UPDATES)
            // 5. base_v - 0.02  (hysteresis > 0.01: UPDATES)
            double seq[5] = { base_v, base_v + 0.005, base_v - 0.005, base_v + 0.02, base_v - 0.02 };
            const char* step_names[5] = {
                "base_v_set",
                "plus_0_005_hysteresis_no_change",
                "minus_0_005_hysteresis_no_change",
                "plus_0_02_exceeds_hysteresis_updates",
                "minus_0_02_exceeds_hysteresis_updates"
            };

            for (int step = 0; step < 5; ++step) {
                double v_in = seq[step];
                db.setIsochrones1(11e9, 1e9, feh_val, 0.3, v_in, 1000.0, 0);

                double eff_max_v = db.getCurrentMaxV();
                double m_min = db.getTrackMMin(target_track);

                // Test star just below m_min (rejected)
                StarParticle s_below;
                s_below.age() = age_val;
                s_below.feh() = feh_val;
                s_below.smass() = m_min - 1e-6;
                db.interpolateStar(s_below);
                bool below_acc = (s_below.mag(0) < 1e5);

                // Test star just above m_min (accepted)
                StarParticle s_above;
                s_above.age() = age_val;
                s_above.feh() = feh_val;
                s_above.smass() = m_min + 1e-6;
                db.interpolateStar(s_above);
                bool above_acc = (s_above.mag(0) < 1e5);

                if (!first_entry) out << ",\n";
                first_entry = false;

                out << "    {\n"
                    << "      \"step_name\": \"" << step_names[step] << "\",\n"
                    << "      \"max_v_input\": " << v_in << ",\n"
                    << "      \"effective_max_v\": " << eff_max_v << ",\n"
                    << "      \"track_idx\": " << target_track << ",\n"
                    << "      \"track_m_min\": " << m_min << ",\n"
                    << "      \"test_below_mass\": " << s_below.smass() << ",\n"
                    << "      \"below_accepted\": " << (below_acc ? "true" : "false") << ",\n"
                    << "      \"test_above_mass\": " << s_above.smass() << ",\n"
                    << "      \"above_accepted\": " << (above_acc ? "true" : "false") << ",\n"
                    << "      \"above_mags\": [" << s_above.mag(0) << ", " << s_above.mag(1) << ", " << s_above.mag(2) << "]\n"
                    << "    }";
            }
        }
    }
    out << "\n  ],\n";

    // ------------------------------------------------------------------------
    // 6. Multi-leaf sequence in Field 1, then reset in Field 2
    // ------------------------------------------------------------------------
    std::cout << "Generating multi-leaf order and reset fixtures..." << std::endl;
    out << "  \"leaf_order_and_reset\": [\n";
    {
        int target_track = db.getTrackIndex(100, 15);
        double age_val = db.getAgeGrid()[100];
        double feh_val = db.getFeHGrid()[15];
        bool first_entry = true;

        // --- Field 1: Fresh DB instance ---
        ParityIsochroneDB db_field1(data_dir + "Isochrones/", "padova/", "DCMC", "J,J-Ks", 1);
        db_field1.setimf(&imf);

        double leaf_max_v[5] = { 10.0, 10.005, 7.5, 12.0, 12.008 };
        for (int l = 0; l < 5; ++l) {
            double v_in = leaf_max_v[l];
            db_field1.setIsochrones1(11e9, 1e9, feh_val, 0.3, v_in, 1000.0, 0);

            double m_min = db_field1.getTrackMMin(target_track);
            StarParticle s;
            s.age() = age_val;
            s.feh() = feh_val;
            s.smass() = 0.8;
            db_field1.interpolateStar(s);
            bool accepted = (s.mag(0) < 1e5);

            if (!first_entry) out << ",\n";
            first_entry = false;

            out << "    {\n"
                << "      \"field_id\": 1,\n"
                << "      \"leaf_idx\": " << l << ",\n"
                << "      \"max_v_input\": " << v_in << ",\n"
                << "      \"effective_max_v\": " << db_field1.getCurrentMaxV() << ",\n"
                << "      \"sample_track_idx\": " << target_track << ",\n"
                << "      \"sample_track_m_min\": " << m_min << ",\n"
                << "      \"star_age\": " << s.age() << ",\n"
                << "      \"star_feh\": " << s.feh() << ",\n"
                << "      \"star_smass\": " << s.smass() << ",\n"
                << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                << "      \"mags\": [" << s.mag(0) << ", " << s.mag(1) << ", " << s.mag(2) << "]\n"
                << "    }";
        }

        // --- Field 2: Fresh DB instance (Reset) ---
        ParityIsochroneDB db_field2(data_dir + "Isochrones/", "padova/", "DCMC", "J,J-Ks", 1);
        db_field2.setimf(&imf);

        // Before applying any leaf, verify initial reset state
        {
            double m_min = db_field2.getTrackMMin(target_track);
            StarParticle s;
            s.age() = age_val;
            s.feh() = feh_val;
            s.smass() = 0.8;
            db_field2.interpolateStar(s);
            bool accepted = (s.mag(0) < 1e5);

            if (!first_entry) out << ",\n";
            first_entry = false;

            out << "    {\n"
                << "      \"field_id\": 2,\n"
                << "      \"leaf_idx\": 0,\n"
                << "      \"max_v_input\": 30.0,\n"
                << "      \"effective_max_v\": " << db_field2.getCurrentMaxV() << ",\n"
                << "      \"sample_track_idx\": " << target_track << ",\n"
                << "      \"sample_track_m_min\": " << m_min << ",\n"
                << "      \"star_age\": " << s.age() << ",\n"
                << "      \"star_feh\": " << s.feh() << ",\n"
                << "      \"star_smass\": " << s.smass() << ",\n"
                << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                << "      \"mags\": [" << s.mag(0) << ", " << s.mag(1) << ", " << s.mag(2) << "]\n"
                << "    }";
        }

        // Field 2 Leaf 1: maxV = 9.0
        {
            db_field2.setIsochrones1(11e9, 1e9, feh_val, 0.3, 9.0, 1000.0, 0);
            double m_min = db_field2.getTrackMMin(target_track);
            StarParticle s;
            s.age() = age_val;
            s.feh() = feh_val;
            s.smass() = 0.8;
            db_field2.interpolateStar(s);
            bool accepted = (s.mag(0) < 1e5);

            if (!first_entry) out << ",\n";
            first_entry = false;

            out << "    {\n"
                << "      \"field_id\": 2,\n"
                << "      \"leaf_idx\": 1,\n"
                << "      \"max_v_input\": 9.0,\n"
                << "      \"effective_max_v\": " << db_field2.getCurrentMaxV() << ",\n"
                << "      \"sample_track_idx\": " << target_track << ",\n"
                << "      \"sample_track_m_min\": " << m_min << ",\n"
                << "      \"star_age\": " << s.age() << ",\n"
                << "      \"star_feh\": " << s.feh() << ",\n"
                << "      \"star_smass\": " << s.smass() << ",\n"
                << "      \"accepted\": " << (accepted ? "true" : "false") << ",\n"
                << "      \"mags\": [" << s.mag(0) << ", " << s.mag(1) << ", " << s.mag(2) << "]\n"
                << "    }";
        }
    }
    out << "\n  ],\n";

    // ------------------------------------------------------------------------
    // 7. ThinDisk0 density with optionE = 0
    // ------------------------------------------------------------------------
    std::cout << "Generating ThinDisk0 (E=0) density fixtures..." << std::endl;
    out << "  \"thindisk0_density_e0\": [\n";
    {
        Interp vcirc(data_dir + "Model/vcirc.dat");
        ThinDisk0 td0(1, 0, data_dir, &vcirc);

        double x_vals[7] = { -15.0, -8.0, -2.0, 0.0, 3.0, 8.0, 15.0 };
        double y_vals[5] = { -10.0, -4.0, 0.0, 4.0, 10.0 };
        double z_vals[6] = { -2.0, -0.5, 0.0, 0.015, 0.5, 2.0 };
        double age_vals[4] = { 1e8, 1e9, 5e9, 1e10 };

        bool first_entry = true;
        for (int ix = 0; ix < 7; ++ix) {
            for (int iy = 0; iy < 5; ++iy) {
                for (int iz = 0; iz < 6; ++iz) {
                    for (int ia = 0; ia < 4; ++ia) {
                        double pos[3] = { x_vals[ix], y_vals[iy], z_vals[iz] };
                        double age = age_vals[ia];
                        double rho = td0.density(pos, age);

                        if (!first_entry) out << ",\n";
                        first_entry = false;

                        out << "    {\n"
                            << "      \"pos\": [" << pos[0] << ", " << pos[1] << ", " << pos[2] << "],\n"
                            << "      \"age\": " << age << ",\n"
                            << "      \"density\": " << rho << "\n"
                            << "    }";
                    }
                }
            }
        }
    }
    out << "\n  ]\n";

    out << "}\n";
    out.close();

    std::cout << "Parity fixtures written successfully to " << out_path << std::endl;
    return 0;
}
