// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from Galaxia.cpp
//! High-throughput batch processing mode for Galaxia survey generation.
//!
//! Loads all resident data (IsochroneDB, ExtinctionMap, VCirc, and all 10 BHTree octrees)
//! into memory **EXACTLY ONCE** and processes field targets across N worker threads.

use crate::backend::{create_backend, BackendKind};
use crate::coords::{icrs_to_galactic, seed_for_tic};
use crate::generator::Generator;
use crate::params::Params;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Configuration options for batch generation.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Path to input CSV chunk file (with header `tic,ra,dec`; `tic` is an integer field id).
    pub csv_path: PathBuf,
    /// Directory where per-field output files (`gx_<tic>.ebf`) will be saved.
    pub out_dir: PathBuf,
    /// Path to GalaxiaData directory.
    pub data_dir: PathBuf,
    /// Accelerator backend to execute on.
    pub backend: BackendKind,
    /// Optional extincted Tmag ceiling cut (e.g. 22.5).
    pub tmag_ceiling: Option<f64>,
    /// Number of worker threads (default 1).
    pub threads: usize,
}

/// Target stellar field record parsed from CSV.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldRecord {
    pub tic: u64,
    pub ra: f64,
    pub dec: f64,
}

/// Parse fields CSV file with header `tic,ra,dec` (and optional additional columns).
pub fn parse_chunk_csv(path: impl AsRef<Path>) -> anyhow::Result<Vec<FieldRecord>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        anyhow::anyhow!(
            "Failed to open CSV file '{}': {}",
            path.as_ref().display(),
            e
        )
    })?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let header_line = match lines.next() {
        Some(Ok(line)) => line,
        Some(Err(e)) => return Err(anyhow::anyhow!("Error reading CSV header: {}", e)),
        None => return Err(anyhow::anyhow!("CSV file is empty")),
    };

    let header_cols: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_lowercase())
        .collect();

    let tic_idx = header_cols
        .iter()
        .position(|c| c == "tic" || c == "tic_id")
        .ok_or_else(|| anyhow::anyhow!("Missing 'tic' column in CSV header: {:?}", header_cols))?;

    let ra_idx = header_cols
        .iter()
        .position(|c| c == "ra" || c == "ra_deg")
        .ok_or_else(|| anyhow::anyhow!("Missing 'ra' column in CSV header: {:?}", header_cols))?;

    let dec_idx = header_cols
        .iter()
        .position(|c| c == "dec" || c == "dec_deg")
        .ok_or_else(|| anyhow::anyhow!("Missing 'dec' column in CSV header: {:?}", header_cols))?;

    let mut records = Vec::new();

    for (line_num, line_res) in lines.enumerate() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed
            .split(',')
            .map(|s| s.trim().trim_matches('"'))
            .collect();
        let max_idx = tic_idx.max(ra_idx).max(dec_idx);
        if parts.len() <= max_idx {
            anyhow::bail!(
                "Malformed CSV line {}: expected at least {} columns, got {}",
                line_num + 2,
                max_idx + 1,
                parts.len()
            );
        }

        let tic: u64 = parts[tic_idx].parse().map_err(|e| {
            anyhow::anyhow!(
                "Line {}: invalid field-id integer '{}': {}",
                line_num + 2,
                parts[tic_idx],
                e
            )
        })?;

        let ra: f64 = parts[ra_idx].parse().map_err(|e| {
            anyhow::anyhow!(
                "Line {}: invalid RA float '{}': {}",
                line_num + 2,
                parts[ra_idx],
                e
            )
        })?;

        if !ra.is_finite() || ra < 0.0 || ra > 360.0 {
            anyhow::bail!(
                "Line {}: RA coordinate {} is out of valid range [0, 360] or non-finite",
                line_num + 2,
                ra
            );
        }

        let dec: f64 = parts[dec_idx].parse().map_err(|e| {
            anyhow::anyhow!(
                "Line {}: invalid Dec float '{}': {}",
                line_num + 2,
                parts[dec_idx],
                e
            )
        })?;

        if !dec.is_finite() || dec < -90.0 || dec > 90.0 {
            anyhow::bail!(
                "Line {}: Dec coordinate {} is out of valid range [-90, 90] or non-finite",
                line_num + 2,
                dec
            );
        }

        records.push(FieldRecord { tic, ra, dec });
    }

    Ok(records)
}

/// Setup survey parameters for a single field cone.
fn setup_field_params(
    field: &FieldRecord,
    data_dir: &Path,
    out_dir: &Path,
    seed: u32,
    l: f64,
    b: f64,
    tmag_ceiling: Option<f64>,
) -> Params {
    let mut params = Params::default();
    params.code_data_dir = format!("{}/", data_dir.display());
    params.output_dir = format!("{}/", out_dir.display());
    params.output_file = format!("gx_{}", field.tic);
    params.geometry_option = 1; // cone
    params.longitude = l;
    params.latitude = b;
    params.survey_area = 0.1; // 0.1 deg²
    params.app_mag_limits = [-100.0, 22.0]; // intrinsic J <= 22
    params.abs_mag_limits = [-1000.0, 1000.0];
    params.color_limits = [-1000.0, 1000.0];
    params.photo_sys = "DCMC".to_string();
    params.magcolor_names = "J,J-Ks".to_string();
    params.f_sample = 1.0;
    params.pop_id = -1; // all 10 populations
    params.warp_flare_on = 1;
    params.r_max = 1000.0;
    params.star_type = 0;
    params.photo_error = 0;
    params.seed = seed as i32;
    params.tmag_ceiling = tmag_ceiling;
    params
}

/// Run batch execution of fields listed in CSV.
pub fn run_batch(config: &BatchConfig) -> anyhow::Result<()> {
    let num_threads = config.threads.max(1);

    println!("\n========================================================");
    println!("galaxia-rs batch mode");
    println!("========================================================");
    println!("CSV chunk file:  {}", config.csv_path.display());
    println!("Output dir:      {}", config.out_dir.display());
    println!("Data dir:        {}", config.data_dir.display());
    println!("Backend:         {}", config.backend);
    println!("Threads:         {}", num_threads);
    if let Some(ceiling) = config.tmag_ceiling {
        println!("Tmag ceiling:    <= {:.2} mag", ceiling);
    } else {
        println!("Tmag ceiling:    none (all J <= 22 intrinsic stars)");
    }

    // 1. Parse CSV targets
    let fields = parse_chunk_csv(&config.csv_path)?;
    println!("Loaded {} target fields from CSV.", fields.len());
    if fields.is_empty() {
        println!("No fields to process.");
        return Ok(());
    }

    // 2. Ensure output directory exists
    std::fs::create_dir_all(&config.out_dir).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create output directory '{}': {}",
            config.out_dir.display(),
            e
        )
    })?;

    // 3. Initialize backend (strict: fail with error if requested backend fails)
    let backend_inst = create_backend(config.backend)?;

    // 4. Load resident data (Isochrones, Extinction, VCirc, 10 BHTrees) EXACTLY ONCE
    println!("\nLoading resident databases into memory (ONE-TIME fixed startup cost)...");
    let load_start = Instant::now();

    let generator = Generator::with_resident_data(&config.data_dir, backend_inst)
        .map_err(|e| anyhow::anyhow!("Failed to initialize resident data: {}", e))?;

    let load_duration = load_start.elapsed();

    // Calculate memory footprint
    let num_trees = generator.resident_trees.as_ref().map_or(0, |t| t.len());
    let mut total_nodes = 0;
    let mut total_leaves = 0;
    if let Some(trees) = &generator.resident_trees {
        for t in trees {
            total_nodes += t.nodes.len();
            total_leaves += t.leafs.len();
        }
    }
    // BHNode is 96 bytes (12 * 8 bytes f64)
    let tree_bytes = (total_nodes + total_leaves) * 96;
    let tree_mb = tree_bytes as f64 / (1024.0 * 1024.0);
    let total_resident_mb = tree_mb + 200.0; // ~40MB Isochrones + ~160MB Extinction

    println!(
        "Resident data loaded in {:.2}s. {} BHTrees: {} nodes, {} leaves ({:.1} MB RAM). Total resident memory: ~{:.1} MB RAM.",
        load_duration.as_secs_f64(),
        num_trees,
        total_nodes,
        total_leaves,
        tree_mb,
        total_resident_mb
    );

    println!(
        "\nStarting batch generation across {} fields ({} worker thread{})...",
        fields.len(),
        num_threads,
        if num_threads == 1 { "" } else { "s" }
    );
    println!("--------------------------------------------------------");

    let batch_start = Instant::now();
    let total_fields = fields.len();

    let mut succeeded_tics: Vec<u64> = Vec::new();
    let (succeeded_count, failed_count, total_stars_retained, total_stars_raw) = if num_threads <= 1
    {
        // Single-thread path (zero synchronisation overhead)
        let mut succeeded = 0;
        let mut failed = 0;
        let mut stars_retained = 0;
        let mut stars_raw = 0;

        for (idx, field) in fields.iter().enumerate() {
            let field_start = Instant::now();
            let seed = seed_for_tic(field.tic);
            // 6-decimal rounding for coordinates
            let ra = (field.ra * 1e6).round() / 1e6;
            let dec = (field.dec * 1e6).round() / 1e6;
            let (l_raw, b_raw) = icrs_to_galactic(ra, dec);
            let l = (l_raw * 1e6).round() / 1e6;
            let b = (b_raw * 1e6).round() / 1e6;
            let params = setup_field_params(
                field,
                &config.data_dir,
                &config.out_dir,
                seed,
                l,
                b,
                config.tmag_ceiling,
            );

            let result = catch_unwind(AssertUnwindSafe(|| {
                generator.run_model_ext_quiet(&params, config.tmag_ceiling, true)
            }));

            match result {
                Ok(Ok((stats, raw_count))) => {
                    succeeded += 1;
                    succeeded_tics.push(field.tic);
                    stars_retained += stats.total_stars;
                    stars_raw += raw_count;
                    let elapsed_field = field_start.elapsed();
                    let cumulative_time = batch_start.elapsed();
                    let fph = (succeeded as f64) / cumulative_time.as_secs_f64() * 3600.0;

                    println!(
                        "[{:>4}/{:<4}] field {:<10} (l={:>6.2}°, b={:>6.2}°, seed={:>5}): spawned {:>6} stars ({} raw) in {:>5.2}s | throughput: {:>6.1} fields/h",
                        idx + 1,
                        total_fields,
                        field.tic,
                        l,
                        b,
                        seed,
                        stats.total_stars,
                        raw_count,
                        elapsed_field.as_secs_f64(),
                        fph
                    );
                }
                Ok(Err(e)) => {
                    failed += 1;
                    eprintln!(
                        "[batch] ERROR: Field {} (ra={:.6}, dec={:.6}) FAILED: {}. Continuing batch...",
                        field.tic, field.ra, field.dec, e
                    );
                }
                Err(panic_payload) => {
                    failed += 1;
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic in worker thread".to_string()
                    };
                    eprintln!(
                        "[batch] ERROR: Field {} (ra={:.6}, dec={:.6}) FAILED: panic: {}. Continuing batch...",
                        field.tic, field.ra, field.dec, panic_msg
                    );
                }
            }
        }
        (succeeded, failed, stars_retained, stars_raw)
    } else {
        // Multi-threaded path: Arc-shared resident Generator and dynamic atomic work queue
        let generator = Arc::new(generator);
        let fields = Arc::new(fields);
        let stdout_mutex = Arc::new(Mutex::new(()));
        let succeeded_tics_mt = Arc::new(Mutex::new(Vec::new()));
        let next_field_idx = Arc::new(AtomicUsize::new(0));
        let completed_count = Arc::new(AtomicUsize::new(0));
        let succeeded_count = Arc::new(AtomicUsize::new(0));
        let failed_count = Arc::new(AtomicUsize::new(0));
        let total_stars_retained = Arc::new(AtomicUsize::new(0));
        let total_stars_raw = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(num_threads);
        for thread_id in 0..num_threads {
            let generator = Arc::clone(&generator);
            let fields = Arc::clone(&fields);
            let stdout_mutex = Arc::clone(&stdout_mutex);
            let succeeded_tics_mt = Arc::clone(&succeeded_tics_mt);
            let next_field_idx = Arc::clone(&next_field_idx);
            let completed_count = Arc::clone(&completed_count);
            let succeeded_count = Arc::clone(&succeeded_count);
            let failed_count = Arc::clone(&failed_count);
            let total_stars_retained = Arc::clone(&total_stars_retained);
            let total_stars_raw = Arc::clone(&total_stars_raw);
            let data_dir = config.data_dir.clone();
            let out_dir = config.out_dir.clone();
            let tmag_ceiling = config.tmag_ceiling;

            let handle = std::thread::Builder::new()
                .name(format!("galaxia-worker-{}", thread_id))
                .spawn(move || {
                    loop {
                        let idx = next_field_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_fields {
                            break;
                        }

                        let field = &fields[idx];
                        let field_start = Instant::now();
                        let seed = seed_for_tic(field.tic);
                        // 6-decimal rounding for coordinates
                        let ra = (field.ra * 1e6).round() / 1e6;
                        let dec = (field.dec * 1e6).round() / 1e6;
                        let (l_raw, b_raw) = icrs_to_galactic(ra, dec);
                        let l = (l_raw * 1e6).round() / 1e6;
                        let b = (b_raw * 1e6).round() / 1e6;
                        let params = setup_field_params(field, &data_dir, &out_dir, seed, l, b, tmag_ceiling);

                        let gen_ref = Arc::clone(&generator);
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            gen_ref.run_model_ext_quiet(&params, tmag_ceiling, true)
                        }));

                        match result {
                            Ok(Ok((stats, raw_count))) => {
                                succeeded_count.fetch_add(1, Ordering::SeqCst);
                                total_stars_retained.fetch_add(stats.total_stars, Ordering::SeqCst);
                                total_stars_raw.fetch_add(raw_count, Ordering::SeqCst);
                                succeeded_tics_mt.lock().unwrap().push(field.tic);

                                let done = completed_count.fetch_add(1, Ordering::SeqCst) + 1;
                                let elapsed_field = field_start.elapsed();
                                let cumulative_time = batch_start.elapsed();
                                let fph = (done as f64) / cumulative_time.as_secs_f64() * 3600.0;

                                let _lock = stdout_mutex.lock().unwrap();
                                println!(
                                    "[{:>4}/{:<4}] field {:<10} (l={:>6.2}°, b={:>6.2}°, seed={:>5}): spawned {:>6} stars ({} raw) in {:>5.2}s | throughput: {:>6.1} fields/h",
                                    done,
                                    total_fields,
                                    field.tic,
                                    l,
                                    b,
                                    seed,
                                    stats.total_stars,
                                    raw_count,
                                    elapsed_field.as_secs_f64(),
                                    fph
                                );
                            }
                            Ok(Err(e)) => {
                                failed_count.fetch_add(1, Ordering::SeqCst);
                                let _done = completed_count.fetch_add(1, Ordering::SeqCst) + 1;
                                let _lock = stdout_mutex.lock().unwrap();
                                eprintln!(
                                    "[batch] ERROR: Field {} (ra={:.6}, dec={:.6}) FAILED: {}. Continuing batch...",
                                    field.tic, field.ra, field.dec, e
                                );
                            }
                            Err(panic_payload) => {
                                failed_count.fetch_add(1, Ordering::SeqCst);
                                let _done = completed_count.fetch_add(1, Ordering::SeqCst) + 1;
                                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                                    s.to_string()
                                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                                    s.clone()
                                } else {
                                    "unknown panic in worker thread".to_string()
                                };
                                let _lock = stdout_mutex.lock().unwrap();
                                eprintln!(
                                    "[batch] ERROR: Field {} (ra={:.6}, dec={:.6}) FAILED: panic: {}. Continuing batch...",
                                    field.tic, field.ra, field.dec, panic_msg
                                );
                            }
                        }
                    }
                })
                .expect("Failed to spawn worker thread");
            handles.push(handle);
        }

        let mut join_failed = false;
        for handle in handles {
            if let Err(e) = handle.join() {
                eprintln!("[batch] Worker thread join error: {:?}", e);
                join_failed = true;
            }
        }
        if join_failed {
            eprintln!("[batch] ERROR: One or more worker threads panicked or failed to join.");
            anyhow::bail!("Batch processing failed: worker thread join error");
        }

        succeeded_tics = succeeded_tics_mt.lock().unwrap().clone();

        (
            succeeded_count.load(Ordering::SeqCst),
            failed_count.load(Ordering::SeqCst),
            total_stars_retained.load(Ordering::SeqCst),
            total_stars_raw.load(Ordering::SeqCst),
        )
    };

    let total_batch_duration = batch_start.elapsed();
    let overall_fph = (succeeded_count as f64) / total_batch_duration.as_secs_f64() * 3600.0;

    println!("\n========================================================");
    println!("Batch Processing Summary");
    println!("========================================================");
    println!("Total fields:       {}", total_fields);
    println!("Succeeded:          {}", succeeded_count);
    println!("Failed:             {}", failed_count);
    println!("Total stars saved:  {}", total_stars_retained);
    println!("Total raw stars:    {}", total_stars_raw);
    println!(
        "Batch wall-clock:   {:.2}s",
        total_batch_duration.as_secs_f64()
    );
    println!(
        "Average time/field: {:.3}s",
        total_batch_duration.as_secs_f64() / (total_fields as f64)
    );
    println!("Throughput:         {:.1} fields/hour", overall_fph);
    println!("Resident RAM:       ~{:.1} MB", total_resident_mb);
    println!("========================================================\n");

    // Verify accounting: succeeded + failed == total
    if succeeded_count + failed_count != total_fields {
        eprintln!(
            "Error: accounting mismatch: succeeded ({}) + failed ({}) != total ({})",
            succeeded_count, failed_count, total_fields
        );
        anyhow::bail!(
            "Batch processing failed: accounting mismatch (succeeded={}, failed={}, total={})",
            succeeded_count,
            failed_count,
            total_fields
        );
    }

    // Verify all succeeded fields actually produced non-empty EBF files on disk
    let mut missing_ebf_count = 0;
    for tic in &succeeded_tics {
        let ebf_path = config.out_dir.join(format!("gx_{}.ebf", tic));
        if !ebf_path.is_file() {
            eprintln!(
                "Error: Field {} succeeded but final EBF file does not exist: {:?}",
                tic, ebf_path
            );
            missing_ebf_count += 1;
        } else if let Ok(meta) = ebf_path.metadata() {
            if meta.len() == 0 {
                eprintln!(
                    "Error: Field {} succeeded but final EBF file is 0 bytes: {:?}",
                    tic, ebf_path
                );
                missing_ebf_count += 1;
            }
        }
    }
    if missing_ebf_count > 0 {
        anyhow::bail!(
            "Batch processing failed: {} field(s) reported as succeeded have missing or empty EBF files",
            missing_ebf_count
        );
    }

    if failed_count > 0 {
        eprintln!(
            "Error: {} out of {} fields encountered errors during generation.",
            failed_count, total_fields
        );
        anyhow::bail!(
            "Batch processing failed: {} out of {} field(s) failed",
            failed_count,
            total_fields
        );
    }

    Ok(())
}
