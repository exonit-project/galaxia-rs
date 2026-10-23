// SPDX-License-Identifier: AGPL-3.0-or-later
use galaxia_rs::backend::{create_backend, version_string, BackendKind};
use galaxia_rs::batch::{run_batch, BatchConfig};
use galaxia_rs::generator::Generator;
use galaxia_rs::params::Params;
use std::env;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

fn print_usage() {
    println!("\nNAME:");
    println!("\t galaxia-rs-0.2.2 - a fast code to generate a synthetic galaxy survey");
    println!("\nUSAGE:");
    println!("\t galaxia-rs -r parameterfile [--backend <cpu>] [--tmag-ceiling 22.5]");
    println!("\t galaxia-rs batch --csv <chunk.csv> --out-dir <dir> [--data-dir <dir>] [--backend <cpu>] [--tmag-ceiling 22.5] [--threads <N>]");
    println!("\t galaxia-rs -s parameterfile");
    println!("\t galaxia-rs --version [--backend <cpu>] [--tmag-ceiling <float>]");
    println!("\t galaxia-rs --help");
    println!("\t galaxia-rs --copyright");
    println!("DESCRIPTION:");
    println!(
        "\t -r                  run the code to generate stellar data for a single parameter file"
    );
    println!("\t batch               run high-throughput batch generation over a fields CSV (data loaded once, optional --threads)");
    println!("\t -s                  initial setup to generate BHTREE files (NOT implemented; use the BHTree files of the original distribution)");
    println!("\t --backend           backend: 'cpu' (default; the only validated backend, fp64)");
    println!("\t --allow-fp32-test   allow the EXPERIMENTAL GPU backends (cuda, opencl; need non-default cargo features; single-precision/unvalidated, NOT for science)");
    println!("\t --tmag-ceiling      filter stars with extincted Tmag <= ceiling cut");
    println!("\t --copyright         print copyright information");
    println!("\t --help              print this help message");
}

fn print_copyright() {
    println!("\n--------------------------------------------------------");
    println!("galaxia-rs: Rust port of Galaxia (Milky Way synthetic modelling)");
    println!("Original C++ code Copyright (c) 2012 by Sanjib Sharma and Joss Bland-Hawthorn");
    println!("Rust port Copyright (c) 2026 by Marc Balcells (see NOTICE)");
    println!("\nLicensed under the GNU Affero General Public License v3 (AGPL-3.0-or-later).");
    println!("--------------------------------------------------------\n");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    // Default backend is CPU (cpu-fp64, the only validated backend)
    let default_backend = BackendKind::Cpu;
    let mut backend_kind = default_backend;
    let mut is_version = false;
    let mut is_help = false;
    let mut is_copyright = false;
    let mut allow_fp32_test = false;
    let mut global_tmag_ceiling: Option<f64> = None;

    // Pre-scan for global flags like --backend, --allow-fp32-test, --tmag-ceiling, --version, --help
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--backend" && i + 1 < args.len() {
            backend_kind = match BackendKind::from_str(&args[i + 1]) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            };
            i += 2;
            continue;
        } else if args[i] == "--allow-fp32-test" {
            allow_fp32_test = true;
        } else if args[i] == "--tmag-ceiling" {
            if i + 1 < args.len() {
                match args[i + 1].parse::<f64>() {
                    Ok(v) => global_tmag_ceiling = Some(v),
                    Err(e) => {
                        eprintln!(
                            "Error: Invalid --tmag-ceiling value '{}': {}",
                            args[i + 1],
                            e
                        );
                        process::exit(1);
                    }
                }
                i += 2;
                continue;
            } else {
                eprintln!("Error: --tmag-ceiling requires a numeric float value");
                process::exit(1);
            }
        } else if args[i] == "--version" {
            is_version = true;
        } else if args[i] == "--help" || args[i] == "-h" {
            is_help = true;
        } else if args[i] == "--copyright" {
            is_copyright = true;
        }
        i += 1;
    }

    if is_help {
        print_usage();
        process::exit(0);
    }
    if is_copyright {
        print_copyright();
        process::exit(0);
    }

    // Backend gate: refuse any backend other than CPU unless --allow-fp32-test is given
    if backend_kind != BackendKind::Cpu && !allow_fp32_test {
        eprintln!(
            "Error: backend '{}' is EXPERIMENTAL and not validated for science (only cpu is validated). Pass --allow-fp32-test to enable it for testing.",
            backend_kind
        );
        process::exit(1);
    }

    if is_version {
        println!("{}", version_string(backend_kind, global_tmag_ceiling));
        process::exit(0);
    }

    // Command dispatch: "batch" subcommand or "-r" / "-s" flags
    let first_arg = &args[1];

    if first_arg == "batch" {
        let mut csv_path: Option<PathBuf> = None;
        let mut out_dir: Option<PathBuf> = None;
        let mut data_dir: Option<PathBuf> = None;
        let mut tmag_ceiling: Option<f64> = None;
        let mut threads: usize = 1;

        let mut j = 2;
        while j < args.len() {
            match args[j].as_str() {
                "--csv" => {
                    if j + 1 < args.len() {
                        csv_path = Some(PathBuf::from(&args[j + 1]));
                        j += 2;
                    } else {
                        eprintln!("Error: --csv requires a file path");
                        process::exit(1);
                    }
                }
                "--out-dir" => {
                    if j + 1 < args.len() {
                        out_dir = Some(PathBuf::from(&args[j + 1]));
                        j += 2;
                    } else {
                        eprintln!("Error: --out-dir requires a directory path");
                        process::exit(1);
                    }
                }
                "--data-dir" => {
                    if j + 1 < args.len() {
                        data_dir = Some(PathBuf::from(&args[j + 1]));
                        j += 2;
                    } else {
                        eprintln!("Error: --data-dir requires a directory path");
                        process::exit(1);
                    }
                }
                "--tmag-ceiling" => {
                    if j + 1 < args.len() {
                        tmag_ceiling = match args[j + 1].parse::<f64>() {
                            Ok(v) => Some(v),
                            Err(e) => {
                                eprintln!(
                                    "Error: Invalid --tmag-ceiling value '{}': {}",
                                    args[j + 1],
                                    e
                                );
                                process::exit(1);
                            }
                        };
                        j += 2;
                    } else {
                        eprintln!("Error: --tmag-ceiling requires a numeric float value");
                        process::exit(1);
                    }
                }
                "--threads" => {
                    if j + 1 < args.len() {
                        threads = match args[j + 1].parse::<usize>() {
                            Ok(v) if v >= 1 => v,
                            Ok(_) => {
                                eprintln!("Error: --threads must be >= 1");
                                process::exit(1);
                            }
                            Err(e) => {
                                eprintln!(
                                    "Error: Invalid --threads value '{}': {}",
                                    args[j + 1],
                                    e
                                );
                                process::exit(1);
                            }
                        };
                        j += 2;
                    } else {
                        eprintln!("Error: --threads requires a positive integer value");
                        process::exit(1);
                    }
                }
                "--backend" => {
                    // Already scanned, skip value
                    j += 2;
                }
                "--allow-fp32-test" => {
                    // Already scanned, skip flag
                    j += 1;
                }
                other => {
                    eprintln!("Unknown argument in batch mode: {}", other);
                    print_usage();
                    process::exit(1);
                }
            }
        }

        let csv = match csv_path {
            Some(p) => p,
            None => {
                eprintln!("Error: 'batch' requires --csv <chunk.csv>");
                print_usage();
                process::exit(1);
            }
        };

        let out = match out_dir {
            Some(d) => d,
            None => {
                eprintln!("Error: 'batch' requires --out-dir <dir>");
                print_usage();
                process::exit(1);
            }
        };

        let data = data_dir.unwrap_or_else(|| PathBuf::from(Params::default().code_data_dir));

        let config = BatchConfig {
            csv_path: csv,
            out_dir: out,
            data_dir: data,
            backend: backend_kind,
            tmag_ceiling,
            threads,
        };

        if let Err(e) = run_batch(&config) {
            eprintln!("\nError in batch execution: {}", e);
            process::exit(1);
        }
        process::exit(0);
    }

    // Legacy / single-field run (-r parameterfile)
    let mut option: Option<char> = None;
    let mut param_file: Option<String> = None;

    let mut k = 1;
    while k < args.len() {
        let arg = &args[k];
        if arg == "--backend" || arg == "--tmag-ceiling" {
            k += 2;
            continue;
        } else if arg == "--allow-fp32-test"
            || arg == "--version"
            || arg == "--help"
            || arg == "--copyright"
        {
            k += 1;
            continue;
        } else if arg.starts_with('-') && !arg.starts_with("--") {
            let opt_char = arg.chars().nth(1).unwrap_or(' ');
            match opt_char {
                'r' => option = Some('r'),
                's' => option = Some('s'),
                'a' => option = Some('a'),
                'e' => option = Some('e'),
                _ => {
                    eprintln!("Unknown option: {}", arg);
                    print_usage();
                    process::exit(1);
                }
            }
        } else {
            param_file = Some(arg.clone());
        }
        k += 1;
    }

    match option {
        Some('r') => {
            let file_path = match param_file {
                Some(f) => f,
                None => {
                    eprintln!("Error: Parameter file not specified for -r");
                    print_usage();
                    process::exit(1);
                }
            };

            let mut params = match Params::from_file(&file_path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error loading parameter file '{}': {}", file_path, e);
                    process::exit(1);
                }
            };
            if let Some(ceiling) = global_tmag_ceiling {
                params.tmag_ceiling = Some(ceiling);
            }

            params.print_details();

            // Strict backend initialization: exits with code 1 if a GPU backend fails; there is no silent CPU fallback.
            let backend = match create_backend(backend_kind) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Error initializing accelerator backend: {}", e);
                    process::exit(1);
                }
            };

            let generator = match Generator::with_backend(&params.code_data_dir, backend) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Error initializing generator: {}", e);
                    process::exit(1);
                }
            };

            match generator.run_model(&params) {
                Ok(stats) => {
                    println!(
                        "\nSimulation completed successfully: {} stars written to {}.",
                        stats.total_stars, stats.output_path
                    );
                }
                Err(e) => {
                    eprintln!("Error during simulation run: {}", e);
                    process::exit(1);
                }
            }
        }
        Some('s') => {
            eprintln!("Setup option (-s) not implemented yet");
            process::exit(2);
        }
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}
