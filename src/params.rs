// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parser for Galaxia parameter files (`.par`).
//!
//! Replicates the parsing logic and default configuration from Galaxia's `Parameters.cpp`
//! and `Parameters.h`.

use std::fs;
use std::path::Path;
use thiserror::Error;

/// Default location of the original `GalaxiaData` directory (relative to the working directory).
/// Override with the `GALAXIA_DATA_DIR` environment variable (legacy alias `GALAXIA_DATA`) or the `codeDataDir` parameter-file tag.
pub const DEFAULT_CODE_DATA_DIR: &str = "GalaxiaData/";

#[derive(Error, Debug)]
pub enum ParamsError {
    #[error("I/O error reading parameter file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse number in parameter '{tag}' with value '{value}': {source}")]
    ParseNumber {
        tag: String,
        value: String,
        #[source]
        source: std::num::ParseFloatError,
    },

    #[error("Failed to parse integer in parameter '{tag}' with value '{value}': {source}")]
    ParseInt {
        tag: String,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("Missing required parameter file or argument")]
    MissingArgument,

    #[error("Unknown parameter tag: '{0}'")]
    UnknownTag(String),
}

pub type Result<T> = std::result::Result<T, ParamsError>;

/// Alias for Params matching common naming across modules.
pub type GalaxiaParams = Params;

/// Configuration parameters for a Galaxia simulation run.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    // General options
    pub output_file: String,
    pub output_dir: String,
    pub code_data_dir: String,
    pub parameter_file: String,
    pub nbody_file: String,
    pub add_string: String,
    pub debug: i32,

    // Photometric options
    pub photo_sys: String,
    pub magcolor_names: String,
    pub app_mag_limits: [f64; 2],
    pub abs_mag_limits: [f64; 2],
    pub color_limits: [f64; 2],

    // Survey / geometry options
    pub geometry_option: i32,
    pub longitude: f64,
    pub latitude: f64,
    pub survey_area: f64,
    pub f_sample: f64,
    pub pop_id: i32,
    pub warp_flare_on: i32,
    pub seed: i32,
    pub r_max: f64,
    pub star_type: i32,
    pub photo_error: i32,

    // Solar kinematics and observer position
    // pos_c = [rSun[0], rSun[1], rSun[2], vSun[0], vSun[1], vSun[2]]
    pub pos_c: [f64; 6],
    pub sigma_r: f64,
    pub sigma_vr: f64,
    pub sigma_mu: f64,
    pub sigma_fe: f64,
    pub sigma_al: f64,
    pub hdim: i32,
    pub nres: i32,

    // Execution option: 0 = setup (BHTree), 1 = run, 2 = append mag, 3 = extinction
    pub option: i32,

    /// Optional extincted Tmag ceiling cut (recorded in output metadata)
    pub tmag_ceiling: Option<f64>,
}

impl Default for Params {
    fn default() -> Self {
        let code_data = std::env::var("GALAXIA_DATA_DIR")
            .or_else(|_| std::env::var("GALAXIA_DATA"))
            .unwrap_or_else(|_| DEFAULT_CODE_DATA_DIR.to_string());
        let code_data_dir = if code_data.ends_with('/') || code_data.ends_with('\\') {
            code_data
        } else {
            format!("{}{}", code_data, std::path::MAIN_SEPARATOR)
        };

        Self {
            output_file: "Examples/galaxy".to_string(),
            output_dir: "Examples/".to_string(),
            code_data_dir,
            parameter_file: String::new(),
            nbody_file: String::new(),
            add_string: String::new(),
            debug: 0,

            photo_sys: "UBV".to_string(),
            magcolor_names: "V,B-V".to_string(),
            app_mag_limits: [-100.0, 23.0],
            abs_mag_limits: [-100.0, 100.0],
            color_limits: [-10.0, 10.0],

            geometry_option: 0,
            longitude: 0.0,
            latitude: 90.0,
            survey_area: 0.0,
            f_sample: 1.0,
            pop_id: -1,
            warp_flare_on: 1,
            seed: 17,
            r_max: 1e10,
            star_type: 0,
            photo_error: 0,

            // Exact constants from Parameters.cpp:
            // rSun = [-8.0, 0.0, 0.015], vSun = [11.1, 239.08, 7.25]
            pos_c: [-8.0, 0.0, 0.015, 11.1, 239.08, 7.25],
            sigma_r: 0.1,
            sigma_vr: 10.0,
            sigma_mu: 200.0,
            sigma_fe: 0.1,
            sigma_al: 0.1,
            hdim: 6,
            nres: 64,

            option: 1,
            tmag_ceiling: None,
        }
    }
}

impl Params {
    /// Create new parameters initialized with default Galaxia values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load and parse parameters from a `.par` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref)?;
        let mut params = Self::from_str(&content)?;
        params.parameter_file = path_ref.to_string_lossy().to_string();
        Ok(params)
    }

    /// Parse parameters from a string content.
    pub fn from_str(content: &str) -> Result<Self> {
        let mut params = Self::default();
        params.parse_content(content)?;
        params.check_compilation();
        Ok(params)
    }

    fn parse_content(&mut self, content: &str) -> Result<()> {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('%') {
                continue;
            }

            let mut tokens = trimmed.split_whitespace();
            let tag = match tokens.next() {
                Some(t) => t,
                None => continue,
            };
            let val = match tokens.next() {
                Some(v) => v,
                None => continue,
            };

            match tag {
                "outputFile" => self.output_file = val.to_string(),
                "outputDir" => self.output_dir = val.to_string(),
                "codeDataDir" => self.code_data_dir = val.to_string(),
                "photoSys" => self.photo_sys = val.to_string(),
                "magcolorNames" => self.magcolor_names = val.to_string(),

                "appMagLimits[0]" => {
                    self.app_mag_limits[0] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "appMagLimits[1]" => {
                    self.app_mag_limits[1] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "absMagLimits[0]" => {
                    self.abs_mag_limits[0] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "absMagLimits[1]" => {
                    self.abs_mag_limits[1] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "colorLimits[0]" => {
                    self.color_limits[0] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "colorLimits[1]" => {
                    self.color_limits[1] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }

                "geometryOption" => {
                    self.geometry_option = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "longitude" => {
                    self.longitude = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "latitude" => {
                    self.latitude = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "surveyArea" => {
                    self.survey_area = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "fSample" => {
                    self.f_sample = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "popID" => {
                    self.pop_id = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "warpFlareOn" => {
                    self.warp_flare_on = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "seed" => {
                    self.seed = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "r_max" => {
                    self.r_max = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "starType" => {
                    self.star_type = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "photoError" => {
                    self.photo_error = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }

                // Observer position & velocities
                "rSun[0]" => {
                    self.pos_c[0] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "rSun[1]" => {
                    self.pos_c[1] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "rSun[2]" => {
                    self.pos_c[2] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "vSun[0]" => {
                    self.pos_c[3] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "vSun[1]" => {
                    self.pos_c[4] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "vSun[2]" => {
                    self.pos_c[5] = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }

                "sigma_r" => {
                    self.sigma_r = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "sigma_vr" => {
                    self.sigma_vr = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "sigma_mu" => {
                    self.sigma_mu = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "sigma_fe" => {
                    self.sigma_fe = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "sigma_al" => {
                    self.sigma_al = val.parse().map_err(|e| ParamsError::ParseNumber {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "hdim" => {
                    self.hdim = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                "nres" => {
                    self.nres = val.parse().map_err(|e| ParamsError::ParseInt {
                        tag: tag.to_string(),
                        value: val.to_string(),
                        source: e,
                    })?;
                }
                _ => {
                    // Ignore unrecognized tags or comments
                }
            }
        }
        Ok(())
    }

    /// Derived quantities and normalization matching C++ `checkCompilation()`.
    pub fn check_compilation(&mut self) {
        if self.output_file.ends_with(".ebf") {
            let len = self.output_file.len();
            self.output_file.truncate(len - 4);
        }
        if !self.output_dir.is_empty()
            && !self.output_dir.ends_with('/')
            && !self.output_dir.ends_with('\\')
        {
            self.output_dir.push(std::path::MAIN_SEPARATOR);
        }
        if !self.code_data_dir.is_empty()
            && !self.code_data_dir.ends_with('/')
            && !self.code_data_dir.ends_with('\\')
        {
            self.code_data_dir.push(std::path::MAIN_SEPARATOR);
        }
    }

    /// Full path to the output EBF file.
    pub fn output_ebf_path(&self) -> String {
        let dir = Path::new(&self.output_dir);
        let mut file_name = self.output_file.clone();
        if !file_name.ends_with(".ebf") {
            file_name.push_str(".ebf");
        }
        dir.join(file_name).to_string_lossy().into_owned()
    }

    /// Print summary of parameter details identical to C++ Galaxia.
    pub fn print_details(&self) {
        println!("----------------------------------");
        println!("Parameter Details");
        println!("Input Halo Sat File ={}", self.nbody_file);
        println!("Photometric System  ={}", self.photo_sys);
        println!("Geometry Option     ={}", self.geometry_option);
        println!("Survey Area         ={:.6}", self.survey_area);
        println!(
            "Absolute Mag Limits ={:.6} {:.6}",
            self.abs_mag_limits[0], self.abs_mag_limits[1]
        );
        println!(
            "Apparent Mag Limits ={:.6} {:.6}",
            self.app_mag_limits[0], self.app_mag_limits[1]
        );
        println!(
            "Color Limits        ={:.6} {:.6}",
            self.color_limits[0], self.color_limits[1]
        );
        println!("fSample             ={:.6}", self.f_sample);
        println!("----------------------------------");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let p = Params::default();
        assert_eq!(p.output_file, "Examples/galaxy");
        assert_eq!(p.output_dir, "Examples/");
        assert_eq!(p.photo_sys, "UBV");
        assert_eq!(p.magcolor_names, "V,B-V");
        assert_eq!(p.app_mag_limits, [-100.0, 23.0]);
        assert_eq!(p.abs_mag_limits, [-100.0, 100.0]);
        assert_eq!(p.color_limits, [-10.0, 10.0]);
        assert_eq!(p.geometry_option, 0);
        assert_eq!(p.f_sample, 1.0);
        assert_eq!(p.pop_id, -1);
        assert_eq!(p.warp_flare_on, 1);
        assert_eq!(p.seed, 17);
        assert_eq!(p.pos_c, [-8.0, 0.0, 0.015, 11.1, 239.08, 7.25]);
    }

    #[test]
    fn test_parse_t_par() {
        let sample = r#"
outputFile                          gx_test
outputDir                           /tmp/gxt/
photoSys                            DCMC
magcolorNames                       J,J-Ks
appMagLimits[0]                     -100
appMagLimits[1]                     22
absMagLimits[0]                     -1000
absMagLimits[1]                     1000
colorLimits[0]                      -1000
colorLimits[1]                      1000
geometryOption                      1
longitude                           60.000000
latitude                            10.000000
surveyArea                          0.1
fSample                             1.0
popID                               -1
warpFlareOn                         1
seed                                1234
r_max                               1000
starType                            0
photoError                          0
"#;
        let p = Params::from_str(sample).unwrap();
        assert_eq!(p.output_file, "gx_test");
        assert_eq!(p.output_dir, "/tmp/gxt/");
        assert_eq!(p.photo_sys, "DCMC");
        assert_eq!(p.magcolor_names, "J,J-Ks");
        assert_eq!(p.app_mag_limits, [-100.0, 22.0]);
        assert_eq!(p.abs_mag_limits, [-1000.0, 1000.0]);
        assert_eq!(p.color_limits, [-1000.0, 1000.0]);
        assert_eq!(p.geometry_option, 1);
        assert_eq!(p.longitude, 60.0);
        assert_eq!(p.latitude, 10.0);
        assert_eq!(p.survey_area, 0.1);
        assert_eq!(p.f_sample, 1.0);
        assert_eq!(p.pop_id, -1);
        assert_eq!(p.warp_flare_on, 1);
        assert_eq!(p.seed, 1234);
        assert_eq!(p.r_max, 1000.0);
        assert_eq!(p.star_type, 0);
        assert_eq!(p.photo_error, 0);
        assert_eq!(p.output_ebf_path(), "/tmp/gxt/gx_test.ebf");
    }
}
