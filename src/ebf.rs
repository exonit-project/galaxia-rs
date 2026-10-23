// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reader and writer for the Efficient Binary Format (EBF) used by Galaxia.
//!
//! EBF is a structured, cross-platform binary format created by Sanjib Sharma for
//! storing multi-dimensional astronomical datasets.
//!
//! # Format Specifications
//! - **Version 1.1**:
//!   - 8-byte signature: `[0x8a, 0x45, 0x42, 0x46, 0xae, 0x2b, 0xaf, 0x0a]` (`\x8aEBF\xae+\xaf\n`)
//!   - 4-byte version: `[1, 1, 0, 0]`
//!   - 32-byte header info: 8 x int32 (`endiantest`, `headersize`, `namesize`, `datatype`, `datasize`, `rank`, `unitsize`, `sdefsize`)
//!   - 4-byte flags
//!   - 8-byte capacity (int64)
//!   - `rank * 8` bytes dimensions (`dim[rank]`)
//!   - Variable strings: `name`, `unit`, `sdef`
//!   - Padding to `headersize` bytes from header start
//!   - Followed by raw data of size `capacity` bytes.
//! - **Version 1.0**:
//!   - 256-byte fixed header beginning with `EBF`
//!   - Used in legacy Galaxia extinction maps (e.g. `ExMap3d_1024.ebf`).
//!
//! This module supports reading both v1.0 and v1.1 formats, listing keys, inspecting
//! shapes/types/units, endianness conversion, and writing v1.1 files compatible with
//! Python's `ebf.read` and C++ Galaxia.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use thiserror::Error;

pub const EBF_SIG_V11: [u8; 8] = [0x8a, 0x45, 0x42, 0x46, 0xae, 0x2b, 0xaf, 0x0a];
pub const EBF_ENDIAN_MAGIC: i32 = 0x64636261; // 1684234849 ("abcd" little-endian)
pub const EBF_V10_ENDIAN_MAGIC: i32 = 256;

#[derive(Error, Debug)]
pub enum EbfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid EBF signature or unrecognized header at offset {offset}")]
    InvalidSignature { offset: u64 },

    #[error("Unsupported EBF version: {major}.{minor}")]
    UnsupportedVersion { major: u8, minor: u8 },

    #[error("Unknown or invalid data type code: {0}")]
    InvalidDataType(i32),

    #[error("Dataset '{0}' not found in file")]
    DatasetNotFound(String),

    #[error("Type mismatch for dataset '{name}': expected {expected}, found {found:?}")]
    TypeMismatch {
        name: String,
        expected: &'static str,
        found: DataType,
    },

    #[error("Data length mismatch: expected {expected} bytes, got {got} bytes")]
    LengthMismatch { expected: usize, got: usize },

    #[error("EBF error: {0}")]
    Custom(String),
}

pub type Result<T> = std::result::Result<T, EbfError>;

/// Data type code in EBF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Char = 1,
    Int32 = 2,
    Int64 = 3,
    Float32 = 4,
    Float64 = 5,
    Int16 = 6,
    String = 7,
    Struct = 8,
    Int8 = 9,
    UInt8 = 10,
    UInt16 = 11,
    UInt32 = 12,
    UInt64 = 13,
}

impl DataType {
    pub fn from_code(code: i32) -> Result<Self> {
        match code {
            1 => Ok(Self::Char),
            2 => Ok(Self::Int32),
            3 => Ok(Self::Int64),
            4 => Ok(Self::Float32),
            5 => Ok(Self::Float64),
            6 => Ok(Self::Int16),
            7 => Ok(Self::String),
            8 => Ok(Self::Struct),
            9 => Ok(Self::Int8),
            10 => Ok(Self::UInt8),
            11 => Ok(Self::UInt16),
            12 => Ok(Self::UInt32),
            13 => Ok(Self::UInt64),
            _ => Err(EbfError::InvalidDataType(code)),
        }
    }

    pub fn code(self) -> i32 {
        self as i32
    }

    pub fn item_size(self) -> usize {
        match self {
            Self::Char | Self::String | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
            Self::Struct => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Int16 => "int16",
            Self::String => "string",
            Self::Struct => "struct",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
        }
    }
}

/// Container for decoded EBF dataset values.
#[derive(Debug, Clone, PartialEq)]
pub enum EbfData {
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    UInt8(Vec<u8>),
    UInt16(Vec<u16>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
    String(String),
    Raw(Vec<u8>),
}

impl EbfData {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Int8(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::UInt8(v) => v.len(),
            Self::UInt16(v) => v.len(),
            Self::UInt32(v) => v.len(),
            Self::UInt64(v) => v.len(),
            Self::String(s) => s.len(),
            Self::Raw(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert any numeric dataset into `Vec<f64>`.
    pub fn as_f64_vec(&self) -> Result<Vec<f64>> {
        match self {
            Self::Float64(v) => Ok(v.clone()),
            Self::Float32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::Int8(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::Int16(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::Int32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::Int64(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::UInt8(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::UInt16(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::UInt32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            Self::UInt64(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            _ => Err(EbfError::Custom(
                "Cannot convert non-numeric data to f64".to_string(),
            )),
        }
    }

    /// Convert any numeric dataset into `Vec<f32>`.
    pub fn as_f32_vec(&self) -> Result<Vec<f32>> {
        match self {
            Self::Float32(v) => Ok(v.clone()),
            Self::Float64(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::Int8(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::Int16(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::Int32(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::Int64(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::UInt8(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::UInt16(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::UInt32(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            Self::UInt64(v) => Ok(v.iter().map(|&x| x as f32).collect()),
            _ => Err(EbfError::Custom(
                "Cannot convert non-numeric data to f32".to_string(),
            )),
        }
    }

    /// Convert dataset to `Vec<i32>` if integral.
    pub fn as_i32_vec(&self) -> Result<Vec<i32>> {
        match self {
            Self::Int32(v) => Ok(v.clone()),
            Self::Int16(v) => Ok(v.iter().map(|&x| x as i32).collect()),
            Self::Int8(v) => Ok(v.iter().map(|&x| x as i32).collect()),
            Self::UInt8(v) => Ok(v.iter().map(|&x| x as i32).collect()),
            Self::UInt16(v) => Ok(v.iter().map(|&x| x as i32).collect()),
            Self::Int64(v) => Ok(v.iter().map(|&x| x as i32).collect()),
            _ => Err(EbfError::Custom(
                "Cannot convert dataset to i32".to_string(),
            )),
        }
    }

    /// Convert dataset to `Vec<i64>` if integral.
    pub fn as_i64_vec(&self) -> Result<Vec<i64>> {
        match self {
            Self::Int64(v) => Ok(v.clone()),
            Self::Int32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            Self::Int16(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            Self::Int8(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            Self::UInt8(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            Self::UInt16(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            Self::UInt32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            _ => Err(EbfError::Custom(
                "Cannot convert dataset to i64".to_string(),
            )),
        }
    }
}

/// Metadata and layout of an EBF dataset stored on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetHeader {
    pub name: String,
    pub unit: String,
    pub sdef: String,
    pub datatype: DataType,
    pub datasize: usize,
    pub rank: usize,
    pub dims: Vec<usize>,
    pub capacity: usize,
    pub header_pos: u64,
    pub data_pos: u64,
    pub flag_swap: bool,
    pub version: (u8, u8),
}

impl DatasetHeader {
    /// Number of elements = product(dims). If rank is 0, returns 1.
    pub fn elements(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }
}

/// Complete dataset with name, units, dimensions, and typed data.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub name: String,
    pub unit: String,
    pub dims: Vec<usize>,
    pub data: EbfData,
}

impl Dataset {
    pub fn elements(&self) -> usize {
        if self.dims.is_empty() {
            self.data.len()
        } else {
            self.dims.iter().product()
        }
    }
}

/// Reader for EBF binary files.
pub struct EbfFile {
    file: File,
    headers: Vec<DatasetHeader>,
    key_map: HashMap<String, usize>,
}

impl EbfFile {
    /// Open an EBF file and parse the headers of all contained datasets.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;

        let mut headers = Vec::new();
        let mut key_map = HashMap::new();

        let mut cur_pos = 0u64;
        while cur_pos < file_len {
            file.seek(SeekFrom::Start(cur_pos))?;
            let header_pos = cur_pos;

            // Peek at signature
            let mut sig_buf = [0u8; 8];
            let bytes_read = file.read(&mut sig_buf)?;
            if bytes_read == 0 {
                break;
            }
            if bytes_read < 8 {
                return Err(EbfError::InvalidSignature { offset: header_pos });
            }

            if sig_buf == EBF_SIG_V11 {
                // Version 1.1 header
                let mut ver_buf = [0u8; 4];
                file.read_exact(&mut ver_buf)?;
                let major = ver_buf[0];
                let minor = ver_buf[1];

                if major != 1 || minor != 1 {
                    return Err(EbfError::UnsupportedVersion { major, minor });
                }

                let mut u4 = [0i32; 8];
                let mut u4_bytes = [0u8; 32];
                file.read_exact(&mut u4_bytes)?;
                for (i, chunk) in u4_bytes.chunks_exact(4).enumerate() {
                    u4[i] = i32::from_le_bytes(chunk.try_into().unwrap());
                }

                let endiantest = u4[0];
                let mut flag_swap = false;
                if endiantest != EBF_ENDIAN_MAGIC {
                    if endiantest.swap_bytes() == EBF_ENDIAN_MAGIC {
                        flag_swap = true;
                        for val in u4.iter_mut() {
                            *val = val.swap_bytes();
                        }
                    } else {
                        return Err(EbfError::InvalidSignature { offset: header_pos });
                    }
                }

                let headersize = u4[1] as usize;
                let namesize = u4[2] as usize;
                let datatype_code = u4[3];
                let datasize = u4[4] as usize;
                let rank = u4[5] as usize;
                let unitsize = u4[6] as usize;
                let sdefsize = u4[7] as usize;

                let mut flags = [0u8; 4];
                file.read_exact(&mut flags)?;

                let mut cap_bytes = [0u8; 8];
                file.read_exact(&mut cap_bytes)?;
                let mut capacity = i64::from_le_bytes(cap_bytes);
                if flag_swap {
                    capacity = capacity.swap_bytes();
                }
                let capacity = capacity as usize;

                let mut dims = Vec::with_capacity(rank);
                if rank > 0 {
                    let mut dim_buf = vec![0u8; rank * 8];
                    file.read_exact(&mut dim_buf)?;
                    for chunk in dim_buf.chunks_exact(8) {
                        let mut d = i64::from_le_bytes(chunk.try_into().unwrap());
                        if flag_swap {
                            d = d.swap_bytes();
                        }
                        dims.push(d as usize);
                    }
                } else {
                    dims.push(1);
                }

                let mut name_bytes = vec![0u8; namesize];
                file.read_exact(&mut name_bytes)?;
                let name = String::from_utf8_lossy(&name_bytes).trim().to_lowercase();

                let mut unit_bytes = vec![0u8; unitsize];
                file.read_exact(&mut unit_bytes)?;
                let unit = String::from_utf8_lossy(&unit_bytes).to_string();

                let mut sdef_bytes = vec![0u8; sdefsize];
                file.read_exact(&mut sdef_bytes)?;
                let sdef = String::from_utf8_lossy(&sdef_bytes).to_string();

                let data_pos = header_pos + headersize as u64;
                let datatype = DataType::from_code(datatype_code)?;

                let header = DatasetHeader {
                    name: name.clone(),
                    unit,
                    sdef,
                    datatype,
                    datasize,
                    rank,
                    dims,
                    capacity,
                    header_pos,
                    data_pos,
                    flag_swap,
                    version: (major, minor),
                };

                let idx = headers.len();
                key_map.insert(name.clone(), idx);
                let trimmed_name = name.trim_start_matches('/').to_string();
                key_map.insert(trimmed_name, idx);
                headers.push(header);

                cur_pos = data_pos + capacity as u64;
            } else if &sig_buf[0..3] == b"EBF" {
                // Version 1.0 header (256 bytes)
                file.seek(SeekFrom::Start(header_pos))?;
                let mut buf256 = [0u8; 256];
                file.read_exact(&mut buf256)?;

                let name_raw = &buf256[6..106];
                let name_end = name_raw
                    .iter()
                    .position(|&b| b == 0 || b == b' ')
                    .unwrap_or(100);
                let name = String::from_utf8_lossy(&name_raw[..name_end])
                    .trim()
                    .to_lowercase();

                let endiantest_raw = i32::from_le_bytes(buf256[144..148].try_into().unwrap());
                let mut flag_swap = false;
                if endiantest_raw != EBF_V10_ENDIAN_MAGIC {
                    if endiantest_raw.swap_bytes() == EBF_V10_ENDIAN_MAGIC {
                        flag_swap = true;
                    } else {
                        return Err(EbfError::InvalidSignature { offset: header_pos });
                    }
                }

                let read_i32 = |start: usize| -> i32 {
                    let v = i32::from_le_bytes(buf256[start..start + 4].try_into().unwrap());
                    if flag_swap {
                        v.swap_bytes()
                    } else {
                        v
                    }
                };

                let datatype_code = read_i32(148);
                let datasize = read_i32(152) as usize;
                let rank = read_i32(156) as usize;

                let mut dims = Vec::with_capacity(rank);
                for i in 0..rank {
                    let off = 192 + i * 8;
                    let mut d = i64::from_le_bytes(buf256[off..off + 8].try_into().unwrap());
                    if flag_swap {
                        d = d.swap_bytes();
                    }
                    dims.push(d as usize);
                }
                if dims.is_empty() {
                    dims.push(1);
                }

                let elements: usize = dims.iter().product();
                let capacity = elements * datasize;
                let data_pos = header_pos + 256;
                let datatype = DataType::from_code(datatype_code)?;

                let header = DatasetHeader {
                    name: name.clone(),
                    unit: String::new(),
                    sdef: String::new(),
                    datatype,
                    datasize,
                    rank,
                    dims,
                    capacity,
                    header_pos,
                    data_pos,
                    flag_swap,
                    version: (1, 0),
                };

                let idx = headers.len();
                key_map.insert(name.clone(), idx);
                let trimmed_name = name.trim_start_matches('/').to_string();
                key_map.insert(trimmed_name, idx);
                headers.push(header);

                cur_pos = data_pos + capacity as u64;
            } else {
                return Err(EbfError::InvalidSignature { offset: header_pos });
            }
        }

        Ok(Self {
            file,
            headers,
            key_map,
        })
    }

    /// List all dataset keys contained in this file.
    pub fn keys(&self) -> Vec<String> {
        self.headers.iter().map(|h| h.name.clone()).collect()
    }

    /// List user dataset keys (excluding internal `/.ebf/` and `/.tr/` metadata tables).
    pub fn data_keys(&self) -> Vec<String> {
        self.headers
            .iter()
            .map(|h| h.name.clone())
            .filter(|k| !k.starts_with("/.ebf/") && !k.starts_with("/.tr/"))
            .collect()
    }

    /// Check whether a dataset key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.find_header_idx(key).is_some()
    }

    /// Retrieve the metadata header for a dataset key.
    pub fn header(&self, key: &str) -> Option<&DatasetHeader> {
        self.find_header_idx(key).map(|idx| &self.headers[idx])
    }

    fn find_header_idx(&self, key: &str) -> Option<usize> {
        let norm = key.trim().to_lowercase();
        if let Some(&idx) = self.key_map.get(&norm) {
            return Some(idx);
        }
        let with_slash = if norm.starts_with('/') {
            norm.clone()
        } else {
            format!("/{}", norm)
        };
        if let Some(&idx) = self.key_map.get(&with_slash) {
            return Some(idx);
        }
        let without_slash = norm.trim_start_matches('/').to_string();
        if let Some(&idx) = self.key_map.get(&without_slash) {
            return Some(idx);
        }
        // Also check if `.data` suffix was omitted (e.g. `ExMap3d` -> `exmap3d.data`)
        let with_data = format!("{}.data", with_slash);
        if let Some(&idx) = self.key_map.get(&with_data) {
            return Some(idx);
        }
        None
    }

    /// Read a complete dataset (header + typed data) for the given key.
    pub fn read(&mut self, key: &str) -> Result<Dataset> {
        let idx = self
            .find_header_idx(key)
            .ok_or_else(|| EbfError::DatasetNotFound(key.to_string()))?;
        let header = self.headers[idx].clone();

        self.file.seek(SeekFrom::Start(header.data_pos))?;
        let elements = header.elements();
        let expected_bytes = elements * header.datasize;
        let mut raw = vec![0u8; expected_bytes];
        self.file.read_exact(&mut raw)?;

        let data = decode_ebf_data(
            &raw,
            header.datatype,
            header.datasize,
            elements,
            header.flag_swap,
        )?;

        Ok(Dataset {
            name: header.name,
            unit: header.unit,
            dims: header.dims,
            data,
        })
    }

    /// Read dataset directly as `Vec<f32>` (converting if needed).
    pub fn read_f32(&mut self, key: &str) -> Result<Vec<f32>> {
        let ds = self.read(key)?;
        ds.data.as_f32_vec()
    }

    /// Read dataset directly as `Vec<f64>` (converting if needed).
    pub fn read_f64(&mut self, key: &str) -> Result<Vec<f64>> {
        let ds = self.read(key)?;
        ds.data.as_f64_vec()
    }

    /// Read dataset directly as `Vec<i32>`.
    pub fn read_i32(&mut self, key: &str) -> Result<Vec<i32>> {
        let ds = self.read(key)?;
        ds.data.as_i32_vec()
    }

    /// Read dataset directly as `Vec<i64>`.
    pub fn read_i64(&mut self, key: &str) -> Result<Vec<i64>> {
        let ds = self.read(key)?;
        ds.data.as_i64_vec()
    }
}

fn decode_ebf_data(
    raw: &[u8],
    datatype: DataType,
    _datasize: usize,
    elements: usize,
    swap: bool,
) -> Result<EbfData> {
    match datatype {
        DataType::Float32 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(4) {
                let mut val = f32::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    let bits = val.to_bits().swap_bytes();
                    val = f32::from_bits(bits);
                }
                v.push(val);
            }
            Ok(EbfData::Float32(v))
        }
        DataType::Float64 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(8) {
                let mut val = f64::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    let bits = val.to_bits().swap_bytes();
                    val = f64::from_bits(bits);
                }
                v.push(val);
            }
            Ok(EbfData::Float64(v))
        }
        DataType::Int32 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(4) {
                let mut val = i32::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    val = val.swap_bytes();
                }
                v.push(val);
            }
            Ok(EbfData::Int32(v))
        }
        DataType::Int64 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(8) {
                let mut val = i64::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    val = val.swap_bytes();
                }
                v.push(val);
            }
            Ok(EbfData::Int64(v))
        }
        DataType::Int16 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(2) {
                let mut val = i16::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    val = val.swap_bytes();
                }
                v.push(val);
            }
            Ok(EbfData::Int16(v))
        }
        DataType::Int8 => {
            let v = raw.iter().map(|&b| b as i8).collect();
            Ok(EbfData::Int8(v))
        }
        DataType::UInt8 | DataType::Char => Ok(EbfData::UInt8(raw.to_vec())),
        DataType::UInt16 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(2) {
                let mut val = u16::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    val = val.swap_bytes();
                }
                v.push(val);
            }
            Ok(EbfData::UInt16(v))
        }
        DataType::UInt32 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(4) {
                let mut val = u32::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    val = val.swap_bytes();
                }
                v.push(val);
            }
            Ok(EbfData::UInt32(v))
        }
        DataType::UInt64 => {
            let mut v = Vec::with_capacity(elements);
            for chunk in raw.chunks_exact(8) {
                let mut val = u64::from_le_bytes(chunk.try_into().unwrap());
                if swap {
                    val = val.swap_bytes();
                }
                v.push(val);
            }
            Ok(EbfData::UInt64(v))
        }
        DataType::String => {
            let s = String::from_utf8_lossy(raw)
                .trim_end_matches('\0')
                .to_string();
            Ok(EbfData::String(s))
        }
        _ => Ok(EbfData::Raw(raw.to_vec())),
    }
}

/// Writer for EBF binary files.
pub struct EbfWriter {
    file: File,
}

impl EbfWriter {
    /// Create a new EBF file (truncating existing).
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self { file })
    }

    /// Open an EBF file for appending datasets (or create if not existing).
    pub fn open_append(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { file })
    }

    /// Append a typed dataset to the file.
    pub fn write_dataset(
        &mut self,
        name: &str,
        unit: &str,
        dims: &[usize],
        data: &EbfData,
    ) -> Result<()> {
        match data {
            EbfData::Float32(slice) => self.write_f32(name, unit, dims, slice),
            EbfData::Float64(slice) => self.write_f64(name, unit, dims, slice),
            EbfData::Int32(slice) => self.write_i32(name, unit, dims, slice),
            EbfData::Int64(slice) => self.write_i64(name, unit, dims, slice),
            EbfData::Int16(slice) => {
                let mut raw = Vec::with_capacity(slice.len() * 2);
                for &x in slice {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
                self.write_raw(name, unit, dims, DataType::Int16, 2, &raw)
            }
            EbfData::Int8(slice) => {
                let raw: Vec<u8> = slice.iter().map(|&x| x as u8).collect();
                self.write_raw(name, unit, dims, DataType::Int8, 1, &raw)
            }
            EbfData::UInt8(slice) => self.write_raw(name, unit, dims, DataType::UInt8, 1, slice),
            EbfData::UInt16(slice) => {
                let mut raw = Vec::with_capacity(slice.len() * 2);
                for &x in slice {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
                self.write_raw(name, unit, dims, DataType::UInt16, 2, &raw)
            }
            EbfData::UInt32(slice) => {
                let mut raw = Vec::with_capacity(slice.len() * 4);
                for &x in slice {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
                self.write_raw(name, unit, dims, DataType::UInt32, 4, &raw)
            }
            EbfData::UInt64(slice) => {
                let mut raw = Vec::with_capacity(slice.len() * 8);
                for &x in slice {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
                self.write_raw(name, unit, dims, DataType::UInt64, 8, &raw)
            }
            EbfData::String(s) => {
                let raw = s.as_bytes();
                self.write_raw(name, unit, &[raw.len()], DataType::String, 1, raw)
            }
            EbfData::Raw(raw) => self.write_raw(name, unit, dims, DataType::UInt8, 1, raw),
        }
    }

    /// Append a `float32` dataset.
    pub fn write_f32(
        &mut self,
        name: &str,
        unit: &str,
        dims: &[usize],
        values: &[f32],
    ) -> Result<()> {
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.write_raw(name, unit, dims, DataType::Float32, 4, &raw)
    }

    /// Append a `float64` dataset.
    pub fn write_f64(
        &mut self,
        name: &str,
        unit: &str,
        dims: &[usize],
        values: &[f64],
    ) -> Result<()> {
        let mut raw = Vec::with_capacity(values.len() * 8);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.write_raw(name, unit, dims, DataType::Float64, 8, &raw)
    }

    /// Append an `int32` dataset.
    pub fn write_i32(
        &mut self,
        name: &str,
        unit: &str,
        dims: &[usize],
        values: &[i32],
    ) -> Result<()> {
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.write_raw(name, unit, dims, DataType::Int32, 4, &raw)
    }

    /// Append an `int64` dataset.
    pub fn write_i64(
        &mut self,
        name: &str,
        unit: &str,
        dims: &[usize],
        values: &[i64],
    ) -> Result<()> {
        let mut raw = Vec::with_capacity(values.len() * 8);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.write_raw(name, unit, dims, DataType::Int64, 8, &raw)
    }

    /// Append a `char` (byte/string) dataset (e.g. `"/log"`).
    pub fn write_char(
        &mut self,
        name: &str,
        unit: &str,
        dims: &[usize],
        values: &[u8],
    ) -> Result<()> {
        self.write_raw(name, unit, dims, DataType::Char, 1, values)
    }

    fn write_raw(
        &mut self,
        name_in: &str,
        unit_in: &str,
        dims_in: &[usize],
        datatype: DataType,
        datasize: usize,
        raw_bytes: &[u8],
    ) -> Result<()> {
        let name_clean = if name_in.starts_with('/') {
            name_in.to_lowercase()
        } else {
            format!("/{}", name_in.to_lowercase())
        };
        let name_bytes = name_clean.as_bytes();
        let unit_bytes = unit_in.as_bytes();
        let sdef_bytes = b"";

        let rank = if dims_in.is_empty() { 1 } else { dims_in.len() };
        let dims: Vec<i64> = if dims_in.is_empty() {
            vec![1]
        } else {
            dims_in.iter().map(|&d| d as i64).collect()
        };

        let extrasize: usize = if rank == 1 && dims[0] == 1 { 16 } else { 64 };
        let namesize = name_bytes.len();
        let unitsize = unit_bytes.len();
        let sdefsize = sdef_bytes.len();
        let headersize = 56 + namesize + unitsize + sdefsize + extrasize + 8 * rank;
        let capacity = raw_bytes.len() as i64;

        let mut header_buf = Vec::with_capacity(headersize);
        header_buf.extend_from_slice(&EBF_SIG_V11);
        header_buf.extend_from_slice(&[1, 1, 0, 0]); // version 1.1

        let u4: [i32; 8] = [
            EBF_ENDIAN_MAGIC,
            headersize as i32,
            namesize as i32,
            datatype.code(),
            datasize as i32,
            rank as i32,
            unitsize as i32,
            sdefsize as i32,
        ];
        for val in u4 {
            header_buf.extend_from_slice(&val.to_le_bytes());
        }

        header_buf.extend_from_slice(&[0, 0, 0, 0]); // flags
        header_buf.extend_from_slice(&capacity.to_le_bytes());
        for &d in &dims {
            header_buf.extend_from_slice(&d.to_le_bytes());
        }

        header_buf.extend_from_slice(name_bytes);
        header_buf.extend_from_slice(unit_bytes);
        header_buf.extend_from_slice(sdef_bytes);

        // Padding with '<' (ASCII 60) up to headersize bytes
        let written = header_buf.len();
        if written < headersize {
            header_buf.resize(headersize, 60);
        }

        self.file.write_all(&header_buf)?;
        self.file.write_all(raw_bytes)?;
        self.file.flush()?;
        Ok(())
    }
}

// Convenient top-level functions

/// Read a complete dataset from an EBF file.
pub fn read(path: impl AsRef<Path>, key: &str) -> Result<Dataset> {
    let mut f = EbfFile::open(path)?;
    f.read(key)
}

/// Read a dataset as `Vec<f32>`.
pub fn read_f32(path: impl AsRef<Path>, key: &str) -> Result<Vec<f32>> {
    let mut f = EbfFile::open(path)?;
    f.read_f32(key)
}

/// Read a dataset as `Vec<f64>`.
pub fn read_f64(path: impl AsRef<Path>, key: &str) -> Result<Vec<f64>> {
    let mut f = EbfFile::open(path)?;
    f.read_f64(key)
}

/// Read a dataset as `Vec<i32>`.
pub fn read_i32(path: impl AsRef<Path>, key: &str) -> Result<Vec<i32>> {
    let mut f = EbfFile::open(path)?;
    f.read_i32(key)
}

/// Read a dataset as `Vec<i64>`.
pub fn read_i64(path: impl AsRef<Path>, key: &str) -> Result<Vec<i64>> {
    let mut f = EbfFile::open(path)?;
    f.read_i64(key)
}

/// List all dataset keys in an EBF file.
pub fn keys(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let f = EbfFile::open(path)?;
    Ok(f.keys())
}

/// Write or append a `float32` dataset to an EBF file (`mode` can be "w" or "a").
pub fn write_f32(
    path: impl AsRef<Path>,
    key: &str,
    unit: &str,
    dims: &[usize],
    values: &[f32],
    mode: &str,
) -> Result<()> {
    let mut writer = if mode == "a" {
        EbfWriter::open_append(path)?
    } else {
        EbfWriter::create(path)?
    };
    writer.write_f32(key, unit, dims, values)
}

/// Write or append a `float64` dataset to an EBF file (`mode` can be "w" or "a").
pub fn write_f64(
    path: impl AsRef<Path>,
    key: &str,
    unit: &str,
    dims: &[usize],
    values: &[f64],
    mode: &str,
) -> Result<()> {
    let mut writer = if mode == "a" {
        EbfWriter::open_append(path)?
    } else {
        EbfWriter::create(path)?
    };
    writer.write_f64(key, unit, dims, values)
}

/// Write or append an `int32` dataset to an EBF file (`mode` can be "w" or "a").
pub fn write_i32(
    path: impl AsRef<Path>,
    key: &str,
    unit: &str,
    dims: &[usize],
    values: &[i32],
    mode: &str,
) -> Result<()> {
    let mut writer = if mode == "a" {
        EbfWriter::open_append(path)?
    } else {
        EbfWriter::create(path)?
    };
    writer.write_i32(key, unit, dims, values)
}

/// Write or append an `int64` dataset to an EBF file (`mode` can be "w" or "a").
pub fn write_i64(
    path: impl AsRef<Path>,
    key: &str,
    unit: &str,
    dims: &[usize],
    values: &[i64],
    mode: &str,
) -> Result<()> {
    let mut writer = if mode == "a" {
        EbfWriter::open_append(path)?
    } else {
        EbfWriter::create(path)?
    };
    writer.write_i64(key, unit, dims, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("galaxia_rs_test_ebf.ebf");

        {
            let mut writer = EbfWriter::create(&test_file).unwrap();
            writer
                .write_f32("/test/f32_data", "km/s", &[3], &[1.5, 2.5, 3.5])
                .unwrap();
            writer
                .write_f64(
                    "/test/f64_matrix",
                    "Msun",
                    &[2, 2],
                    &[10.0, 20.0, 30.0, 40.0],
                )
                .unwrap();
            writer
                .write_i32("/test/popid", "", &[4], &[0, 1, 2, 3])
                .unwrap();
        }

        let mut reader = EbfFile::open(&test_file).unwrap();
        assert!(reader.contains_key("/test/f32_data"));
        assert!(reader.contains_key("/test/f64_matrix"));
        assert!(reader.contains_key("/test/popid"));

        let f32_vals = reader.read_f32("/test/f32_data").unwrap();
        assert_eq!(f32_vals, vec![1.5, 2.5, 3.5]);

        let f64_vals = reader.read_f64("/test/f64_matrix").unwrap();
        assert_eq!(f64_vals, vec![10.0, 20.0, 30.0, 40.0]);

        let i32_vals = reader.read_i32("/test/popid").unwrap();
        assert_eq!(i32_vals, vec![0, 1, 2, 3]);

        // Cleanup
        let _ = std::fs::remove_file(test_file);
    }
}
