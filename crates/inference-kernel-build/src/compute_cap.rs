//! Compute capability detection and management

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::process::Command;

/// GPU architecture specification
///
/// Supports both numeric (80, 90) and string-based (90a, 100a) formats.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuArch {
    /// Base compute capability number (e.g., 90, 100, 120)
    pub base: usize,
    /// Optional suffix for accelerated variants (e.g., "a" for async)
    pub suffix: Option<String>,
}

impl GpuArch {
    /// Create a new GPU architecture from base number
    pub fn new(base: usize) -> Self {
        Self { base, suffix: None }
    }

    /// Create a new GPU architecture with suffix (e.g., 90a, 100a)
    pub fn with_suffix(base: usize, suffix: &str) -> Self {
        Self {
            base,
            suffix: Some(suffix.to_string()),
        }
    }

    /// Parse from string like "90", "90a", "100a", "sm_90a"
    ///
    /// If no suffix is provided (e.g., "90"), auto-suffix is applied for sm_90+.
    /// To explicitly disable the suffix, use the numeric API directly.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_lowercase();

        // Strip "sm_" prefix if present
        let s = s.strip_prefix("sm_").unwrap_or(&s);

        // Check for suffix (letters at the end) - support 'a' and 'f' suffixes
        let (num_part, explicit_suffix) = if let Some(num) = s.strip_suffix('f') {
            (num, Some("f".to_string()))
        } else if let Some(num) = s.strip_suffix('a') {
            (num, Some("a".to_string()))
        } else {
            (s, None)
        };

        // "8.6" and "12.1" as nvidia-smi prints them, or "86" and "121"
        let base = match num_part.split_once('.') {
            Some((major, minor)) if !major.is_empty() && minor.len() == 1 => {
                format!("{major}{minor}").parse::<usize>()
            }
            Some(_) => "".parse::<usize>(),
            None => num_part
                .parse::<usize>()
                .map(|base| if base < 20 { base * 10 } else { base }),
        }
        .map_err(|_| {
            Error::ComputeCapDetectionFailed(format!("Invalid compute capability: {}", s))
        })?;

        // If explicit suffix provided, use it; otherwise auto-suffix for >=90
        if explicit_suffix.is_some() {
            Ok(Self {
                base,
                suffix: explicit_suffix,
            })
        } else {
            Ok(Self::auto_suffix(base))
        }
    }

    /// Create GPU arch with auto-detected suffix for newer architectures
    ///
    /// Suffix selection follows PTX ISA rules:
    /// - SM 120: 'a' suffix - enables arch-specific NVFP4/MXFP4 MMA instructions
    /// - SM 121 (GB10/Spark): 'f' suffix - family-level features only
    /// - SM 90-100/103: 'a' suffix - async/accelerated features
    /// - SM < 90: no suffix
    pub fn auto_suffix(base: usize) -> Self {
        match base {
            120 => Self::with_suffix(base, "a"),
            b if b > 120 => Self::with_suffix(b, "f"),
            b if b >= 90 => Self::with_suffix(b, "a"),
            b => Self::new(b),
        }
    }

    /// Get the nvcc --gpu-architecture string (e.g., "sm_90a", "sm_80")
    pub fn to_nvcc_arch(&self) -> String {
        match &self.suffix {
            Some(s) => format!("sm_{}{}", self.base, s),
            None => format!("sm_{}", self.base),
        }
    }

    /// Get the nvcc -gencode argument (e.g., "-gencode=arch=compute_90a,code=sm_90a")
    ///
    /// This format is preferred for fat binary support and explicit architecture targeting.
    pub fn to_gencode_arg(&self) -> String {
        let compute = match &self.suffix {
            Some(s) => format!("compute_{}{}", self.base, s),
            None => format!("compute_{}", self.base),
        };
        let sm = self.to_nvcc_arch();
        format!("-gencode=arch={},code={}", compute, sm)
    }

    /// Get the base compute capability number
    pub fn base(&self) -> usize {
        self.base
    }
}

impl std::fmt::Display for GpuArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.suffix {
            Some(s) => write!(f, "{}{}", self.base, s),
            None => write!(f, "{}", self.base),
        }
    }
}

impl From<usize> for GpuArch {
    fn from(base: usize) -> Self {
        Self::auto_suffix(base)
    }
}

/// The `-gencode` arguments for every arch in `archs`: one fat binary carrying SASS for each.
pub fn gencode_args(archs: &[GpuArch]) -> Vec<String> {
    archs.iter().map(GpuArch::to_gencode_arg).collect()
}

/// The archs as one key (`sm_80,sm_90a`); a single arch keys as `to_nvcc_arch` alone.
pub fn arch_key(archs: &[GpuArch]) -> String {
    archs
        .iter()
        .map(GpuArch::to_nvcc_arch)
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse a list like "80,86,90" (commas, semicolons or spaces), deduplicated and sorted by base.
pub fn parse_arch_list(s: &str) -> Result<Vec<GpuArch>> {
    let mut archs = s
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(GpuArch::parse)
        .collect::<Result<Vec<_>>>()?;
    archs.sort_by(|a, b| (a.base, &a.suffix).cmp(&(b.base, &b.suffix)));
    archs.dedup();
    if archs.is_empty() {
        return Err(Error::ComputeCapDetectionFailed(format!(
            "No compute capability in {s:?}"
        )));
    }
    Ok(archs)
}

/// Compute capability configuration
#[derive(Debug, Clone, Default)]
pub struct ComputeCapability {
    /// Default compute cap (auto-detected or manually set)
    default_cap: Option<GpuArch>,
    /// Per-file overrides (filename pattern -> compute cap)
    overrides: HashMap<String, GpuArch>,
}

impl ComputeCapability {
    /// Create new compute capability config with auto-detection
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default compute capability (numeric, auto-selects suffix)
    pub fn with_default(mut self, cap: usize) -> Self {
        self.default_cap = Some(GpuArch::auto_suffix(cap));
        self
    }

    /// Set default compute capability with explicit arch string (e.g., "90a", "100a")
    pub fn with_default_arch(mut self, arch: &str) -> Self {
        if let Ok(gpu_arch) = GpuArch::parse(arch) {
            self.default_cap = Some(gpu_arch);
        }
        self
    }

    /// Add compute cap override for files matching pattern (numeric)
    ///
    /// Pattern can be:
    /// - Exact filename: "my_kernel.cu"
    /// - Glob pattern: "sm90_*.cu", "*_hopper.cu"
    pub fn with_override(mut self, pattern: &str, cap: usize) -> Self {
        self.overrides
            .insert(pattern.to_string(), GpuArch::auto_suffix(cap));
        self
    }

    /// Add compute cap override with explicit arch string (e.g., "90a", "100a")
    pub fn with_override_arch(mut self, pattern: &str, arch: &str) -> Self {
        if let Ok(gpu_arch) = GpuArch::parse(arch) {
            self.overrides.insert(pattern.to_string(), gpu_arch);
        }
        self
    }

    /// Get the GPU archs a specific file compiles for
    ///
    /// Priority:
    /// 1. Per-file override matching pattern (that arch alone)
    /// 2. Default compute cap (that arch alone)
    /// 3. Detected: CUDA_COMPUTE_CAP (one value or a list), else the first GPU nvidia-smi lists
    pub fn get_for_file(&self, filename: &str) -> Result<Vec<GpuArch>> {
        for (pattern, arch) in &self.overrides {
            if matches_pattern(filename, pattern) {
                return Ok(vec![arch.clone()]);
            }
        }
        self.get_defaults()
    }

    /// Get every default GPU architecture, lowest first
    pub fn get_defaults(&self) -> Result<Vec<GpuArch>> {
        if let Some(arch) = &self.default_cap {
            return Ok(vec![arch.clone()]);
        }
        detect_compute_caps()
    }

    /// Get the lowest default GPU architecture, the one every compile-time minimum must hold for
    pub fn get_default(&self) -> Result<GpuArch> {
        Ok(self.get_defaults()?.remove(0))
    }

    /// Check if any overrides are configured
    pub fn has_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }
}

/// Detect the lowest compute capability to build for (see [`detect_compute_caps`])
pub fn detect_compute_cap() -> Result<GpuArch> {
    Ok(detect_compute_caps()?.remove(0))
}

/// Detect every compute capability to build for, lowest first
///
/// Priority:
/// 1. CUDA_COMPUTE_CAP environment variable: one value ("90", "90a", "100a") or a list ("80,86,90")
/// 2. nvidia-smi query, its first GPU (a list is asked for, never inferred from a mixed host)
pub fn detect_compute_caps() -> Result<Vec<GpuArch>> {
    if let Ok(cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        return parse_arch_list(&cap_str);
    }
    detect_from_nvidia_smi()
}

/// Detect compute capability using nvidia-smi
fn detect_from_nvidia_smi() -> Result<Vec<GpuArch>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv"])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_nvidia_smi_output(&stdout)
        }
        Ok(output) => Err(Error::ComputeCapDetectionFailed(format!(
            "nvidia-smi failed: {}. \
            If building in Docker, set CUDA_COMPUTE_CAP environment variable (e.g., CUDA_COMPUTE_CAP=90).",
            String::from_utf8_lossy(&output.stderr)
        ))),
        Err(e) => Err(Error::ComputeCapDetectionFailed(format!(
            "Failed to run nvidia-smi: {}. \
            If building in Docker, set CUDA_COMPUTE_CAP environment variable (e.g., CUDA_COMPUTE_CAP=90). \
            GPU is not accessible during 'docker build' - only during 'docker run --gpus all'.",
            e
        ))),
    }
}

/// Parse nvidia-smi output for compute capability: the first GPU after the header line
fn parse_nvidia_smi_output(output: &str) -> Result<Vec<GpuArch>> {
    let line = output
        .lines()
        .skip(1)
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| {
            Error::ComputeCapDetectionFailed("Unexpected nvidia-smi output".to_string())
        })?;
    Ok(vec![GpuArch::parse(line)?])
}

/// Match filename against pattern (simple glob matching)
fn matches_pattern(filename: &str, pattern: &str) -> bool {
    // Handle exact match
    if filename == pattern {
        return true;
    }

    // Simple glob matching for * wildcard
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();

        if parts.len() == 2 {
            let (prefix, suffix) = (parts[0], parts[1]);
            return filename.starts_with(prefix) && filename.ends_with(suffix);
        }

        // Handle single * at start or end
        if let Some(stripped) = pattern.strip_prefix('*') {
            return filename.ends_with(stripped);
        }
        if let Some(stripped) = pattern.strip_suffix('*') {
            return filename.starts_with(stripped);
        }
    }

    false
}

/// Get GPU architecture string for nvcc (e.g., "sm_90a" or "sm_80")
///
/// This is a convenience function. For more control, use GpuArch directly.
pub fn get_gpu_arch_string(compute_cap: usize) -> String {
    GpuArch::auto_suffix(compute_cap).to_nvcc_arch()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_pattern() {
        assert!(matches_pattern("kernel.cu", "kernel.cu"));
        assert!(matches_pattern("sm90_kernel.cu", "sm90_*.cu"));
        assert!(matches_pattern("kernel_hopper.cu", "*_hopper.cu"));
        assert!(matches_pattern("prefix_middle_suffix.cu", "prefix_*.cu"));
        assert!(!matches_pattern("other.cu", "sm90_*.cu"));
    }

    #[test]
    fn test_gpu_arch_string() {
        assert_eq!(get_gpu_arch_string(80), "sm_80");
        assert_eq!(get_gpu_arch_string(90), "sm_90a");
        assert_eq!(get_gpu_arch_string(100), "sm_100a");
        assert_eq!(get_gpu_arch_string(120), "sm_120a");
        assert_eq!(get_gpu_arch_string(121), "sm_121f");
    }

    #[test]
    fn test_gpu_arch_parse() {
        let arch = GpuArch::parse("90a").unwrap();
        assert_eq!(arch.base, 90);
        assert_eq!(arch.suffix, Some("a".to_string()));
        assert_eq!(arch.to_nvcc_arch(), "sm_90a");

        let arch = GpuArch::parse("100a").unwrap();
        assert_eq!(arch.base, 100);
        assert_eq!(arch.to_nvcc_arch(), "sm_100a");

        let arch = GpuArch::parse("sm_120a").unwrap();
        assert_eq!(arch.base, 120);
        assert_eq!(arch.to_nvcc_arch(), "sm_120a");

        let arch = GpuArch::parse("80").unwrap();
        assert_eq!(arch.base, 80);
        assert_eq!(arch.suffix, None);
        assert_eq!(arch.to_nvcc_arch(), "sm_80");
    }

    #[test]
    fn test_gpu_arch_auto_suffix() {
        assert_eq!(GpuArch::auto_suffix(80).to_nvcc_arch(), "sm_80");
        assert_eq!(GpuArch::auto_suffix(89).to_nvcc_arch(), "sm_89");
        assert_eq!(GpuArch::auto_suffix(90).to_nvcc_arch(), "sm_90a");
        assert_eq!(GpuArch::auto_suffix(100).to_nvcc_arch(), "sm_100a");
        assert_eq!(GpuArch::auto_suffix(120).to_nvcc_arch(), "sm_120a");
        assert_eq!(GpuArch::auto_suffix(121).to_nvcc_arch(), "sm_121f");
    }

    #[test]
    fn test_arch_lists() {
        let archs = parse_arch_list("90, 80;86 80").unwrap();
        assert_eq!(arch_key(&archs), "sm_80,sm_86,sm_90a");
        assert_eq!(
            gencode_args(&archs),
            [
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_90a,code=sm_90a",
            ]
        );
        // one value keys exactly as before
        assert_eq!(arch_key(&parse_arch_list("8.6").unwrap()), "sm_86");
        assert!(parse_arch_list(" , ").is_err());
        assert_eq!(
            arch_key(&parse_arch_list("121a,121f,121a").unwrap()),
            "sm_121a,sm_121f"
        );
        assert!(GpuArch::parse("8.").is_err());
        assert_eq!(GpuArch::parse("12.1").unwrap().to_nvcc_arch(), "sm_121f");
        // a mixed host builds for its first GPU unless CUDA_COMPUTE_CAP lists more
        assert_eq!(
            arch_key(&parse_nvidia_smi_output("compute_cap\n8.6\n9.0\n").unwrap()),
            "sm_86"
        );
    }

    #[test]
    fn test_gpu_arch_gencode() {
        // Pre-Hopper architectures (no suffix)
        assert_eq!(
            GpuArch::auto_suffix(75).to_gencode_arg(),
            "-gencode=arch=compute_75,code=sm_75"
        );
        assert_eq!(
            GpuArch::auto_suffix(80).to_gencode_arg(),
            "-gencode=arch=compute_80,code=sm_80"
        );
        assert_eq!(
            GpuArch::auto_suffix(89).to_gencode_arg(),
            "-gencode=arch=compute_89,code=sm_89"
        );

        // Hopper/Blackwell architectures (with 'a' suffix)
        assert_eq!(
            GpuArch::auto_suffix(90).to_gencode_arg(),
            "-gencode=arch=compute_90a,code=sm_90a"
        );
        assert_eq!(
            GpuArch::auto_suffix(100).to_gencode_arg(),
            "-gencode=arch=compute_100a,code=sm_100a"
        );

        // SM120 (RTX 5090/B200) - 'a' suffix for arch-specific NVFP4/MXFP4 MMA
        assert_eq!(
            GpuArch::auto_suffix(120).to_gencode_arg(),
            "-gencode=arch=compute_120a,code=sm_120a"
        );

        // SM121 (GB10/Spark) - 'f' suffix for family features only (no NVFP4 hardware)
        assert_eq!(
            GpuArch::auto_suffix(121).to_gencode_arg(),
            "-gencode=arch=compute_121f,code=sm_121f"
        );
    }
}
