use anyhow::Result;

use inference_selection::{DoctorStatus, run_doctor as run_doctor_report};

const UNKNOWN: &str = "unknown";

pub fn run_doctor(json: bool) -> Result<()> {
    if json {
        let report = run_doctor_report();
        let out = serde_json::to_string_pretty(&report)?;
        println!("{out}");
        return Ok(());
    }

    let report = run_doctor_report();
    let system = &report.system;

    // Header
    println!();
    println!("Environment Diagnosis");
    println!("---------------------");
    println!(
        "[INFO] OS: {} ({})",
        system.os.as_deref().unwrap_or("unknown"),
        system.kernel.as_deref().unwrap_or("unknown")
    );

    // CPU info with extensions
    let mut cpu_ext = Vec::new();
    if system.cpu.avx {
        cpu_ext.push("AVX");
    }
    if system.cpu.avx2 {
        cpu_ext.push("AVX2");
    }
    if system.cpu.fma {
        cpu_ext.push("FMA");
    }
    if system.cpu.avx512 {
        cpu_ext.push("AVX-512");
    }
    let ext_str = if cpu_ext.is_empty() {
        "none".to_string()
    } else {
        cpu_ext.join(", ")
    };

    println!(
        "[INFO] CPU: {} ({} cores, extensions: {})",
        system.cpu.brand.as_deref().unwrap_or("unknown"),
        system.cpu.logical_cores,
        ext_str
    );
    println!(
        "[INFO] RAM: {:.1} GB total, {:.1} GB available",
        system.memory.total_bytes as f64 / 1e9,
        system.memory.available_bytes as f64 / 1e9
    );

    // Accelerator section
    let gpu_devices: Vec<_> = system.devices.iter().filter(|d| d.kind != "cpu").collect();
    if !gpu_devices.is_empty() {
        println!();
        println!("Accelerator Check");
        println!("-----------------");

        for dev in gpu_devices {
            let label = match dev.ordinal {
                Some(ord) => format!("{}[{}]", dev.kind.to_uppercase(), ord),
                None => dev.kind.to_uppercase(),
            };
            let total = dev
                .total_memory_bytes
                .map(|v| format!("{:.1} GB", v as f64 / 1e9))
                .unwrap_or_else(|| "unknown".to_string());
            let avail = dev
                .available_memory_bytes
                .map(|v| format!("{:.1} GB", v as f64 / 1e9))
                .unwrap_or_else(|| "unknown".to_string());

            // Include compute capability and flash attention status if available
            let cc_str = if let Some((major, minor)) = dev.compute_capability {
                let fa_v2 = if dev.flash_attn_compatible == Some(true) {
                    "✅"
                } else {
                    "❌"
                };
                let fa_v3 = if dev.flash_attn_v3_compatible == Some(true) {
                    "✅"
                } else {
                    "❌"
                };
                format!(" - Compute {major}.{minor} (FA v2: {fa_v2}, v3: {fa_v3})")
            } else {
                String::new()
            };

            println!("[INFO] {label}: {total} total, {avail} free{cc_str}");
        }

        let toolchain = &report.toolchain;
        if system.build.cuda {
            println!(
                "[INFO] CUDA: build {}, local nvcc {}, driver {} (supports CUDA {})",
                system
                    .build
                    .cuda_toolkit_version
                    .as_deref()
                    .unwrap_or(UNKNOWN),
                toolchain.nvcc.as_deref().unwrap_or(UNKNOWN),
                toolchain.nvidia_driver.as_deref().unwrap_or(UNKNOWN),
                toolchain.driver_cuda.as_deref().unwrap_or(UNKNOWN),
            );
        }
        if system.build.metal {
            println!(
                "[INFO] Metal: {}",
                toolchain.xcode.as_deref().unwrap_or("Xcode unknown")
            );
        }
    }

    // Installation section
    println!();
    println!("inference.rs Installation");
    println!("-----------------------");
    println!("[INFO] Version: {}", system.build.version);
    println!("[INFO] Git revision: {}", system.build.git_revision);

    let mut features = Vec::new();
    if system.build.cuda {
        features.push("cuda");
    }
    if system.build.metal {
        features.push("metal");
    }
    if system.build.cudnn {
        features.push("cudnn");
    }
    if system.build.flash_attn {
        features.push("flash-attn");
    }
    if system.build.flash_attn_v3 {
        features.push("flash-attn-v3");
    }
    if system.build.cutile {
        features.push("cutile");
    }
    if system.build.accelerate {
        features.push("accelerate");
    }
    if system.build.mkl {
        features.push("mkl");
    }
    let features_str = if features.is_empty() {
        "none".to_string()
    } else {
        features.join(", ")
    };
    println!("[INFO] Build features: {features_str}");

    // Checks section
    println!();
    println!("System Checks");
    println!("-------------");

    let mut warn_count = 0;
    let mut error_count = 0;

    for check in &report.checks {
        let (status_str, emoji) = match check.status {
            DoctorStatus::Ok => ("PASS", "✅"),
            DoctorStatus::Warn => {
                warn_count += 1;
                ("WARN", "⚠️")
            }
            DoctorStatus::Error => {
                error_count += 1;
                ("ERROR", "❌")
            }
        };
        println!("[{status_str}] {emoji} {}", check.message);
        if let Some(suggestion) = &check.suggestion {
            println!("       hint: {suggestion}");
        }
    }

    // Summary
    println!();
    println!("Summary");
    println!("-------");
    if error_count > 0 {
        println!(
            "❌ {} error(s) found. Please address the issues above.",
            error_count
        );
    } else if warn_count > 0 {
        println!(
            "⚠️ {} warning(s) found. System is functional but may have issues.",
            warn_count
        );
    } else {
        println!("✅ Your system is healthy. Ready to infer! 🚀");
    }
    println!();

    Ok(())
}
