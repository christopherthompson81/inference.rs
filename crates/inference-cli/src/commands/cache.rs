//! Hugging Face cache management commands

use anyhow::Result;
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};
use inference_api::model_inventory::{delete_hf_cache_repo, hf_cache_listing};

/// List all cached models
pub fn run_cache_list() -> Result<()> {
    let listing = hf_cache_listing(None)
        .ok_or_else(|| anyhow::anyhow!("Cannot determine Hugging Face hub cache directory"))?;

    println!();
    println!("Hugging Face model cache");
    println!("-----------------------");
    println!();

    if listing.repos.is_empty() {
        println!("No cached models found.");
        println!();
        println!("Cache directory: {}", listing.path);
        return Ok(());
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Model"),
            Cell::new("Size"),
            Cell::new("Last Used"),
        ]);
    for repo in &listing.repos {
        table.add_row(vec![
            Cell::new(&repo.repo_id),
            Cell::new(format_size(repo.size_bytes)),
            Cell::new(last_modified(repo.modified)),
        ]);
    }
    let total_size = listing.repos.iter().map(|repo| repo.size_bytes).sum();
    let model_count = listing.repos.len();
    println!("{table}");
    println!();
    println!(
        "Total: {} model{}, {}",
        model_count,
        if model_count == 1 { "" } else { "s" },
        format_size(total_size)
    );
    println!();
    println!("Cache directory: {}", listing.path);
    println!();

    Ok(())
}

/// Delete a specific model from cache
pub fn run_cache_delete(model_id: &str) -> Result<()> {
    let deleted = delete_hf_cache_repo(model_id, None).map_err(|error| {
        anyhow::anyhow!("{error}\nRun 'inference cache list' to see cached models.")
    })?;
    println!(
        "✅ Deleted '{}' ({})",
        deleted.repo_id,
        format_size(deleted.freed_bytes)
    );
    Ok(())
}

/// Human-readable age of a Unix-seconds timestamp
fn last_modified(modified: Option<u64>) -> String {
    let Some(modified) = modified else {
        return "unknown".to_string();
    };
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |now| now.as_secs().saturating_sub(modified));
    let days = elapsed / 86400;
    if days == 0 {
        "today".to_string()
    } else if days == 1 {
        "yesterday".to_string()
    } else if days < 7 {
        format!("{} days ago", days)
    } else if days < 30 {
        format!("{} weeks ago", days / 7)
    } else if days < 365 {
        format!("{} months ago", days / 30)
    } else {
        format!("{} years ago", days / 365)
    }
}

/// Format bytes as human-readable size
pub(super) fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}
