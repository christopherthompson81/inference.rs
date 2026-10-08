//! The header declares exactly the functions the crate exports, and compiles as strict C99.

use std::{collections::BTreeSet, fs, path::Path, process::Command};

const HEADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/include/inference.h");
const SRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
// Mirrors tests/export_surface.py, which checks the same set against a built library's symbols.
const DECL_MARKER: &str = "INFERENCE_API";

fn declared() -> BTreeSet<String> {
    let header = fs::read_to_string(HEADER).unwrap();
    header
        .split(DECL_MARKER)
        .skip(1)
        .filter_map(|decl| {
            let name_end = decl.find('(')?;
            let name = decl[..name_end]
                .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                .next()?;
            name.starts_with("inference_").then(|| name.to_string())
        })
        .collect()
}

fn exported_in(dir: &Path, names: &mut BTreeSet<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            exported_in(&path, names);
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        for attribute in ["\n#[no_mangle]", "\n#[unsafe(no_mangle)]"] {
            for item in source.split(attribute).skip(1) {
                let signature = &item[..item.find(['(', '=', ';']).unwrap()];
                if let Some((_, name)) = signature.rsplit_once("fn ")
                    && !name.trim().starts_with('$')
                {
                    names.insert(name.trim().to_string());
                }
            }
        }
        // an `entry_points!` table line names its entry point before the `=>`
        for line in source.lines() {
            if let Some((head, _)) = line.split_once(" =>")
                && let Some(name) = head.split_whitespace().last()
                && name.starts_with("inference_")
            {
                names.insert(name.to_string());
            }
        }
    }
}

fn exported() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    exported_in(Path::new(SRC), &mut names);
    names
}

#[test]
fn header_declares_exactly_the_exports() {
    let (declared, exported) = (declared(), exported());
    assert!(!exported.is_empty());
    assert_eq!(
        declared.difference(&exported).collect::<Vec<_>>(),
        Vec::<&String>::new(),
        "declared but not exported"
    );
    assert_eq!(
        exported.difference(&declared).collect::<Vec<_>>(),
        Vec::<&String>::new(),
        "exported but not declared"
    );
}

#[test]
fn header_compiles_as_strict_c99() {
    let Ok(output) = Command::new("cc")
        .args([
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsyntax-only",
            "-x",
            "c",
        ])
        .arg(HEADER)
        .output()
    else {
        eprintln!("no C compiler (`cc`) on PATH; skipping");
        return;
    };
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
