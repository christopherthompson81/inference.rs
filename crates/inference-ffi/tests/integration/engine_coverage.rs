//! Every public `Engine` method reaches the C ABI, so a server built on the bindings can do what ours does.

const ENGINE_SOURCE: &str = include_str!("../../../inference-api/src/engine.rs");
const FFI_SOURCES: [&str; 3] = [
    include_str!("../../src/engine.rs"),
    include_str!("../../src/callbacks.rs"),
    include_str!("../../src/lib.rs"),
];

// Methods the ABI covers some other way, or that a binding has no use for.
const NOT_EXPORTED: [(&str, &str); 9] = [
    (
        "load_with_callbacks",
        "the ABI loads through load_json, which takes the callbacks",
    ),
    (
        "adapter_config",
        "the adapter policy, from the caller's own spec",
    ),
    (
        "agent_permission",
        "the agent policy, from the caller's own spec",
    ),
    ("skill_store", "the store the skill entries already reach"),
    ("owner", "the caller made the owner's handle"),
    (
        "chat_with_approver",
        "a binding answers approvals from the stream's events with resolve_approval",
    ),
    ("shutdown", "freeing the last handle shuts the engine down"),
    (
        "default_model_id",
        "the models list marks the default model",
    ),
    (
        "calibration",
        "exported as calibration_start, calibration_status and calibration_apply",
    ),
];

fn engine_methods() -> Vec<&'static str> {
    let body = ENGINE_SOURCE
        .split_once("\nimpl Engine {\n")
        .expect("engine.rs has `impl Engine`")
        .1;
    let body = &body[..body.find("\n}\n").expect("`impl Engine` closes")];
    body.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let rest = line
                .strip_prefix("pub fn ")
                .or_else(|| line.strip_prefix("pub async fn "))?;
            Some(
                &rest[..rest
                    .find(['(', '<'])
                    .expect("a method name ends at its parameters")],
            )
        })
        .collect()
}

// A call `.name(` or a path `::name` handed on as a function, but not a longer name that starts with it.
fn mentions(source: &str, name: &str) -> bool {
    source.match_indices(name).any(|(at, _)| {
        let before = &source[..at];
        let after = source[at + name.len()..].chars().next();
        (before.ends_with('.') || before.ends_with("::"))
            && !after.is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

fn ffi_calls(name: &str) -> bool {
    let json = format!("{name}_json");
    FFI_SOURCES
        .iter()
        .any(|source| mentions(source, name) || mentions(source, &json))
}

#[test]
fn every_engine_method_has_an_abi_entry_or_a_reason() {
    let methods = engine_methods();
    assert!(methods.len() > 40, "found only {methods:?}");
    let missing: Vec<_> = methods
        .iter()
        .filter(|name| !ffi_calls(name) && !NOT_EXPORTED.iter().any(|(listed, _)| listed == *name))
        .collect();
    assert!(
        missing.is_empty(),
        "Engine methods with no ABI entry: {missing:?}"
    );
    for (listed, _) in NOT_EXPORTED {
        assert!(
            methods.contains(&listed),
            "`{listed}` is no longer an Engine method"
        );
        assert!(
            !ffi_calls(listed),
            "`{listed}` is exported now; drop it from NOT_EXPORTED"
        );
    }
}
