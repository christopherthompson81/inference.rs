//! Non-finite floats in a checkpoint's `config.json`. Python's `json` writes them as bare `Infinity`, `-Infinity` and
//! `NaN`, which JSON (and serde_json) rejects; NeMo's exporter writes `{"__float__": "Infinity"}` instead. A config is
//! read with the bare literals rewritten to NeMo's form, which a field that takes them decodes.

/// The key of NeMo's encoding of a non-finite float.
pub const NON_FINITE_FLOAT_KEY: &str = "__float__";
const LITERALS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];

/// `text` with every bare non-finite literal outside a string as `{"__float__": "<literal>"}`.
pub fn encode_non_finite_floats(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut in_string = false;
    while let Some(c) = rest.chars().next() {
        if in_string {
            match c {
                '\\' => {
                    let escaped = rest.chars().nth(1).map_or(0, char::len_utf8);
                    out.push_str(&rest[..1 + escaped]);
                    rest = &rest[1 + escaped..];
                    continue;
                }
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        } else if let Some(literal) = LITERALS.iter().find(|literal| rest.starts_with(**literal)) {
            out.push_str(&format!("{{\"{NON_FINITE_FLOAT_KEY}\": \"{literal}\"}}"));
            rest = &rest[literal.len()..];
            continue;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::encode_non_finite_floats;

    #[test]
    fn bare_literals_become_nemos_encoding_and_strings_stay() {
        let text =
            r#"{"limit": [0.0, Infinity], "low": -Infinity, "x": NaN, "s": "Infinity \"NaN\""}"#;
        let value: serde_json::Value =
            serde_json::from_str(&encode_non_finite_floats(text)).unwrap();
        assert_eq!(value["limit"][1]["__float__"], "Infinity");
        assert_eq!(value["low"]["__float__"], "-Infinity");
        assert_eq!(value["x"]["__float__"], "NaN");
        assert_eq!(value["s"], "Infinity \"NaN\"");
        let plain = r#"{"a": 1.5, "b": "c"}"#;
        assert_eq!(encode_non_finite_floats(plain), plain);
    }
}
