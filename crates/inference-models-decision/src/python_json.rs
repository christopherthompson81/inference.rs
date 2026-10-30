//! JSON as Python's `json.dumps` writes it, which the decision models were trained to read.

use serde_json::Value;

/// `json.dumps`'s layout: an indent (one line without), the separators between items and after keys, ASCII escaping,
/// and whether floats take Python's `repr`.
pub struct PythonJson<'a> {
    pub indent: Option<usize>,
    pub item_sep: &'a str,
    pub key_sep: &'a str,
    pub ensure_ascii: bool,
    pub python_floats: bool,
}

impl PythonJson<'_> {
    /// `json.dumps(value, ensure_ascii=...)`'s defaults, with serde's float text.
    pub const fn default_layout(ensure_ascii: bool) -> PythonJson<'static> {
        PythonJson {
            indent: None,
            item_sep: ", ",
            key_sep: ": ",
            ensure_ascii,
            python_floats: false,
        }
    }

    pub fn dumps(&self, value: &Value) -> String {
        let mut out = String::new();
        self.write(value, 0, &mut out);
        out
    }

    fn newline(&self, level: usize, out: &mut String) {
        if let Some(indent) = self.indent {
            out.push('\n');
            out.push_str(&" ".repeat(indent * level));
        }
    }

    fn write(&self, value: &Value, level: usize, out: &mut String) {
        match value {
            Value::String(text) => python_string(text, self.ensure_ascii, out),
            Value::Number(n) if self.python_floats && n.is_f64() => {
                python_float(n.as_f64().unwrap_or_default(), out);
            }
            Value::Array(items) if !items.is_empty() => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(self.item_sep);
                    }
                    self.newline(level + 1, out);
                    self.write(item, level + 1, out);
                }
                self.newline(level, out);
                out.push(']');
            }
            Value::Object(map) if !map.is_empty() => {
                out.push('{');
                for (i, (key, item)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(self.item_sep);
                    }
                    self.newline(level + 1, out);
                    python_string(key, self.ensure_ascii, out);
                    out.push_str(self.key_sep);
                    self.write(item, level + 1, out);
                }
                self.newline(level, out);
                out.push('}');
            }
            Value::Array(_) => out.push_str("[]"),
            Value::Object(_) => out.push_str("{}"),
            other => out.push_str(&other.to_string()),
        }
    }
}

/// JSON as Python's `json.dumps` writes it, `", "` and `": "` between items, which the models read their inputs as.
pub fn python_json(value: &Value, ensure_ascii: bool, out: &mut String) {
    PythonJson::default_layout(ensure_ascii).write(value, 0, out);
}

/// Python's `repr` of a float: the shortest digits that read back, in exponent form below 1e-4 and from 1e16.
fn python_float(value: f64, out: &mut String) {
    let sci = format!("{value:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp = exp.parse::<i32>().unwrap_or_default();
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits = mantissa
        .chars()
        .filter(char::is_ascii_digit)
        .collect::<String>();
    let n = digits.len() as i32;
    out.push_str(sign);
    if !(-4..16).contains(&exp) {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!(
            "e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        ));
    } else if exp >= n - 1 {
        out.push_str(&digits);
        out.push_str(&"0".repeat((exp - (n - 1)) as usize));
        out.push_str(".0");
    } else if exp >= 0 {
        let point = (exp + 1) as usize;
        out.push_str(&digits[..point]);
        out.push('.');
        out.push_str(&digits[point..]);
    } else {
        out.push_str("0.");
        out.push_str(&"0".repeat((-exp - 1) as usize));
        out.push_str(&digits);
    }
}

fn python_string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ensure_ascii && (!c.is_ascii() || c == '\u{7f}')) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Text as the model reads it: a string as it is, any other value as `json.dumps` writes it.
pub fn as_text(value: &Value, ensure_ascii: bool) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => {
            let mut out = String::new();
            python_json(other, ensure_ascii, &mut out);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_text_values_read_as_python_writes_them() {
        let value = serde_json::json!({"plan": "pro", "n": 2, "tags": ["a", "é"], "ok": null});
        assert_eq!(
            as_text(&value, false),
            r#"{"plan": "pro", "n": 2, "tags": ["a", "é"], "ok": null}"#
        );
        assert_eq!(as_text(&serde_json::json!(["é"]), true), "[\"\\u00e9\"]");
        assert_eq!(as_text(&serde_json::json!("as is"), true), "as is");
    }
}
