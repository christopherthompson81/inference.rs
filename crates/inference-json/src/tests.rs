use serde_json::json;

use crate::JsonError;

// Defines the items twice, deriving serde's `Deserialize` in `serde_side` and `FromJson` in `json_side`.
macro_rules! twins {
    ($($item:item)*) => {
        #[allow(dead_code)]
        mod serde_side {
            use std::collections::HashMap;
            $(#[derive(serde::Deserialize)] $item)*
        }
        #[allow(dead_code)]
        mod json_side {
            use std::collections::HashMap;
            $(#[derive(crate::FromJson)] $item)*
        }
    };
}

fn one() -> u32 {
    1
}

twins! {
    #[derive(Debug, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct Inner {
        pub unit_words: Vec<String>,
        maybe: Option<u32>,
        #[serde(default)]
        flag: bool,
        #[serde(default = "super::one", alias = "cnt")]
        count: u32,
    }

    #[derive(Debug)]
    #[serde(deny_unknown_fields)]
    pub struct Strict {
        a: String,
        #[serde(skip)]
        hidden: u8,
        #[serde(skip_serializing_if = "Option::is_none")]
        b: Option<i8>,
    }

    #[derive(Debug, Default)]
    #[serde(default)]
    pub struct AllDefault {
        x: u16,
        y: String,
    }

    #[derive(Debug)]
    pub struct Outer {
        #[serde(flatten)]
        inner: Inner,
        #[serde(rename = "p")]
        pos: Pos,
        e: Either,
        map: HashMap<String, f32>,
        pair: (u8, String),
    }

    #[derive(Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum Pos {
        BeforeAll,
        #[serde(alias = "behind")]
        After,
    }

    #[derive(Debug)]
    #[serde(untagged)]
    pub enum Either {
        One(String),
        Many(HashMap<String, String>),
        Shaped { width: u32 },
        Nothing,
    }

    #[derive(Debug)]
    pub enum External {
        Unit,
        Newtype(u32),
        Pair(u8, u8),
        Fields { a: String, #[serde(default)] b: bool },
    }

    #[derive(Debug)]
    #[serde(tag = "type", rename_all = "lowercase")]
    pub enum Internal {
        Text { text: String },
        Image(Img),
        Empty,
    }

    #[derive(Debug)]
    #[serde(deny_unknown_fields)]
    pub struct Img {
        url: String,
    }

    #[derive(Debug)]
    #[serde(tag = "t", content = "c")]
    pub enum Adjacent {
        Num(i64),
        Word { w: String },
        Plain,
    }

    #[derive(Debug)]
    #[serde(transparent)]
    pub struct Wrapper {
        value: Vec<u8>,
    }

    #[derive(Debug)]
    pub struct Newtype(String);

    #[derive(Debug)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum Loud {
        QuietOne,
        #[serde(other)]
        Unknown,
    }

    #[derive(Debug)]
    pub struct Generic<T> {
        item: T,
        items: Vec<T>,
    }
}

macro_rules! same {
    ($ty:ident $(<$g:ty>)?, $value:expr) => {{
        let v = $value;
        let a = <serde_side::$ty $(<$g>)? as serde::Deserialize>::deserialize(&v).map(|x| format!("{x:?}"));
        let b = crate::from_value::<json_side::$ty $(<$g>)?>(&v).map(|x| format!("{x:?}"));
        match (&a, &b) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "{v}"),
            (Err(_), Err(_)) => {}
            _ => panic!("{v}: serde {a:?}, FromJson {b:?}"),
        }
        b
    }};
}

#[test]
fn structs_decode_as_serde_does() {
    assert!(same!(Inner, json!({"unitWords": ["x"], "flag": true, "cnt": 4})).is_ok());
    assert!(same!(Inner, json!({"unitWords": [], "maybe": 3})).is_ok());
    assert!(same!(Inner, json!({})).is_err());
    assert!(same!(Inner, json!({"unitWords": [], "flag": null})).is_err());
    assert!(same!(Strict, json!({"a": "s", "b": -3})).is_ok());
    assert!(same!(Strict, json!({"a": "s", "c": 1})).is_err());
    assert!(same!(Strict, json!({"a": "s", "hidden": 1})).is_err());
    assert!(same!(AllDefault, json!({"y": "z"})).is_ok());
    assert!(same!(Wrapper, json!([1, 2])).is_ok());
    assert!(same!(Newtype, json!("n")).is_ok());
    assert!(same!(Generic<u8>, json!({"item": 1, "items": [2, 3]})).is_ok());
    assert!(same!(Generic<u8>, json!({"item": 300, "items": []})).is_err());
    let outer = json!({"unitWords": [], "p": "before_all", "e": {"k": "v"}, "map": {"a": 1.5}, "pair": [1, "x"]});
    assert!(same!(Outer, outer).is_ok());
}

#[test]
fn enums_decode_as_serde_does() {
    for v in [
        json!("before_all"),
        json!("after"),
        json!("behind"),
        json!("BeforeAll"),
        json!(1),
    ] {
        let _ = same!(Pos, v);
    }
    for v in [
        json!("s"),
        json!({"k": "v"}),
        json!({"width": 3}),
        json!(null),
        json!(3),
    ] {
        let _ = same!(Either, v);
    }
    for v in [
        json!("Unit"),
        json!({"Newtype": 5}),
        json!({"Pair": [1, 2]}),
        json!({"Pair": [1]}),
        json!({"Fields": {"a": "x"}}),
        json!({"Fields": {"b": true}}),
        json!("Newtype"),
        json!({"Nope": 1}),
        json!({"Unit": null, "Newtype": 1}),
    ] {
        let _ = same!(External, v);
    }
    for v in [
        json!({"type": "text", "text": "hi"}),
        json!({"type": "image", "url": "u"}),
        json!({"type": "image", "url": "u", "extra": 1}),
        json!({"type": "empty"}),
        json!({"type": "video"}),
        json!({"text": "hi"}),
    ] {
        let _ = same!(Internal, v);
    }
    for v in [
        json!({"t": "Num", "c": 4}),
        json!({"t": "Word", "c": {"w": "x"}}),
        json!({"t": "Plain"}),
        json!({"c": 1}),
    ] {
        let _ = same!(Adjacent, v);
    }
    for v in [json!("QUIET_ONE"), json!("anything")] {
        let _ = same!(Loud, v);
    }
}

#[test]
fn numbers_keep_serdes_range_checks() {
    assert!(crate::from_value::<u8>(&json!(255)).is_ok());
    assert!(crate::from_value::<u8>(&json!(256)).is_err());
    assert!(crate::from_value::<u32>(&json!(-1)).is_err());
    assert!(crate::from_value::<u32>(&json!(1.5)).is_err());
    assert!(crate::from_value::<i8>(&json!(-128)).is_ok());
    assert_eq!(crate::from_value::<f32>(&json!(2)).unwrap(), 2.0);
}

#[test]
fn errors_name_the_path_to_the_failing_value() {
    let e: JsonError = crate::from_str::<json_side::Outer>(
        r#"{"unitWords": ["x", 2], "p": "after", "e": "s", "map": {}, "pair": [1, "x"]}"#,
    )
    .unwrap_err();
    assert_eq!(
        e.to_string(),
        "unitWords[1]: invalid type: integer `2`, expected a string"
    );
    let e = crate::from_value::<json_side::External>(&json!({"Fields": {"a": 3}})).unwrap_err();
    assert_eq!(
        e.to_string(),
        "Fields.a: invalid type: integer `3`, expected a string"
    );
    let e = crate::from_value::<json_side::Strict>(&json!({"a": "s", "c": 1})).unwrap_err();
    assert_eq!(e.to_string(), "unknown field `c`, expected `a` or `b`");
    let e = crate::from_value::<json_side::Pos>(&json!("middle")).unwrap_err();
    assert_eq!(
        e.to_string(),
        "unknown variant `middle`, expected `before_all` or `after`"
    );
}

#[test]
fn the_derived_deserialize_nests_in_serde_containers() {
    #[derive(serde::Deserialize, Debug)]
    pub struct Holder {
        inner: json_side::Inner,
    }
    let h: Holder = serde_json::from_str(r#"{"inner": {"unitWords": ["a"]}}"#).unwrap();
    assert_eq!(h.inner.unit_words, ["a"]);
    let e = serde_json::from_str::<Holder>(r#"{"inner": {"unitWords": [1]}}"#).unwrap_err();
    assert!(e.to_string().contains("unitWords[0]: invalid type"), "{e}");
}
