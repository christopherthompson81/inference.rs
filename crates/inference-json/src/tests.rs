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
        pub x: u16,
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
    #[serde(rename_all = "lowercase")]
    pub enum Choice {
        Auto,
        None,
        #[serde(untagged)]
        Named(Named),
        #[serde(untagged)]
        Free(String),
    }

    #[derive(Debug)]
    pub struct Named {
        name: String,
    }

    #[derive(Debug)]
    pub struct Content(#[serde(with = "either::serde_untagged")] either::Either<String, Vec<u8>>);

    #[derive(Debug)]
    pub struct MaybeContent {
        #[serde(default, with = "either::serde_untagged_optional")]
        inner: Option<either::Either<u8, String>>,
        tagged: either::Either<u8, String>,
        count: std::num::NonZero<u32>,
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
    for v in [
        json!("auto"),
        json!("none"),
        json!({"name": "f"}),
        json!("free text"),
        json!(3),
    ] {
        let _ = same!(Choice, v);
    }
    for v in [json!("text"), json!([1, 2]), json!(3)] {
        let _ = same!(Content, v);
    }
    for v in [
        json!({"tagged": {"Left": 1}, "count": 2}),
        json!({"inner": "s", "tagged": {"Right": "r"}, "count": 1}),
        json!({"inner": null, "tagged": {"Left": 1}, "count": 1}),
        json!({"tagged": {"Up": 1}, "count": 1}),
        json!({"tagged": {"Left": 1}, "count": 0}),
    ] {
        let _ = same!(MaybeContent, v);
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
fn durations_read_serdes_encoding() {
    let d = std::time::Duration::new(3, 500);
    let encoded = serde_json::to_value(d).unwrap();
    assert_eq!(
        crate::from_value::<std::time::Duration>(&encoded).unwrap(),
        d
    );
    assert_eq!(
        crate::from_value::<std::time::Duration>(&json!([3, 500])).unwrap(),
        d
    );
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
fn a_json_serde_field_decodes_through_its_own_deserialize() {
    #[derive(crate::FromJson, Debug)]
    struct Foreign(#[json(serde)] std::net::Ipv4Addr);
    assert_eq!(
        crate::from_value::<Foreign>(&json!("10.0.0.1")).unwrap().0,
        std::net::Ipv4Addr::new(10, 0, 0, 1)
    );
    assert!(crate::from_value::<Foreign>(&json!("nope")).is_err());
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

#[derive(crate::FromJson, Debug, PartialEq)]
struct Listing {
    source: Option<String>,
    limit: Option<usize>,
    #[serde(default)]
    verbose: bool,
    ratio: Option<f32>,
}

#[test]
fn query_strings_read_their_scalars_from_text() {
    let q: Listing = crate::from_query("source=a%20b&limit=5&verbose=true&ratio=0.5").unwrap();
    let expected = Listing {
        source: Some("a b".into()),
        limit: Some(5),
        verbose: true,
        ratio: Some(0.5),
    };
    assert_eq!(q, expected);
    assert_eq!(crate::from_query::<Listing>("").unwrap().limit, None);
    let e = crate::from_query::<Listing>("limit=many").unwrap_err();
    assert_eq!(
        e.to_string(),
        "limit: invalid value: string \"many\", expected usize"
    );
    let e = crate::from_query::<Listing>("source=a&source=b").unwrap_err();
    assert_eq!(e.to_string(), "duplicate field `source`");
}

#[test]
fn text_scalars_stay_inside_their_call() {
    assert!(crate::from_query::<Listing>("limit=5").is_ok());
    assert!(crate::from_value::<u32>(&json!("5")).is_err());
}

#[test]
fn parsed_text_may_not_repeat_a_key() {
    let e = crate::from_str::<serde_json::Value>(r#"{"a": 1, "b": {"c": 1, "c": 2}}"#).unwrap_err();
    assert!(e.to_string().starts_with("duplicate field `c`"), "{e}");
    assert!(!e.is_syntax());
    assert!(
        crate::from_str::<serde_json::Value>(r#"{"a": "#)
            .unwrap_err()
            .is_syntax()
    );
    let v = crate::parse_str(r#"{"a": [1, -2, 2.5, "x", null, true, {"b": {}}]}"#).unwrap();
    assert_eq!(v, json!({"a": [1, -2, 2.5, "x", null, true, {"b": {}}]}));
}

#[test]
fn files_read_through_the_bridge_keep_the_last_of_a_repeated_key() {
    #[derive(serde::Deserialize)]
    struct Holder {
        inner: json_side::AllDefault,
    }
    let h: Holder = serde_json::from_str(r#"{"inner": {"x": 1, "x": 2}}"#).unwrap();
    assert_eq!(h.inner.x, 2);
}
