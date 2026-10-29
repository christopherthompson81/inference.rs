//! JSON decoding without a serde visitor per type: text is parsed once into a [`serde_json::Value`], and each type
//! reads it through helpers generic only over the field type, so one copy serves every struct with a field of that
//! type. `#[derive(FromJson)]` reads the type's `#[serde(...)]` attributes, so serde's `Serialize` and utoipa's
//! schemas keep agreeing with the decoder, and it also implements `Deserialize` by way of a `Value`, so a migrated type
//! still nests in serde-derived containers and non-JSON formats.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::path::PathBuf;
use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};
use serde_json::{Map, Value};

extern crate self as inference_json;

pub use inference_json_derive::FromJson;

#[cfg(feature = "axum")]
mod query;
#[cfg(feature = "axum")]
pub use query::{QueryParams, QueryParamsRejection};

thread_local! {
    // set while a URL query or CSV row decodes: its scalars arrive as text
    static TEXT_SCALARS: Cell<bool> = const { Cell::new(false) };
}

/// Decodes `v` with numbers and bools read from strings, for sources whose values are all text.
pub fn from_text_map<T: FromJson>(v: &Value) -> Result<T, JsonError> {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEXT_SCALARS.with(|t| t.set(self.0));
        }
    }
    let _reset = Reset(TEXT_SCALARS.with(|t| t.replace(true)));
    T::from_json(v)
}

/// A URL query string (`a=1&b=x`) decoded into `T`; a repeated key is an error, as in serde's derive.
pub fn from_query<T: FromJson>(query: &str) -> Result<T, JsonError> {
    let mut map = Map::new();
    for (k, v) in form_urlencoded::parse(query.as_bytes()) {
        if map
            .insert(k.to_string(), Value::String(v.into_owned()))
            .is_some()
        {
            return Err(JsonError::new(format!("duplicate field `{k}`")));
        }
    }
    from_text_map(&Value::Object(map))
}

fn text_scalar(v: &Value) -> Option<&str> {
    match v {
        Value::String(s) if TEXT_SCALARS.with(Cell::get) => Some(s),
        _ => None,
    }
}

/// A decode failure: the message and the path to the value that failed (`models[0].model.Plain.dtype`).
#[derive(Debug, Clone, PartialEq)]
pub struct JsonError {
    path: String,
    message: String,
}

impl JsonError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            path: String::new(),
            message: message.into(),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// The error inside the value at `key`.
    pub fn at(mut self, key: &str) -> Self {
        self.path = match (self.path.is_empty(), self.path.starts_with('[')) {
            (true, _) => key.to_owned(),
            (false, true) => format!("{key}{}", self.path),
            (false, false) => format!("{key}.{}", self.path),
        };
        self
    }

    /// The error inside the item at `index`.
    pub fn at_index(mut self, index: usize) -> Self {
        self.path = match (self.path.is_empty(), self.path.starts_with('[')) {
            (true, _) => format!("[{index}]"),
            (false, true) => format!("[{index}]{}", self.path),
            (false, false) => format!("[{index}].{}", self.path),
        };
        self
    }

    pub fn invalid_type(got: &Value, expected: &str) -> Self {
        Self::new(format!(
            "invalid type: {}, expected {expected}",
            unexpected(got)
        ))
    }

    pub fn invalid_value(got: &Value, expected: &str) -> Self {
        Self::new(format!(
            "invalid value: {}, expected {expected}",
            unexpected(got)
        ))
    }

    pub fn missing_field(key: &str) -> Self {
        Self::new(format!("missing field `{key}`"))
    }

    pub fn unknown_field(key: &str, known: &[&str]) -> Self {
        Self::new(format!(
            "unknown field `{key}`, {}",
            one_of(known, "fields")
        ))
    }

    pub fn unknown_variant(got: &str, known: &[&str]) -> Self {
        Self::new(format!(
            "unknown variant `{got}`, {}",
            one_of(known, "variants")
        ))
    }

    pub fn untagged(ty: &str) -> Self {
        Self::new(format!(
            "data did not match any variant of untagged enum {ty}"
        ))
    }

    fn serde(e: serde_json::Error) -> Self {
        Self::new(e.to_string())
    }
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.path.is_empty() {
            true => f.write_str(&self.message),
            false => write!(f, "{}: {}", self.path, self.message),
        }
    }
}

impl std::error::Error for JsonError {}

impl From<serde_json::Error> for JsonError {
    fn from(e: serde_json::Error) -> Self {
        Self::serde(e)
    }
}

// serde's wording, so errors read the same as before the switch
fn unexpected(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => format!("boolean `{b}`"),
        Value::Number(n) if n.is_f64() => format!("floating point `{n}`"),
        Value::Number(n) => format!("integer `{n}`"),
        Value::String(s) => format!("string {s:?}"),
        Value::Array(_) => "sequence".into(),
        Value::Object(_) => "map".into(),
    }
}

fn one_of(known: &[&str], what: &str) -> String {
    match known {
        [] => format!("there are no {what}"),
        [one] => format!("expected `{one}`"),
        [first, second] => format!("expected `{first}` or `{second}`"),
        _ => {
            let list: Vec<String> = known.iter().map(|k| format!("`{k}`")).collect();
            format!("expected one of {}", list.join(", "))
        }
    }
}

/// A type that decodes from a JSON value.
pub trait FromJson: Sized {
    fn from_json(v: &Value) -> Result<Self, JsonError>;

    /// The value a struct field of this type takes when its key is absent. `None`: the key is required.
    fn absent() -> Option<Self> {
        None
    }
}

pub fn from_value<T: FromJson>(v: &Value) -> Result<T, JsonError> {
    T::from_json(v)
}

pub fn from_str<T: FromJson>(src: &str) -> Result<T, JsonError> {
    T::from_json(&parse_str(src)?)
}

pub fn from_slice<T: FromJson>(src: &[u8]) -> Result<T, JsonError> {
    T::from_json(&parse_slice(src)?)
}

/// serde_json's parser, compiled once here for every type.
pub fn parse_str(src: &str) -> Result<Value, JsonError> {
    serde_json::from_str(src).map_err(JsonError::serde)
}

pub fn parse_slice(src: &[u8]) -> Result<Value, JsonError> {
    serde_json::from_slice(src).map_err(JsonError::serde)
}

/// `FromJson` through a type's own `Deserialize`, for types whose decoding is hand-written.
pub fn from_serde<T: serde::de::DeserializeOwned>(v: &Value) -> Result<T, JsonError> {
    T::deserialize(v).map_err(JsonError::serde)
}

/// Implements `FromJson` for types with a hand-written `Deserialize` impl.
#[macro_export]
macro_rules! via_serde {
    ($($ty:ty),* $(,)?) => {
        $(impl $crate::FromJson for $ty {
            fn from_json(v: &$crate::__private::Value) -> ::core::result::Result<Self, $crate::JsonError> {
                $crate::from_serde(v)
            }
        })*
    };
}

/// Implements `FromJson` for types that parse from a JSON string with `FromStr`.
#[macro_export]
macro_rules! via_from_str {
    ($($ty:ty),* $(,)?) => {
        $(impl $crate::FromJson for $ty {
            fn from_json(v: &$crate::__private::Value) -> ::core::result::Result<Self, $crate::JsonError> {
                let s = $crate::__private::str(v, "a string")?;
                s.parse().map_err(|e| $crate::JsonError::new(::std::string::ToString::to_string(&e)))
            }
        })*
    };
}

/// What the derive calls; not a stable API.
#[doc(hidden)]
pub mod __private {
    pub use serde;
    pub use serde_json::{Map, Value};

    use super::{FromJson, JsonError};

    pub fn object<'a>(v: &'a Value, expected: &str) -> Result<&'a Map<String, Value>, JsonError> {
        v.as_object()
            .ok_or_else(|| JsonError::invalid_type(v, expected))
    }

    pub fn array<'a>(v: &'a Value, expected: &str) -> Result<&'a [Value], JsonError> {
        v.as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| JsonError::invalid_type(v, expected))
    }

    pub fn str<'a>(v: &'a Value, expected: &str) -> Result<&'a str, JsonError> {
        v.as_str()
            .ok_or_else(|| JsonError::invalid_type(v, expected))
    }

    /// The value at `key` or the first alias present, with the key it was found under.
    pub fn lookup<'a>(
        o: &'a Map<String, Value>,
        key: &'a str,
        aliases: &[&'a str],
    ) -> Option<(&'a str, &'a Value)> {
        o.get(key)
            .map(|v| (key, v))
            .or_else(|| aliases.iter().find_map(|a| o.get(*a).map(|v| (*a, v))))
    }

    pub fn field<T: FromJson>(
        o: &Map<String, Value>,
        key: &str,
        aliases: &[&str],
    ) -> Result<T, JsonError> {
        match lookup(o, key, aliases) {
            Some((k, v)) => T::from_json(v).map_err(|e| e.at(k)),
            None => T::absent().ok_or_else(|| JsonError::missing_field(key)),
        }
    }

    /// `#[serde(default)]`: an absent key takes `default`; a present `null` is still decoded, as in serde.
    pub fn field_or<T: FromJson>(
        o: &Map<String, Value>,
        key: &str,
        aliases: &[&str],
        default: T,
    ) -> Result<T, JsonError> {
        match lookup(o, key, aliases) {
            Some((k, v)) => T::from_json(v).map_err(|e| e.at(k)),
            None => Ok(default),
        }
    }

    /// `deserialize_with` / `with`: the field through a serde function, reading the `Value` as its deserializer.
    pub fn field_with<T>(
        o: &Map<String, Value>,
        key: &str,
        aliases: &[&str],
        default: Option<T>,
        f: fn(&Value) -> Result<T, serde_json::Error>,
    ) -> Result<T, JsonError> {
        match lookup(o, key, aliases) {
            Some((k, v)) => f(v).map_err(|e| JsonError::from(e).at(k)),
            None => default.ok_or_else(|| JsonError::missing_field(key)),
        }
    }

    /// A field through a `fn(&Value) -> Result<T, JsonError>`: native `either::serde_untagged`, or `#[json(serde)]`.
    pub fn field_json<T>(
        o: &Map<String, Value>,
        key: &str,
        aliases: &[&str],
        default: Option<T>,
        f: fn(&Value) -> Result<T, JsonError>,
    ) -> Result<T, JsonError> {
        match lookup(o, key, aliases) {
            Some((k, v)) => f(v).map_err(|e| e.at(k)),
            None => default.ok_or_else(|| JsonError::missing_field(key)),
        }
    }

    /// `#[serde(with = "either::serde_untagged")]`: the left type if it decodes, else the right.
    pub fn either_untagged<L: FromJson, R: FromJson>(
        v: &Value,
    ) -> Result<either::Either<L, R>, JsonError> {
        match L::from_json(v) {
            Ok(l) => Ok(either::Either::Left(l)),
            Err(_) => R::from_json(v)
                .map(either::Either::Right)
                .map_err(|_| JsonError::untagged("Either")),
        }
    }

    pub fn either_untagged_optional<L: FromJson, R: FromJson>(
        v: &Value,
    ) -> Result<Option<either::Either<L, R>>, JsonError> {
        match v {
            Value::Null => Ok(None),
            _ => either_untagged(v).map(Some),
        }
    }

    pub fn deny_unknown(o: &Map<String, Value>, known: &[&str]) -> Result<(), JsonError> {
        match o.keys().find(|k| !known.contains(&k.as_str())) {
            Some(k) => Err(JsonError::unknown_field(k, known)),
            None => Ok(()),
        }
    }

    /// An externally tagged enum's variant name and payload: `"Unit"` or `{"Variant": payload}`.
    pub fn external<'a>(
        v: &'a Value,
        expected: &str,
    ) -> Result<(&'a str, Option<&'a Value>), JsonError> {
        match v {
            Value::String(s) => Ok((s, None)),
            Value::Object(o) if o.len() == 1 => {
                let (k, v) = o.iter().next().expect("one entry");
                Ok((k, Some(v)))
            }
            _ => Err(JsonError::invalid_type(v, expected)),
        }
    }

    /// An internally tagged enum's variant name, and its object without the tag for a newtype variant to read.
    pub fn internal<'a>(v: &'a Value, tag: &str, expected: &str) -> Result<&'a str, JsonError> {
        let o = object(v, expected)?;
        match o.get(tag) {
            Some(t) => str(t, "a variant name").map_err(|e| e.at(tag)),
            None => Err(JsonError::missing_field(tag)),
        }
    }

    pub fn without_tag(v: &Value, tag: &str) -> Value {
        let mut o = v.as_object().cloned().unwrap_or_default();
        o.remove(tag);
        Value::Object(o)
    }

    pub fn unit_payload(payload: Option<&Value>, variant: &str) -> Result<(), JsonError> {
        match payload {
            None | Some(Value::Null) => Ok(()),
            Some(v) => Err(JsonError::invalid_type(v, "unit variant").at(variant)),
        }
    }

    pub fn payload<'a>(payload: Option<&'a Value>, variant: &str) -> Result<&'a Value, JsonError> {
        payload.ok_or_else(|| {
            JsonError::new(format!(
                "invalid type: unit variant, expected {variant} with data"
            ))
        })
    }

    pub fn tuple<'a>(v: &'a Value, len: usize, expected: &str) -> Result<&'a [Value], JsonError> {
        let a = array(v, expected)?;
        match a.len() == len {
            true => Ok(a),
            false => Err(JsonError::new(format!(
                "invalid length {}, expected {expected}",
                a.len()
            ))),
        }
    }

    pub fn item<T: FromJson>(a: &[Value], index: usize) -> Result<T, JsonError> {
        T::from_json(&a[index]).map_err(|e| e.at_index(index))
    }

    /// The `Deserialize` impl the derive writes: serde hands over a `Value`, which the type decodes.
    pub fn deserialize<'de, D, T>(d: D) -> Result<T, D::Error>
    where
        D: serde::Deserializer<'de>,
        T: FromJson,
    {
        let v = <Value as serde::Deserialize>::deserialize(d)?;
        T::from_json(&v).map_err(serde::de::Error::custom)
    }
}

impl FromJson for Value {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        Ok(v.clone())
    }
}

/// serde's encoding: `{"secs": u64, "nanos": u32}` or `[secs, nanos]`.
impl FromJson for std::time::Duration {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        let (secs, nanos) = match v {
            Value::Array(_) => <(u64, u32)>::from_json(v)?,
            _ => {
                let o = __private::object(v, "struct Duration")?;
                (
                    __private::field(o, "secs", &[])?,
                    __private::field(o, "nanos", &[])?,
                )
            }
        };
        Ok(std::time::Duration::new(secs, nanos))
    }
}

impl FromJson for serde::de::IgnoredAny {
    fn from_json(_: &Value) -> Result<Self, JsonError> {
        Ok(serde::de::IgnoredAny)
    }
}

impl FromJson for Map<String, Value> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        __private::object(v, "a map").cloned()
    }
}

impl FromJson for String {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        __private::str(v, "a string").map(str::to_owned)
    }
}

impl FromJson for PathBuf {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        __private::str(v, "path string").map(PathBuf::from)
    }
}

impl FromJson for char {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        let mut chars = v.as_str().map(str::chars);
        match chars.as_mut().map(|c| (c.next(), c.next())) {
            Some((Some(c), None)) => Ok(c),
            _ => Err(JsonError::invalid_type(v, "a character")),
        }
    }
}

impl FromJson for bool {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        if let Some(s) = text_scalar(v) {
            return s
                .parse()
                .map_err(|_| JsonError::invalid_value(v, "a boolean"));
        }
        v.as_bool()
            .ok_or_else(|| JsonError::invalid_type(v, "a boolean"))
    }
}

impl FromJson for () {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        match v {
            Value::Null => Ok(()),
            _ => Err(JsonError::invalid_type(v, "unit")),
        }
    }
}

fn integer<T: TryFrom<u64> + TryFrom<i64>>(v: &Value, name: &str) -> Result<T, JsonError> {
    if let Some(s) = text_scalar(v) {
        let parsed = match s.parse::<u64>() {
            Ok(u) => T::try_from(u).ok(),
            Err(_) => s.parse::<i64>().ok().and_then(|i| T::try_from(i).ok()),
        };
        return parsed.ok_or_else(|| JsonError::invalid_value(v, name));
    }
    let n = match v {
        Value::Number(n) => n,
        _ => return Err(JsonError::invalid_type(v, name)),
    };
    let fits = match (n.as_u64(), n.as_i64()) {
        (Some(u), _) => T::try_from(u).ok(),
        (None, Some(i)) => T::try_from(i).ok(),
        (None, None) => return Err(JsonError::invalid_type(v, name)),
    };
    fits.ok_or_else(|| JsonError::invalid_value(v, name))
}

macro_rules! integers {
    ($($t:ty),*) => {$(
        impl FromJson for $t {
            fn from_json(v: &Value) -> Result<Self, JsonError> {
                integer(v, stringify!($t))
            }
        }
    )*};
}
integers!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

macro_rules! nonzero {
    ($($t:ty),*) => {$(
        impl FromJson for std::num::NonZero<$t> {
            fn from_json(v: &Value) -> Result<Self, JsonError> {
                std::num::NonZero::new(<$t>::from_json(v)?)
                    .ok_or_else(|| JsonError::invalid_value(v, concat!("a nonzero ", stringify!($t))))
            }
        }
    )*};
}
nonzero!(u8, u16, u32, u64, usize);

impl FromJson for f64 {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        if let Some(s) = text_scalar(v) {
            return s.parse().map_err(|_| JsonError::invalid_value(v, "f64"));
        }
        v.as_f64().ok_or_else(|| JsonError::invalid_type(v, "f64"))
    }
}

impl FromJson for f32 {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        if let Some(s) = text_scalar(v) {
            return s.parse().map_err(|_| JsonError::invalid_value(v, "f32"));
        }
        v.as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| JsonError::invalid_type(v, "f32"))
    }
}

impl<T: FromJson> FromJson for Option<T> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        match v {
            Value::Null => Ok(None),
            _ => T::from_json(v).map(Some),
        }
    }

    fn absent() -> Option<Self> {
        Some(None)
    }
}

/// serde's own encoding of `Either`: `{"Left": l}` or `{"Right": r}` (`either::serde_untagged` is the other one).
impl<L: FromJson, R: FromJson> FromJson for either::Either<L, R> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        match __private::external(v, "enum Either")? {
            ("Left", Some(l)) => L::from_json(l)
                .map(either::Either::Left)
                .map_err(|e| e.at("Left")),
            ("Right", Some(r)) => R::from_json(r)
                .map(either::Either::Right)
                .map_err(|e| e.at("Right")),
            (other, _) => Err(JsonError::unknown_variant(other, &["Left", "Right"])),
        }
    }
}

impl<T: FromJson> FromJson for Box<T> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        T::from_json(v).map(Box::new)
    }
}

impl<T: FromJson> FromJson for Arc<T> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        T::from_json(v).map(Arc::new)
    }
}

fn items<T: FromJson>(
    v: &Value,
) -> Result<impl Iterator<Item = Result<T, JsonError>> + '_, JsonError> {
    let a = __private::array(v, "a sequence")?;
    Ok(a.iter()
        .enumerate()
        .map(|(i, x)| T::from_json(x).map_err(|e| e.at_index(i))))
}

impl<T: FromJson> FromJson for Vec<T> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        items(v)?.collect()
    }
}

impl<T: FromJson> FromJson for VecDeque<T> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        items(v)?.collect()
    }
}

impl<T: FromJson + Eq + Hash, S: BuildHasher + Default> FromJson for HashSet<T, S> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        items(v)?.collect()
    }
}

impl<T: FromJson + Ord> FromJson for BTreeSet<T> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        items(v)?.collect()
    }
}

impl<T: FromJson + Eq + Hash> FromJson for IndexSet<T> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        items(v)?.collect()
    }
}

impl<T: FromJson, const N: usize> FromJson for [T; N] {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        let expected = format!("an array of length {N}");
        let a = __private::tuple(v, N, &expected)?;
        let got: Vec<T> = (0..N)
            .map(|i| __private::item(a, i))
            .collect::<Result<_, _>>()?;
        got.try_into().map_err(|_| JsonError::new(expected))
    }
}

macro_rules! tuples {
    ($($len:literal => ($($t:ident $i:tt),*);)*) => {$(
        impl<$($t: FromJson),*> FromJson for ($($t,)*) {
            fn from_json(v: &Value) -> Result<Self, JsonError> {
                let a = __private::tuple(v, $len, concat!("a tuple of size ", $len))?;
                Ok(($(__private::item::<$t>(a, $i)?,)*))
            }
        }
    )*};
}
tuples! {
    1 => (A 0);
    2 => (A 0, B 1);
    3 => (A 0, B 1, C 2);
    4 => (A 0, B 1, C 2, D 3);
}

/// A map key: JSON object keys are strings, and a numeric key type parses one, as serde_json does.
pub trait FromJsonKey: Sized {
    fn from_key(k: &str) -> Result<Self, JsonError>;
}

impl FromJsonKey for String {
    fn from_key(k: &str) -> Result<Self, JsonError> {
        Ok(k.to_owned())
    }
}

impl FromJsonKey for PathBuf {
    fn from_key(k: &str) -> Result<Self, JsonError> {
        Ok(PathBuf::from(k))
    }
}

macro_rules! numeric_keys {
    ($($t:ty),*) => {$(
        impl FromJsonKey for $t {
            fn from_key(k: &str) -> Result<Self, JsonError> {
                k.parse().map_err(|_| JsonError::new(format!("invalid map key `{k}`, expected {}", stringify!($t))))
            }
        }
    )*};
}
numeric_keys!(u8, u16, u32, u64, usize, i32, i64);

fn entries<K: FromJsonKey, V: FromJson>(
    v: &Value,
) -> Result<impl Iterator<Item = Result<(K, V), JsonError>> + '_, JsonError> {
    let o = __private::object(v, "a map")?;
    Ok(o.iter()
        .map(|(k, x)| Ok((K::from_key(k)?, V::from_json(x).map_err(|e| e.at(k))?))))
}

impl<K: FromJsonKey + Eq + Hash, V: FromJson, S: BuildHasher + Default> FromJson
    for HashMap<K, V, S>
{
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        entries(v)?.collect()
    }
}

impl<K: FromJsonKey + Ord, V: FromJson> FromJson for BTreeMap<K, V> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        entries(v)?.collect()
    }
}

impl<K: FromJsonKey + Eq + Hash, V: FromJson> FromJson for IndexMap<K, V> {
    fn from_json(v: &Value) -> Result<Self, JsonError> {
        entries(v)?.collect()
    }
}

#[cfg(test)]
mod tests;
