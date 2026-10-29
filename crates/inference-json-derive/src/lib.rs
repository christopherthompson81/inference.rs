//! `#[derive(FromJson)]`: decodes a struct or enum from a `serde_json::Value` through `inference_json`'s shared
//! helpers, reading the type's `#[serde(...)]` attributes with serde's meaning, and implements `Deserialize` by way of
//! a `Value`. Serialize-only attributes are ignored; an attribute the decoder cannot honor is a compile error.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens};
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Expr, Fields, Ident, LitStr, Path, Type, meta::ParseNestedMeta,
    parse_macro_input, parse_quote,
};

#[proc_macro_derive(FromJson, attributes(serde))]
pub fn derive_from_json(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[derive(Default)]
struct Container {
    rename_all: Option<String>,
    default: Option<Option<Path>>,
    deny_unknown_fields: bool,
    tag: Option<String>,
    content: Option<String>,
    untagged: bool,
    transparent: bool,
    from: Option<Type>,
    try_from: Option<Type>,
    expecting: Option<String>,
}

#[derive(Default)]
struct Member {
    rename: Option<String>,
    aliases: Vec<String>,
    default: Option<Option<Path>>,
    skip: bool,
    flatten: bool,
    deserialize_with: Option<Path>,
    other: bool,
}

fn lit_str(m: &ParseNestedMeta) -> syn::Result<LitStr> {
    m.value()?.parse::<LitStr>()
}

/// `rename = "x"` or `rename(deserialize = "x", serialize = "y")`: the name the decoder reads, if any.
fn de_name(m: &ParseNestedMeta) -> syn::Result<Option<String>> {
    if m.input.peek(syn::Token![=]) {
        return Ok(Some(lit_str(m)?.value()));
    }
    let mut name = None;
    m.parse_nested_meta(|inner| {
        let value = lit_str(&inner)?.value();
        if inner.path.is_ident("deserialize") {
            name = Some(value);
        }
        Ok(())
    })?;
    Ok(name)
}

fn skip_value(m: &ParseNestedMeta) -> syn::Result<()> {
    if m.input.peek(syn::Token![=]) {
        m.value()?.parse::<Expr>()?;
    } else if m.input.peek(syn::token::Paren) {
        m.parse_nested_meta(|inner| skip_value(&inner))?;
    }
    Ok(())
}

fn container_attrs(attrs: &[Attribute]) -> syn::Result<Container> {
    let mut c = Container::default();
    for a in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        a.parse_nested_meta(|m| {
            let p = &m.path;
            if p.is_ident("rename_all") {
                c.rename_all = de_name(&m)?;
            } else if p.is_ident("default") {
                c.default = Some(match m.input.peek(syn::Token![=]) {
                    true => Some(lit_str(&m)?.parse::<Path>()?),
                    false => None,
                });
            } else if p.is_ident("deny_unknown_fields") {
                c.deny_unknown_fields = true;
            } else if p.is_ident("tag") {
                c.tag = Some(lit_str(&m)?.value());
            } else if p.is_ident("content") {
                c.content = Some(lit_str(&m)?.value());
            } else if p.is_ident("untagged") {
                c.untagged = true;
            } else if p.is_ident("transparent") {
                c.transparent = true;
            } else if p.is_ident("from") {
                c.from = Some(lit_str(&m)?.parse()?);
            } else if p.is_ident("try_from") {
                c.try_from = Some(lit_str(&m)?.parse()?);
            } else if p.is_ident("expecting") {
                c.expecting = Some(lit_str(&m)?.value());
            } else if p.is_ident("rename")
                || p.is_ident("into")
                || p.is_ident("bound")
                || p.is_ident("crate")
            {
                skip_value(&m)?;
            } else {
                return Err(m.error("FromJson: unsupported #[serde] container attribute"));
            }
            Ok(())
        })?;
    }
    Ok(c)
}

fn member_attrs(attrs: &[Attribute]) -> syn::Result<Member> {
    let mut f = Member::default();
    for a in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        a.parse_nested_meta(|m| {
            let p = &m.path;
            if p.is_ident("rename") {
                if let Some(name) = de_name(&m)? {
                    f.rename = Some(name);
                }
            } else if p.is_ident("alias") {
                f.aliases.push(lit_str(&m)?.value());
            } else if p.is_ident("default") {
                f.default = Some(match m.input.peek(syn::Token![=]) {
                    true => Some(lit_str(&m)?.parse::<Path>()?),
                    false => None,
                });
            } else if p.is_ident("skip") || p.is_ident("skip_deserializing") {
                f.skip = true;
            } else if p.is_ident("flatten") {
                f.flatten = true;
            } else if p.is_ident("deserialize_with") {
                f.deserialize_with = Some(lit_str(&m)?.parse()?);
            } else if p.is_ident("with") {
                let mut module: Path = lit_str(&m)?.parse()?;
                module.segments.push(parse_quote!(deserialize));
                f.deserialize_with = Some(module);
            } else if p.is_ident("other") {
                f.other = true;
            } else if p.is_ident("skip_serializing")
                || p.is_ident("skip_serializing_if")
                || p.is_ident("serialize_with")
                || p.is_ident("borrow")
                || p.is_ident("bound")
                || p.is_ident("getter")
            {
                skip_value(&m)?;
            } else {
                return Err(m.error("FromJson: unsupported #[serde] field or variant attribute"));
            }
            Ok(())
        })?;
    }
    Ok(f)
}

/// serde's `RenameRule::apply_to_variant` (PascalCase input) and `apply_to_field` (snake_case input).
fn renamed(name: &str, rule: Option<&str>, variant: bool) -> Result<String, String> {
    let Some(rule) = rule else {
        return Ok(name.to_owned());
    };
    let snake = || {
        let mut out = String::new();
        for (i, ch) in name.char_indices() {
            if i > 0 && ch.is_uppercase() {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        }
        out
    };
    let pascal = || {
        let mut out = String::new();
        let mut upper = true;
        for ch in name.chars() {
            if ch == '_' {
                upper = true;
            } else if upper {
                out.push(ch.to_ascii_uppercase());
                upper = false;
            } else {
                out.push(ch);
            }
        }
        out
    };
    let lower_first = |s: String| {
        let mut c = s.chars();
        c.next()
            .map(|f| f.to_ascii_lowercase().to_string() + c.as_str())
            .unwrap_or_default()
    };
    Ok(match (rule, variant) {
        ("PascalCase", true) => name.to_owned(),
        ("lowercase", true) => name.to_ascii_lowercase(),
        ("UPPERCASE", true) => name.to_ascii_uppercase(),
        ("camelCase", true) => lower_first(name.to_owned()),
        ("snake_case", true) => snake(),
        ("SCREAMING_SNAKE_CASE", true) => snake().to_ascii_uppercase(),
        ("kebab-case", true) => snake().replace('_', "-"),
        ("SCREAMING-KEBAB-CASE", true) => snake().to_ascii_uppercase().replace('_', "-"),
        ("lowercase" | "snake_case", false) => name.to_owned(),
        ("UPPERCASE" | "SCREAMING_SNAKE_CASE", false) => name.to_ascii_uppercase(),
        ("PascalCase", false) => pascal(),
        ("camelCase", false) => lower_first(pascal()),
        ("kebab-case", false) => name.replace('_', "-"),
        ("SCREAMING-KEBAB-CASE", false) => name.to_ascii_uppercase().replace('_', "-"),
        (other, _) => return Err(format!("FromJson: unsupported rename_all = \"{other}\"")),
    })
}

fn ident_name(ident: &Ident) -> String {
    let s = ident.to_string();
    s.strip_prefix("r#").map(str::to_owned).unwrap_or(s)
}

const P: &str = "::inference_json::__private";

fn private() -> Tokens {
    P.parse().unwrap()
}

/// How a struct's (or struct variant's) named fields are read from object `o`.
struct NamedFields {
    inits: Vec<Tokens>,
    keys: Vec<String>,
}

fn named_fields(
    fields: &syn::FieldsNamed,
    c: &Container,
    defaults: Option<&Ident>,
) -> syn::Result<NamedFields> {
    let pv = private();
    let mut inits = Vec::new();
    let mut keys = Vec::new();
    for f in &fields.named {
        let a = member_attrs(&f.attrs)?;
        let ident = f.ident.as_ref().unwrap();
        let ty = &f.ty;
        let key = match &a.rename {
            Some(r) => r.clone(),
            None => renamed(&ident_name(ident), c.rename_all.as_deref(), false)
                .map_err(|e| syn::Error::new_spanned(ident, e))?,
        };
        let aliases = &a.aliases;
        let default_value = match (&a.default, defaults) {
            (Some(Some(p)), _) => Some(quote!(#p())),
            (Some(None), _) => Some(quote!(<#ty as ::core::default::Default>::default())),
            (None, Some(d)) => Some(quote!(#d.#ident)),
            (None, None) => None,
        };
        let init = if a.skip {
            default_value.unwrap_or_else(|| quote!(<#ty as ::core::default::Default>::default()))
        } else if a.flatten {
            if c.deny_unknown_fields {
                return Err(syn::Error::new_spanned(
                    ident,
                    "FromJson: `flatten` with `deny_unknown_fields`",
                ));
            }
            quote!(<#ty as ::inference_json::FromJson>::from_json(v)?)
        } else {
            keys.push(key.clone());
            keys.extend(aliases.iter().cloned());
            if let Some(with) = &a.deserialize_with {
                let default = match &default_value {
                    Some(d) => quote!(::core::option::Option::Some(#d)),
                    None => quote!(::core::option::Option::None),
                };
                quote!(#pv::field_with::<#ty>(o, #key, &[#(#aliases),*], #default, |v| #with(v))?)
            } else {
                match default_value {
                    Some(d) => quote!(#pv::field_or::<#ty>(o, #key, &[#(#aliases),*], #d)?),
                    None => quote!(#pv::field::<#ty>(o, #key, &[#(#aliases),*])?),
                }
            }
        };
        inits.push(quote!(#ident: #init));
    }
    Ok(NamedFields { inits, keys })
}

/// A tuple struct's or variant's fields from `value`, built with `ctor` (`Self` or `Self::Variant`).
fn tuple_fields(
    fields: &syn::FieldsUnnamed,
    expected: &str,
    value: &Tokens,
    ctor: Tokens,
) -> Tokens {
    let pv = private();
    let n = fields.unnamed.len();
    if n == 1 {
        let ty = &fields.unnamed[0].ty;
        return quote!(#ctor(<#ty as ::inference_json::FromJson>::from_json(#value)?));
    }
    let items = fields.unnamed.iter().enumerate().map(|(i, f)| {
        let ty = &f.ty;
        quote!(#pv::item::<#ty>(a, #i)?)
    });
    quote!({
        let a = #pv::tuple(#value, #n, #expected)?;
        #ctor(#(#items),*)
    })
}

fn expand_struct(s: &syn::DataStruct, c: &Container, label: &str) -> syn::Result<Tokens> {
    let pv = private();
    let expected = c
        .expecting
        .clone()
        .unwrap_or_else(|| format!("struct {label}"));
    match &s.fields {
        Fields::Named(named) if c.transparent => {
            let f = named
                .named
                .iter()
                .find(|f| !member_attrs(&f.attrs).map(|a| a.skip).unwrap_or(false))
                .unwrap();
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            let rest = named.named.iter().filter(|g| g.ident != f.ident).map(|g| {
                let i = g.ident.as_ref().unwrap();
                quote!(#i: ::core::default::Default::default())
            });
            Ok(quote!(::core::result::Result::Ok(Self {
                #ident: <#ty as ::inference_json::FromJson>::from_json(v)?, #(#rest),*
            })))
        }
        Fields::Named(named) => {
            let d = format_ident!("__defaults");
            let defaults = c.default.as_ref().map(|p| match p {
                Some(p) => quote!(let #d: Self = #p();),
                None => quote!(let #d = <Self as ::core::default::Default>::default();),
            });
            let NamedFields { inits, keys } =
                named_fields(named, c, c.default.as_ref().map(|_| &d))?;
            let deny = c
                .deny_unknown_fields
                .then(|| quote!(#pv::deny_unknown(o, &[#(#keys),*])?;));
            Ok(quote! {
                let o = #pv::object(v, #expected)?;
                #deny
                #defaults
                ::core::result::Result::Ok(Self { #(#inits),* })
            })
        }
        Fields::Unnamed(u) => {
            let body = tuple_fields(u, &expected, &quote!(v), quote!(Self));
            Ok(quote!(::core::result::Result::Ok(#body)))
        }
        Fields::Unit => Ok(quote! {
            <() as ::inference_json::FromJson>::from_json(v)?;
            ::core::result::Result::Ok(Self)
        }),
    }
}

/// The expression that builds variant `vi` from `value` (its payload, or the whole object when internally tagged).
fn variant_body(
    var: &syn::Variant,
    c: &Container,
    label: &str,
    value: &Tokens,
) -> syn::Result<Tokens> {
    let pv = private();
    let vi = &var.ident;
    let expected = format!("{label}::{}", ident_name(vi));
    Ok(match &var.fields {
        Fields::Unit => quote!(Self::#vi),
        Fields::Unnamed(u) => tuple_fields(u, &expected, value, quote!(Self::#vi)),
        Fields::Named(named) => {
            // variant fields follow rename_all_fields, which nothing here sets, not the enum's rename_all
            let fc = Container {
                deny_unknown_fields: c.deny_unknown_fields,
                ..Container::default()
            };
            let NamedFields { inits, mut keys } = named_fields(named, &fc, None)?;
            if let Some(tag) = &c.tag {
                keys.push(tag.clone());
            }
            let deny = c
                .deny_unknown_fields
                .then(|| quote!(#pv::deny_unknown(o, &[#(#keys),*])?;));
            let struct_expected = format!("struct variant {expected}");
            quote!({
                let v = #value;
                let o = #pv::object(v, #struct_expected)?;
                #deny
                Self::#vi { #(#inits),* }
            })
        }
    })
}

fn expand_enum(e: &syn::DataEnum, c: &Container, label: &str) -> syn::Result<Tokens> {
    let pv = private();
    let mut names = Vec::new();
    let mut variants = Vec::new();
    let mut other = None;
    for var in &e.variants {
        let a = member_attrs(&var.attrs)?;
        if a.skip {
            continue;
        }
        if a.other {
            other = Some(var.ident.clone());
            continue;
        }
        let name = match &a.rename {
            Some(r) => r.clone(),
            None => renamed(&ident_name(&var.ident), c.rename_all.as_deref(), true)
                .map_err(|e| syn::Error::new_spanned(&var.ident, e))?,
        };
        let mut all = vec![name.clone()];
        all.extend(a.aliases.iter().cloned());
        names.push(name);
        variants.push((var, all));
    }
    let expected = c
        .expecting
        .clone()
        .unwrap_or_else(|| format!("enum {label}"));
    let unknown = match &other {
        Some(o) => quote!(_ => Self::#o),
        None => quote!(other => return ::core::result::Result::Err(
            ::inference_json::JsonError::unknown_variant(other, &[#(#names),*])
        )),
    };

    if c.untagged {
        let mut tries = Vec::new();
        for (var, _) in &variants {
            let body = match &var.fields {
                Fields::Unit => {
                    let vi = &var.ident;
                    quote!(if v.is_null() { return ::core::result::Result::Ok(Self::#vi); })
                }
                _ => {
                    let build = variant_body(var, c, label, &quote!(v))?;
                    quote! {
                        let attempt = (|| -> ::core::result::Result<Self, ::inference_json::JsonError> {
                            ::core::result::Result::Ok(#build)
                        })();
                        if let ::core::result::Result::Ok(x) = attempt {
                            return ::core::result::Result::Ok(x);
                        }
                    }
                }
            };
            tries.push(body);
        }
        let err = c
            .expecting
            .clone()
            .map(|m| quote!(::inference_json::JsonError::new(#m)))
            .unwrap_or_else(|| quote!(::inference_json::JsonError::untagged(#label)));
        return Ok(quote! {
            #(#tries)*
            ::core::result::Result::Err(#err)
        });
    }

    if let Some(tag) = &c.tag {
        let mut arms = Vec::new();
        for (var, all) in &variants {
            let body = match (&var.fields, &c.content) {
                (Fields::Unit, _) => variant_body(var, c, label, &quote!(v))?,
                (_, Some(content)) => {
                    let name = &all[0];
                    let payload = quote!(&{
                        let o = #pv::object(v, #expected)?;
                        o.get(#content).cloned().ok_or_else(|| ::inference_json::JsonError::missing_field(#content))?
                    });
                    let build = variant_body(var, c, label, &payload)?;
                    quote!((|| -> ::core::result::Result<Self, ::inference_json::JsonError> {
                        ::core::result::Result::Ok(#build)
                    })().map_err(|e| e.at(#name))?)
                }
                (Fields::Named(_), None) => variant_body(var, c, label, &quote!(v))?,
                (Fields::Unnamed(_), None) => {
                    variant_body(var, c, label, &quote!(&#pv::without_tag(v, #tag)))?
                }
            };
            arms.push(quote!(#(#all)|* => #body));
        }
        return Ok(quote! {
            ::core::result::Result::Ok(match #pv::internal(v, #tag, #expected)? {
                #(#arms,)*
                #unknown,
            })
        });
    }

    let mut arms = Vec::new();
    for (var, all) in &variants {
        let name = &all[0];
        let body = match &var.fields {
            Fields::Unit => {
                let vi = &var.ident;
                quote!({ #pv::unit_payload(payload, #name)?; Self::#vi })
            }
            _ => {
                let build = variant_body(var, c, label, &quote!(payload))?;
                quote!({
                    let payload = #pv::payload(payload, #name)?;
                    (|| -> ::core::result::Result<Self, ::inference_json::JsonError> {
                        ::core::result::Result::Ok(#build)
                    })().map_err(|e| e.at(#name))?
                })
            }
        };
        arms.push(quote!(#(#all)|* => #body));
    }
    Ok(quote! {
        let (name, payload) = #pv::external(v, #expected)?;
        ::core::result::Result::Ok(match name {
            #(#arms,)*
            #unknown,
        })
    })
}

fn expand(input: &DeriveInput) -> syn::Result<Tokens> {
    let c = container_attrs(&input.attrs)?;
    let name = &input.ident;
    let label = name.to_string();
    let pv = private();
    let body = if let Some(from) = &c.from {
        quote!(::core::result::Result::Ok(<Self as ::core::convert::From<#from>>::from(
            <#from as ::inference_json::FromJson>::from_json(v)?
        )))
    } else if let Some(from) = &c.try_from {
        quote!(<Self as ::core::convert::TryFrom<#from>>::try_from(
            <#from as ::inference_json::FromJson>::from_json(v)?
        ).map_err(|e| ::inference_json::JsonError::new(::std::string::ToString::to_string(&e))))
    } else {
        match &input.data {
            Data::Struct(s) => expand_struct(s, &c, &label)?,
            Data::Enum(e) => expand_enum(e, &c, &label)?,
            Data::Union(_) => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "FromJson: unions are not supported",
                ));
            }
        }
    };

    let mut generics = input.generics.clone();
    for p in generics.type_params_mut() {
        p.bounds.push(parse_quote!(::inference_json::FromJson));
    }
    let (impl_g, ty_g, where_g) = generics.split_for_impl();
    let mut de_generics = generics.clone();
    de_generics.params.insert(0, parse_quote!('__de));
    let (de_impl_g, _, _) = de_generics.split_for_impl();
    Ok(quote! {
        #[automatically_derived]
        impl #impl_g ::inference_json::FromJson for #name #ty_g #where_g {
            fn from_json(v: &#pv::Value) -> ::core::result::Result<Self, ::inference_json::JsonError> {
                #body
            }
        }

        #[automatically_derived]
        impl #de_impl_g #pv::serde::Deserialize<'__de> for #name #ty_g #where_g {
            fn deserialize<__D: #pv::serde::Deserializer<'__de>>(d: __D) -> ::core::result::Result<Self, __D::Error> {
                #pv::deserialize(d)
            }
        }
    })
}
