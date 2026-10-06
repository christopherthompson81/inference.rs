mod images {
    include!(concat!(env!("OUT_DIR"), "/images.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 10] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    image: &'static [u8],
    entries: &'static [&'static str],
    optional_entries: &'static [&'static str],
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    /// Compressed SASS fatbin for each of the build's compute capabilities.
    pub fn image(&self) -> &'static [u8] {
        self.image
    }

    pub fn entries(&self) -> &'static [&'static str] {
        self.entries
    }

    /// Entries a lower listed arch's SASS may lack (kernels under a higher `__CUDA_ARCH__` guard).
    pub fn is_optional(&self, entry: &str) -> bool {
        self.optional_entries.contains(&entry)
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

/// The archs (`86`, `90a`) the modules carry SASS for, lowest first.
pub const ARCHS: &[&str] = images::ARCHS;

// A static, not a const: each codegen unit using a const embeds its own image copy, up to 13 of a module per binary.
macro_rules! mdl {
    ($cst:ident, $entries:ident, $optional:ident, $id:ident) => {
        pub static $cst: Module = Module {
            index: module_index(Id::$id),
            image: images::$cst,
            entries: images::$entries,
            optional_entries: images::$optional,
        };
    };
}

mdl!(AFFINE, AFFINE_ENTRIES, AFFINE_OPTIONAL_ENTRIES, Affine);
mdl!(BINARY, BINARY_ENTRIES, BINARY_OPTIONAL_ENTRIES, Binary);
mdl!(CAST, CAST_ENTRIES, CAST_OPTIONAL_ENTRIES, Cast);
mdl!(CONV, CONV_ENTRIES, CONV_OPTIONAL_ENTRIES, Conv);
mdl!(FILL, FILL_ENTRIES, FILL_OPTIONAL_ENTRIES, Fill);
mdl!(INDEXING, INDEXING_ENTRIES, INDEXING_OPTIONAL_ENTRIES, Indexing);
mdl!(QUANTIZED, QUANTIZED_ENTRIES, QUANTIZED_OPTIONAL_ENTRIES, Quantized);
mdl!(REDUCE, REDUCE_ENTRIES, REDUCE_OPTIONAL_ENTRIES, Reduce);
mdl!(TERNARY, TERNARY_ENTRIES, TERNARY_OPTIONAL_ENTRIES, Ternary);
mdl!(UNARY, UNARY_ENTRIES, UNARY_OPTIONAL_ENTRIES, Unary);

