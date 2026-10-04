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
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    image: &'static [u8],
    entries: &'static [&'static str],
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    /// Compressed SASS fatbin for the build's compute capability.
    pub fn image(&self) -> &'static [u8] {
        self.image
    }

    pub fn entries(&self) -> &'static [&'static str] {
        self.entries
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

// A static, not a const: each codegen unit using a const embeds its own image copy, up to 13 of a module per binary.
macro_rules! mdl {
    ($cst:ident, $entries:ident, $id:ident) => {
        pub static $cst: Module = Module {
            index: module_index(Id::$id),
            image: images::$cst,
            entries: images::$entries,
        };
    };
}

mdl!(AFFINE, AFFINE_ENTRIES, Affine);
mdl!(BINARY, BINARY_ENTRIES, Binary);
mdl!(CAST, CAST_ENTRIES, Cast);
mdl!(CONV, CONV_ENTRIES, Conv);
mdl!(FILL, FILL_ENTRIES, Fill);
mdl!(INDEXING, INDEXING_ENTRIES, Indexing);
mdl!(QUANTIZED, QUANTIZED_ENTRIES, Quantized);
mdl!(REDUCE, REDUCE_ENTRIES, Reduce);
mdl!(SORT, SORT_ENTRIES, Sort);
mdl!(TERNARY, TERNARY_ENTRIES, Ternary);
mdl!(UNARY, UNARY_ENTRIES, Unary);

pub mod ffi;
