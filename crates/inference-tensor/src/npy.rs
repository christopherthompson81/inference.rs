//! Writing tensors as numpy `.npy` files
//! ([npy-format](https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html)).
use crate::{DType, Device, Error, Result, Shape, Tensor};
use byteorder::{LittleEndian, ReadBytesExt};
use half::{bf16, f16, slice::HalfFloatSliceExt};
use std::fs::File;
use std::io::Write;
use std::path::Path;

const NPY_MAGIC_STRING: &[u8] = b"\x93NUMPY";
#[derive(Debug, PartialEq)]
struct Header {
    descr: DType,
    fortran_order: bool,
    shape: Vec<usize>,
}

impl Header {
    fn to_string(&self) -> Result<String> {
        let fortran_order = if self.fortran_order { "True" } else { "False" };
        let mut shape = self
            .shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let descr = match self.descr {
            DType::BF16 => Err(Error::Npy("bf16 is not supported".into()))?,
            DType::F16 => "f2",
            DType::F32 => "f4",
            DType::F64 => "f8",
            DType::I16 => "i2",
            DType::I32 => "i4",
            DType::I64 => "i8",
            DType::U32 => "u4",
            DType::U8 => "u1",
            DType::F8E4M3 => Err(Error::Npy("f8e4m3 is not supported".into()))?,
            DType::F6E2M3 => Err(Error::Npy("f6e2m3 is not supported".into()))?,
            DType::F6E3M2 => Err(Error::Npy("f6e3m2 is not supported".into()))?,
            DType::F4 => Err(Error::Npy("f4 is not supported".into()))?,
            DType::F8E8M0 => Err(Error::Npy("f8e8m0 is not supported".into()))?,
        };
        if !shape.is_empty() {
            shape.push(',')
        }
        Ok(format!(
            "{{'descr': '<{descr}', 'fortran_order': {fortran_order}, 'shape': ({shape}), }}"
        ))
    }

    // Hacky parser for the npy header, a typical example would be:
    // {'descr': '<f8', 'fortran_order': False, 'shape': (128,), }
}

impl Tensor {
    // TODO: Add the possibility to read directly to a device?
    pub(crate) fn from_reader<R: std::io::Read>(
        shape: Shape,
        dtype: DType,
        reader: &mut R,
    ) -> Result<Self> {
        let elem_count = shape.elem_count();
        match dtype {
            DType::BF16 => {
                let mut data_t = vec![bf16::ZERO; elem_count];
                reader.read_u16_into::<LittleEndian>(data_t.reinterpret_cast_mut())?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::F16 => {
                let mut data_t = vec![f16::ZERO; elem_count];
                reader.read_u16_into::<LittleEndian>(data_t.reinterpret_cast_mut())?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::F32 => {
                let mut data_t = vec![0f32; elem_count];
                reader.read_f32_into::<LittleEndian>(&mut data_t)?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::F64 => {
                let mut data_t = vec![0f64; elem_count];
                reader.read_f64_into::<LittleEndian>(&mut data_t)?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::U8 => {
                let mut data_t = vec![0u8; elem_count];
                reader.read_exact(&mut data_t)?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::U32 => {
                let mut data_t = vec![0u32; elem_count];
                reader.read_u32_into::<LittleEndian>(&mut data_t)?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::I16 => {
                let mut data_t = vec![0i16; elem_count];
                reader.read_i16_into::<LittleEndian>(&mut data_t)?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::I32 => {
                let mut data_t = vec![0i32; elem_count];
                reader.read_i32_into::<LittleEndian>(&mut data_t)?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::I64 => {
                let mut data_t = vec![0i64; elem_count];
                reader.read_i64_into::<LittleEndian>(&mut data_t)?;
                Tensor::from_vec(data_t, shape, &Device::Cpu)
            }
            DType::F8E4M3 => {
                let mut data_t = vec![0u8; elem_count];
                reader.read_exact(&mut data_t)?;
                let data_f8: Vec<float8::F8E4M3> =
                    data_t.into_iter().map(float8::F8E4M3::from_bits).collect();
                Tensor::from_vec(data_f8, shape, &Device::Cpu)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "from_reader").bt())
            }
        }
    }

    /// Reads a npy file and return the stored multi-dimensional array as a tensor.
    fn write<T: Write>(&self, f: &mut T) -> Result<()> {
        f.write_all(NPY_MAGIC_STRING)?;
        f.write_all(&[1u8, 0u8])?;
        let header = Header {
            descr: self.dtype(),
            fortran_order: false,
            shape: self.dims().to_vec(),
        };
        let mut header = header.to_string()?;
        let pad = 16 - (NPY_MAGIC_STRING.len() + 5 + header.len()) % 16;
        for _ in 0..pad % 16 {
            header.push(' ')
        }
        header.push('\n');
        f.write_all(&[(header.len() % 256) as u8, (header.len() / 256) as u8])?;
        f.write_all(header.as_bytes())?;
        self.write_bytes(f)
    }

    /// Writes a multi-dimensional array in the npy format.
    pub fn write_npy<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        let mut f = File::create(path.as_ref())?;
        self.write(&mut f)
    }
}
