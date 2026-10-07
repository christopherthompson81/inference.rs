use crate::op::{BackpropOp, Op};
use crate::tensor::from_storage;
use crate::{bail, CpuStorage, CudaStorage, Layout, MetalStorage, Result, Shape, Storage, Tensor};
use std::sync::Arc;

/// Unary ops that can be defined in user-land.
pub trait CustomOp1 {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(&self, _storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _storage: &MetalStorage,
        _layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait CustomOp2 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait CustomOp3 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _arg3: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

impl Tensor {
    /// Applies a unary custom op without backward support
    pub fn apply_op1_no_bwd<C: CustomOp1>(&self, c: &C) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op1(self.layout(), c)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a binary custom op without backward support
    pub fn apply_op2_no_bwd<C: CustomOp2>(&self, rhs: &Self, c: &C) -> Result<Self> {
        let (storage, shape) =
            self.storage()
                .apply_op2(self.layout(), &rhs.storage(), rhs.layout(), c)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a ternary custom op without backward support
    pub fn apply_op3_no_bwd<C: CustomOp3>(&self, t2: &Self, t3: &Self, c: &C) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op3(
            self.layout(),
            &t2.storage(),
            t2.layout(),
            &t3.storage(),
            t3.layout(),
            c,
        )?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a unary custom op.
    pub fn apply_op1_arc(&self, c: Arc<Box<dyn CustomOp1 + Send + Sync>>) -> Result<Self> {
        let (storage, shape) = self
            .storage()
            .apply_op1(self.layout(), c.as_ref().as_ref())?;
        let op = BackpropOp::new1(self, |s| Op::CustomOp1(s, c.clone()));
        Ok(from_storage(storage, shape, op, false))
    }

    pub fn apply_op1<C: 'static + CustomOp1 + Send + Sync>(&self, c: C) -> Result<Self> {
        self.apply_op1_arc(Arc::new(Box::new(c)))
    }

    /// Applies a binary custom op.
    pub fn apply_op2_arc(
        &self,
        rhs: &Self,
        c: Arc<Box<dyn CustomOp2 + Send + Sync>>,
    ) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op2(
            self.layout(),
            &rhs.storage(),
            rhs.layout(),
            c.as_ref().as_ref(),
        )?;
        let op = BackpropOp::new2(self, rhs, |t1, t2| Op::CustomOp2(t1, t2, c.clone()));
        Ok(from_storage(storage, shape, op, false))
    }

    pub fn apply_op2<C: 'static + CustomOp2 + Send + Sync>(&self, r: &Self, c: C) -> Result<Self> {
        self.apply_op2_arc(r, Arc::new(Box::new(c)))
    }

    /// Applies a ternary custom op.
    pub fn apply_op3_arc(
        &self,
        t2: &Self,
        t3: &Self,
        c: Arc<Box<dyn CustomOp3 + Send + Sync>>,
    ) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op3(
            self.layout(),
            &t2.storage(),
            t2.layout(),
            &t3.storage(),
            t3.layout(),
            c.as_ref().as_ref(),
        )?;
        let op = BackpropOp::new3(self, t2, t3, |t1, t2, t3| {
            Op::CustomOp3(t1, t2, t3, c.clone())
        });
        Ok(from_storage(storage, shape, op, false))
    }

    pub fn apply_op3<C: 'static + CustomOp3 + Send + Sync>(
        &self,
        t2: &Self,
        t3: &Self,
        c: C,
    ) -> Result<Self> {
        self.apply_op3_arc(t2, t3, Arc::new(Box::new(c)))
    }
}

// In place ops.

/// Unary ops that can be defined in user-land.
/// These ops work in place and as such back-prop is unsupported.
pub trait InplaceOp1 {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(&self, _storage: &mut CudaStorage, _layout: &Layout) -> Result<()> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(&self, _storage: &mut MetalStorage, _layout: &Layout) -> Result<()> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }
}

/// In-place ops that can be defined in user-land.
/// These ops work in-place and as such back-propagation is unsupported.
pub trait InplaceOpN<const N: usize> {
    fn name(&self) -> &'static str;

    fn cpu_fwd(
        &self,
        dst: &mut CpuStorage,
        dst_l: &Layout,
        srcs: [(&CpuStorage, &Layout); N],
    ) -> Result<()> {
        let _ = (dst, dst_l, srcs);
        bail!("no cpu implementation for {}", self.name())
    }

    fn cuda_fwd(
        &self,
        dst: &mut CudaStorage,
        dst_l: &Layout,
        srcs: [(&CudaStorage, &Layout); N],
    ) -> Result<()> {
        let _ = (dst, dst_l, srcs);
        bail!("no cuda implementation for {}", self.name())
    }

    fn metal_fwd(
        &self,
        dst: &mut MetalStorage,
        dst_l: &Layout,
        srcs: [(&MetalStorage, &Layout); N],
    ) -> Result<()> {
        let _ = (dst, dst_l, srcs);
        bail!("no metal implementation for {}", self.name())
    }
}

macro_rules! forward_op1 {
    ($fwd:ident, $storage:ty) => {
        fn $fwd(
            &self,
            dst: &mut $storage,
            dl: &Layout,
            _: [(&$storage, &Layout); 0],
        ) -> Result<()> {
            InplaceOp1::$fwd(self, dst, dl)
        }
    };
}

impl<C: InplaceOp1> InplaceOpN<0> for C {
    fn name(&self) -> &'static str {
        InplaceOp1::name(self)
    }

    forward_op1!(cpu_fwd, CpuStorage);
    forward_op1!(cuda_fwd, CudaStorage);
    forward_op1!(metal_fwd, MetalStorage);
}

macro_rules! forward_op2 {
    ($fwd:ident, $storage:ty) => {
        fn $fwd(
            &self,
            dst: &mut $storage,
            dl: &Layout,
            srcs: [(&$storage, &Layout); 1],
        ) -> Result<()> {
            let [(s, sl)] = srcs;
            InplaceOp2::$fwd(self, dst, dl, s, sl)
        }
    };
}

impl<C: InplaceOp2> InplaceOpN<1> for C {
    fn name(&self) -> &'static str {
        InplaceOp2::name(self)
    }

    forward_op2!(cpu_fwd, CpuStorage);
    forward_op2!(cuda_fwd, CudaStorage);
    forward_op2!(metal_fwd, MetalStorage);
}

pub trait InplaceOp2 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout)
        -> Result<()>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        s1: &mut CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<()> {
        _ = (s1, l1, s2, l2);
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        s1: &mut MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<()> {
        _ = (s1, l1, s2, l2);
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }
}

macro_rules! forward_op3 {
    ($fwd:ident, $storage:ty) => {
        fn $fwd(
            &self,
            dst: &mut $storage,
            dl: &Layout,
            srcs: [(&$storage, &Layout); 2],
        ) -> Result<()> {
            let [(s1, l1), (s2, l2)] = srcs;
            InplaceOp3::$fwd(self, dst, dl, s1, l1, s2, l2)
        }
    };
}

impl<C: InplaceOp3> InplaceOpN<2> for C {
    fn name(&self) -> &'static str {
        InplaceOp3::name(self)
    }

    forward_op3!(cpu_fwd, CpuStorage);
    forward_op3!(cuda_fwd, CudaStorage);
    forward_op3!(metal_fwd, MetalStorage);
}

pub trait InplaceOp3 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &mut CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<()> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &mut MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<()> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }
}

impl Tensor {
    /// Applies a custom op in-place for the `self` tensor; sources must not share its storage.
    fn inplace_op<const N: usize, C: InplaceOpN<N>>(&self, srcs: [&Self; N], c: &C) -> Result<()> {
        let name = c.name();

        // Ensure writes cannot collide with themselves
        if self.layout().has_internal_overlap() {
            bail!("{name}: dst has repeated elements (zero-stride). Can not write in-place")
        }
        if let Some(i) = srcs.iter().position(|src| self.same_storage(src)) {
            bail!(
                "src {i} shares storage with dst, and `{name}` does not support aliased operands."
            )
        }

        // Acquire locks in order sorted by `Tensor::storage_key`.
        // Avoids deadlock from cycle of waiting locks.
        let dst_key = self.storage_key();

        let mut order: [usize; N] = std::array::from_fn(|i| i);
        order.sort_unstable_by_key(|&i| srcs[i].storage_key());

        let mut guards: [Option<_>; N] = std::array::from_fn(|_| None);
        let mut dst: Option<_> = None;

        for &i in order.iter() {
            let key = srcs[i].storage_key();
            if key > dst_key && dst.is_none() {
                dst = Some(self.storage_mut());
            }
            // Two sources sharing the same allocation, but distinct from dst.
            // `read_recursive` allows for shared read access
            guards[i] = Some(srcs[i].storage());
        }
        // If `dst` is not yet set we acquire it from `self` now.
        let mut dst = match dst {
            Some(g) => g,
            None => self.storage_mut(),
        };

        let operands: [(&Storage, &Layout); N] = std::array::from_fn(|i| {
            let guard = guards[i].as_ref().expect("every source was locked above");
            (&**guard, srcs[i].layout())
        });

        dst.inplace_op(self.layout(), operands, c)
    }

    /// Applies a unary custom op in place.
    pub fn inplace_op1<C: InplaceOp1>(&self, c: &C) -> Result<()> {
        self.inplace_op([], c)
    }

    /// Applies a binary custom op in place (for the first tensor).
    pub fn inplace_op2<C: InplaceOpN<1>>(&self, rhs: &Self, c: &C) -> Result<()> {
        self.inplace_op([rhs], c)
    }

    /// Applies a ternary custom op in place (for the first tensor).
    pub fn inplace_op3<C: InplaceOpN<2>>(&self, t2: &Self, t3: &Self, c: &C) -> Result<()> {
        self.inplace_op([t2, t3], c)
    }
}
