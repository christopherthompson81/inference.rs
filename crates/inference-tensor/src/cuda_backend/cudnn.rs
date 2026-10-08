use crate::WithDType;
use cudarc;
use cudarc::cudnn::safe::{ConvDescriptor, ConvForward, Cudnn, FilterDescriptor, TensorDescriptor};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::cuda_backend::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

impl From<cudarc::cudnn::CudnnError> for crate::Error {
    fn from(err: cudarc::cudnn::CudnnError) -> Self {
        crate::Error::wrap(err)
    }
}

impl From<cudarc::driver::DriverError> for crate::Error {
    fn from(err: cudarc::driver::DriverError) -> Self {
        crate::Error::wrap(err)
    }
}

// Descriptors and algorithm per (shape, dtypes): planning a conv costs more than running small ones
const MAX_PLANS: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ConvShape {
    x: [i32; 4],
    x_strides: Option<[i32; 4]>,
    w: [i32; 4],
    y: [i32; 4],
    pad: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
}

struct Plan<T, Y> {
    conv: ConvDescriptor<Y>,
    x: TensorDescriptor<T>,
    w: FilterDescriptor<T>,
    y: TensorDescriptor<T>,
    algo: cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t,
    workspace_size: usize,
}

type PlanKey = (crate::cuda_backend::DeviceId, TypeId, TypeId, ConvShape);

thread_local! {
    static PLANS: RefCell<HashMap<PlanKey, Box<dyn Any>>> = HashMap::new().into();
    // One workspace per device, grown to the largest plan used, so varying shapes do not each keep their own
    static WORKSPACES: RefCell<HashMap<crate::cuda_backend::DeviceId, CudaSlice<u8>>> = HashMap::new().into();
}

fn cudnn_handle(dev: &crate::cuda_backend::CudaDevice) -> crate::Result<Arc<Cudnn>> {
    let device_id = dev.id();
    Ok(CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?)
}

fn plan<T, Y>(cudnn: &Arc<Cudnn>, shape: &ConvShape) -> crate::Result<Plan<T, Y>>
where
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
{
    use cudarc::cudnn::sys::{cudnnConvolutionMode_t, cudnnMathType_t, cudnnTensorFormat_t};
    let mut conv = cudnn.create_conv2d::<Y>(
        shape.pad,
        shape.stride,
        shape.dilation,
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    // Default math keeps half inputs off tensor cores and runs F32 as TF32 on Ampere; F32 keeps im2col's full precision
    let math =
        if [TypeId::of::<half::f16>(), TypeId::of::<half::bf16>()].contains(&TypeId::of::<T>()) {
            cudnnMathType_t::CUDNN_TENSOR_OP_MATH
        } else {
            cudnnMathType_t::CUDNN_FMA_MATH
        };
    conv.set_math_type(math)?;
    let x = match shape.x_strides {
        None => cudnn.create_4d_tensor::<T>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, shape.x)?,
        Some(strides) => cudnn.create_4d_tensor_ex::<T>(shape.x, strides)?,
    };
    let w = cudnn.create_4d_filter::<T>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, shape.w)?;
    let y = cudnn.create_4d_tensor::<T>(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, shape.y)?;
    let forward = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let algo = forward.pick_algorithm()?;
    let workspace_size = forward.get_workspace_size(algo)?;
    Ok(Plan {
        conv,
        x,
        w,
        y,
        algo,
        workspace_size,
    })
}

fn launch<T, Y>(
    src: &CudaView<T>,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    shape: ConvShape,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()>
where
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType + 'static,
{
    let cudnn = cudnn_handle(dev)?;
    let key = (dev.id(), TypeId::of::<T>(), TypeId::of::<Y>(), shape);
    PLANS.with(|plans| {
        let mut plans = plans.borrow_mut();
        if !plans.contains_key(&key) {
            if plans.len() >= MAX_PLANS {
                plans.clear();
            }
            plans.insert(key, Box::new(plan::<T, Y>(&cudnn, &shape)?));
        }
        let plan = plans
            .get(&key)
            .and_then(|plan| plan.downcast_ref::<Plan<T, Y>>())
            .expect("plans are keyed by their dtypes");
        WORKSPACES.with(|workspaces| {
            let mut workspaces = workspaces.borrow_mut();
            if plan.workspace_size > 0
                && workspaces
                    .get(&dev.id())
                    .is_none_or(|workspace| workspace.len() < plan.workspace_size)
            {
                // cuDNN reads no workspace it has not written, so it needs no zeroing
                let workspace = unsafe { dev.cuda_stream().alloc::<u8>(plan.workspace_size)? };
                workspaces.insert(dev.id(), workspace);
            }
            let workspace = workspaces
                .get_mut(&dev.id())
                .filter(|_| plan.workspace_size > 0);
            let forward = ConvForward {
                conv: &plan.conv,
                x: &plan.x,
                w: &plan.w,
                y: &plan.y,
            };
            unsafe {
                forward.launch::<CudaSlice<u8>, _, _, _>(
                    plan.algo,
                    workspace,
                    (T::one(), T::zero()),
                    src,
                    filter,
                    dst,
                )?;
            }
            Ok(())
        })
    })
}

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType + 'static,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    let p = params;
    // `src` already starts at the layout's offset
    let x_strides = (!src_l.is_contiguous()).then(|| {
        let s = src_l.stride();
        [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32]
    });
    let shape = ConvShape {
        x: [p.b_size as i32, p.c_in as i32, p.i_h as i32, p.i_w as i32],
        x_strides,
        w: [p.c_out as i32, p.c_in as i32, p.k_h as i32, p.k_w as i32],
        y: [
            p.b_size as i32,
            p.c_out as i32,
            p.out_h() as i32,
            p.out_w() as i32,
        ],
        pad: [p.padding as i32; 2],
        stride: [p.stride as i32; 2],
        dilation: [p.dilation as i32; 2],
    };
    launch::<T, Y>(src, filter, dst, shape, dev)
}

// cuDNN tensors have at least 4 dimensions, so a conv1d is a conv2d over [b, c, l, 1]
pub(crate) fn launch_conv1d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType + 'static,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv1D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    let p = params;
    let x_strides = (!src_l.is_contiguous()).then(|| {
        let s = src_l.stride();
        [s[0] as i32, s[1] as i32, s[2] as i32, 1]
    });
    let shape = ConvShape {
        x: [p.b_size as i32, p.c_in as i32, p.l_in as i32, 1],
        x_strides,
        w: [p.c_out as i32, p.c_in as i32, p.k_size as i32, 1],
        y: [p.b_size as i32, p.c_out as i32, p.l_out() as i32, 1],
        pad: [p.padding as i32, 0],
        stride: [p.stride as i32, 1],
        dilation: [p.dilation as i32, 1],
    };
    launch::<T, Y>(src, filter, dst, shape, dev)
}
