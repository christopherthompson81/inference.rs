use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use cudarc;
use cudarc::driver::CudaFunction;
use float8::F8E4M3;
use half::{bf16, f16};
pub use inference_tensor_kernels as kernels;
use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

struct RngGuard<'a>(std::sync::MutexGuard<'a, Option<CudaRng>>);

impl std::ops::Deref for RngGuard<'_> {
    type Target = cudarc::curand::CudaRng;

    fn deref(&self) -> &cudarc::curand::CudaRng {
        &self
            .0
            .as_ref()
            .expect("CudaDevice::rng fills the generator before returning")
            .0
    }
}

const CUDA_GRAPH_HTOD_CACHE_MAX_BYTES: usize = 4096;

/// The archs this build's kernels were compiled for, as cuobjdump names them (`86`, `90a`).
pub fn built_archs() -> &'static [&'static str] {
    kernels::ARCHS
}

/// The built arch whose SASS a device of compute capability `cc` (86 for 8.6) runs, if any.
pub fn kernel_arch(cc: usize) -> Option<usize> {
    kernel_arch_among(kernels::ARCHS, cc)
}

/// The highest of `archs` a device of compute capability `cc` runs, so code under a higher `__CUDA_ARCH__` is absent.
pub fn kernel_arch_among(archs: &[&str], cc: usize) -> Option<usize> {
    archs
        .iter()
        .filter(|arch| crate::kernel_library::runs_on(arch, cc))
        .filter_map(|arch| arch.trim_end_matches(['a', 'f']).parse::<usize>().ok())
        .max()
}

/// The number of CUDA devices the driver shows, or an error when libcuda is missing.
pub fn device_count() -> Result<usize> {
    use cudarc::driver::result;
    require_lib(
        unsafe { cudarc::driver::sys::is_culib_present() },
        "libcuda",
    )?;
    result::init().w()?;
    Ok(result::device::get_count().w()? as usize)
}

/// Compute capability of the device at `ordinal` as `major * 10 + minor`, without creating a context.
pub fn device_compute_cap(ordinal: usize) -> Result<usize> {
    use cudarc::driver::{result, sys::CUdevice_attribute};
    require_lib(
        unsafe { cudarc::driver::sys::is_culib_present() },
        "libcuda",
    )?;
    result::init().w()?;
    let device = result::device::get(ordinal as i32).w()?;
    // SAFETY: device comes from device::get
    let attribute = |attribute| unsafe { result::device::get_attribute(device, attribute) }.w();
    let major = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    Ok((major * 10 + minor) as usize)
}

// cudarc panics when it cannot load a library; this turns a host without one into an error instead.
fn require_lib(present: bool, lib: &str) -> Result<()> {
    if !present {
        crate::bail!("CUDA needs {lib}, which was not found");
    }
    Ok(())
}

type CudaGraphHtodCacheKey = (DeviceId, TypeId, Vec<u8>);
type CudaGraphHtodCache = HashMap<CudaGraphHtodCacheKey, Box<dyn Any>>;

thread_local! {
    static CUDA_GRAPH_HTOD_CACHE: RefCell<CudaGraphHtodCache> =
        RefCell::new(HashMap::new());
    static CUDA_GRAPH_HTOD_CACHE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[must_use]
pub struct CudaGraphHtodCacheGuard;

impl Drop for CudaGraphHtodCacheGuard {
    fn drop(&mut self) {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| {
            debug_assert!(depth.get() > 0);
            depth.set(depth.get().saturating_sub(1));
        });
    }
}

pub struct ModuleStore {
    mdls: [Option<LoadedModule>; kernels::ALL_IDS.len()],
}

// cuModuleGetFunction takes a driver lock another thread's pageable copy holds throughout, so each is looked up once
struct LoadedModule {
    module: Arc<cudarc::driver::CudaModule>,
    funcs: HashMap<String, CudaFunction>,
}

impl LoadedModule {
    fn new(module: Arc<cudarc::driver::CudaModule>) -> Self {
        Self {
            module,
            funcs: HashMap::new(),
        }
    }

    fn function(&mut self, fn_name: &str) -> Result<CudaFunction> {
        if let Some(func) = self.funcs.get(fn_name) {
            return Ok(func.clone());
        }
        let func = self.module.load_function(fn_name).w()?;
        self.funcs.insert(fn_name.to_string(), func.clone());
        Ok(func)
    }
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, LoadedModule>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    // Created on first use, so a device that never draws random numbers never loads libcurand.
    curand: Arc<Mutex<Option<CudaRng>>>,
    seed_value: Arc<RwLock<u64>>,
    compute_cap: usize,
    sm_count: usize,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        unsafe { self.stream.alloc::<T>(len).w() }
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn enable_cuda_graph_htod_cache(&self) -> CudaGraphHtodCacheGuard {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| depth.set(depth.get() + 1));
        CudaGraphHtodCacheGuard
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        if cuda_graph_htod_cache_enabled() && !src.is_empty() {
            return self.memcpy_htod_capture_cached(src, dst);
        }
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.check_capture_copy("clone_dtoh")?;
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::HostSlice<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.check_capture_copy("memcpy_dtoh")?;
        self.stream.memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if cuda_graph_htod_cache_enabled() && !src.is_empty() {
            return self.clone_htod_capture_cached(src);
        }
        self.stream.clone_htod(src).w()
    }

    fn check_capture_copy(&self, op: &str) -> Result<()> {
        if cuda_graph_htod_cache_enabled() {
            let is_capturing = self.cuda_graph_capture_active();
            if is_capturing {
                crate::bail!(
                    "{op} during CUDA graph capture: host/device copies are not \
                     permitted while capturing (use param cache for kernel parameters)"
                );
            }
        }
        Ok(())
    }

    fn memcpy_htod_capture_cached<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        assert!(dst.len() >= src.len());
        if let Some(cached) = self.cached_htod_slice(src)? {
            return self.stream.memcpy_dtod(&cached, dst).w();
        }
        self.stream.memcpy_htod(src, dst).w()
    }

    fn clone_htod_capture_cached<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if let Some(cached) = self.cached_htod_slice(src)? {
            return Ok(cached);
        }
        self.stream.clone_htod(src).w()
    }

    fn cached_htod_slice<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<Option<cudarc::driver::CudaSlice<T>>> {
        let (src_slice, _sync) = unsafe { src.stream_synced_slice(&self.stream) };
        let byte_len = std::mem::size_of_val(src_slice);
        if byte_len > CUDA_GRAPH_HTOD_CACHE_MAX_BYTES {
            if self.cuda_graph_capture_active() {
                crate::bail!("CUDA graph capture cannot upload uncached host data");
            }
            return Ok(None);
        }

        let bytes =
            unsafe { std::slice::from_raw_parts(src_slice.as_ptr().cast::<u8>(), byte_len) }
                .to_vec();
        let key = (self.id, TypeId::of::<T>(), bytes);
        if let Some(cached) = CUDA_GRAPH_HTOD_CACHE.with(|cache| {
            cache.borrow().get(&key).and_then(|cached| {
                cached
                    .downcast_ref::<cudarc::driver::CudaSlice<T>>()
                    .cloned()
            })
        }) {
            return Ok(Some(cached));
        }
        if self.cuda_graph_capture_active() {
            crate::bail!("CUDA graph capture missing cached host data");
        }

        let cached = self.stream.clone_htod(src).w()?;
        CUDA_GRAPH_HTOD_CACHE.with(|cache| {
            cache.borrow_mut().insert(key, Box::new(cached.clone()));
        });
        Ok(Some(cached))
    }

    fn cuda_graph_capture_active(&self) -> bool {
        matches!(
            self.stream.capture_status(),
            Ok(status) if status != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        )
    }
}

pub(crate) fn cuda_graph_htod_cache_enabled() -> bool {
    CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| depth.get() > 0)
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// The same device on a new stream of its context, for copies that overlap the original stream's kernels. Tensors
    /// made through it are allocated and freed on that stream, so the caller orders the two streams.
    pub fn fork_stream(&self) -> Result<Self> {
        Ok(Self {
            stream: self.stream.fork().map_err(crate::Error::wrap)?,
            ..self.clone()
        })
    }

    /// An event after the stream's queued work, which a waiting thread sleeps on: the context's default wait spins.
    pub fn sleeping_event(&self) -> Result<cudarc::driver::CudaEvent> {
        self.stream
            .record_event(Some(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
            ))
            .map_err(crate::Error::wrap)
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        unsafe { self.context.disable_event_tracking() }
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Compute capability as `major * 10 + minor` (86 for 8.6), queried once when the device is created.
    pub fn compute_cap(&self) -> usize {
        self.compute_cap
    }

    pub fn compute_major(&self) -> usize {
        self.compute_cap / 10
    }

    pub fn sm_count(&self) -> usize {
        self.sm_count
    }

    /// Like `get_or_load_custom_func`, from a compiled cubin or fatbin, so loading needs no JIT.
    pub fn get_or_load_custom_image(
        &self,
        fn_name: &str,
        module_name: &str,
        image: &[u8],
    ) -> Result<CudaFunc> {
        self.get_or_load_custom(fn_name, module_name, || {
            cudarc::nvrtc::Ptx::from_binary(image.to_vec())
        })
    }

    fn get_or_load_custom(
        &self,
        fn_name: &str,
        module_name: &str,
        module: impl FnOnce() -> cudarc::nvrtc::Ptx,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(func) = ms.get(module_name).and_then(|m| m.funcs.get(fn_name)) {
            return Ok(CudaFunc {
                func: func.clone(),
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let loaded = match ms.get_mut(module_name) {
            Some(loaded) => loaded,
            None => {
                let cuda_module = self.context.load_module(module()).w()?;
                ms.entry(module_name.to_string())
                    .or_insert(LoadedModule::new(cuda_module))
            }
        };
        let func = loaded.function(fn_name)?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(func) = ms.mdls[mdl.index()]
            .as_ref()
            .and_then(|m| m.funcs.get(fn_name))
        {
            return Ok(CudaFunc {
                func: func.clone(),
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let loaded = match &mut ms.mdls[mdl.index()] {
            Some(loaded) => loaded,
            slot @ None => {
                let image = cudarc::nvrtc::Ptx::from_binary(mdl.image().to_vec());
                let cuda_module = self.context.load_module(image).w()?;
                slot.insert(LoadedModule::new(cuda_module))
            }
        };
        let func = loaded.function(fn_name)?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }
}

impl CudaDevice {
    fn rng(&self) -> Result<RngGuard<'_>> {
        let mut rng = self.curand.lock().unwrap();
        if rng.is_none() {
            require_lib(
                unsafe { cudarc::curand::sys::is_culib_present() },
                "libcurand",
            )?;
            let seed = *self.seed_value.read().unwrap();
            *rng = Some(CudaRng(
                cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?,
            ));
        }
        Ok(RngGuard(rng))
    }

    fn from_context_and_stream(
        context: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        use cudarc::driver::sys::CUdevice_attribute;
        let attribute = |attribute| context.attribute(attribute).w().map(|value| value as usize);
        let major = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let compute_cap = major * 10 + minor;
        // no PTX ships, so a device outside the built arch families has no kernels to run at all
        if kernel_arch(compute_cap).is_none() {
            crate::bail!(
                "CUDA device {} has compute capability {major}.{minor}, but this build carries kernels for sm_{} only; \
                 rebuild with CUDA_COMPUTE_CAP listing {compute_cap}",
                context.ordinal(),
                kernels::ARCHS.join(", sm_")
            )
        }
        let sm_count = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        require_lib(
            unsafe { cudarc::cublas::sys::is_culib_present() },
            "libcublas",
        )?;
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            compute_cap,
            sm_count,
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(None)),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        require_lib(
            unsafe { cudarc::driver::sys::is_culib_present() },
            "libcuda",
        )?;
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.per_thread_stream();
        Self::from_context_and_stream(context, stream)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        *self.seed_value.write().unwrap() = seed;
        *curand = None;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                );
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.rng()?;
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.rng()?;
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        unsafe {
            let elem_count = shape.elem_count();
            let slice = match dtype {
                DType::U8 => {
                    let data = self.alloc::<u8>(elem_count)?;
                    CudaStorageSlice::U8(data)
                }
                DType::U32 => {
                    let data = self.alloc::<u32>(elem_count)?;
                    CudaStorageSlice::U32(data)
                }
                DType::I16 => {
                    let data = self.alloc::<i16>(elem_count)?;
                    CudaStorageSlice::I16(data)
                }
                DType::I32 => {
                    let data = self.alloc::<i32>(elem_count)?;
                    CudaStorageSlice::I32(data)
                }
                DType::I64 => {
                    let data = self.alloc::<i64>(elem_count)?;
                    CudaStorageSlice::I64(data)
                }
                DType::BF16 => {
                    let data = self.alloc::<bf16>(elem_count)?;
                    CudaStorageSlice::BF16(data)
                }
                DType::F16 => {
                    let data = self.alloc::<f16>(elem_count)?;
                    CudaStorageSlice::F16(data)
                }
                DType::F32 => {
                    let data = self.alloc::<f32>(elem_count)?;
                    CudaStorageSlice::F32(data)
                }
                DType::F64 => {
                    let data = self.alloc::<f64>(elem_count)?;
                    CudaStorageSlice::F64(data)
                }
                DType::F8E4M3 => {
                    let data = self.alloc::<F8E4M3>(elem_count)?;
                    CudaStorageSlice::F8E4M3(data)
                }
                DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                    return Err(CudaError::InternalError(
                        "Dummy types not supported in CUDA backend",
                    )
                    .into());
                }
            };
            Ok(CudaStorage {
                slice,
                device: self.clone(),
            })
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Result, Tensor};

    // enough to come from the memory pool rather than a cached small block
    const ELEMENTS: usize = 1 << 20;

    crate::gpu_cases!(
        cuda_cases,
        [a_tensor_dropped_on_a_thread_without_a_context_frees_cleanly]
    );

    // An engine closed from its embedder's thread drops its tensors where no context is current: their frees bind it
    // rather than fail with CUDA_ERROR_INVALID_CONTEXT, which the context keeps and later calls report.
    fn a_tensor_dropped_on_a_thread_without_a_context_frees_cleanly() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let Device::Cuda(cuda) = &device else {
            unreachable!("a CUDA device");
        };
        let tensor = Tensor::zeros(ELEMENTS, DType::F32, &device)?;
        device.synchronize()?;
        std::thread::spawn(move || drop(tensor))
            .join()
            .expect("the dropping thread panicked");
        cuda.cuda_stream()
            .context()
            .check_err()
            .map_err(crate::Error::wrap)
    }
}
