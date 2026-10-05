// The slice of ggml.c and ggml-cuda.cu that fattn links against, so the vendored kernels build without ggml.

#include "common.cuh"
#include "convert.cuh"

#include <cstdarg>

thread_local cudaStream_t inference_fattn_current_stream = nullptr;

namespace {

struct type_traits_t {
    const char * name;
    int64_t      blck_size;
    size_t       type_size;
    bool         quantized;
};

type_traits_t traits(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return {"f32", 1, sizeof(float), false};
        case GGML_TYPE_F16:  return {"f16", 1, sizeof(ggml_fp16_t), false};
        case GGML_TYPE_BF16: return {"bf16", 1, sizeof(ggml_bf16_t), false};
        case GGML_TYPE_I32:  return {"i32", 1, sizeof(int32_t), false};
        case GGML_TYPE_I8:   return {"i8", 1, sizeof(int8_t), false}; // fp8 e4m3 K/V (fattn_kv_src::fp8)
        case GGML_TYPE_Q4_0: return {"q4_0", QK4_0, sizeof(block_q4_0), true};
        case GGML_TYPE_Q4_1: return {"q4_1", QK4_1, sizeof(block_q4_1), true};
        case GGML_TYPE_Q5_0: return {"q5_0", QK5_0, sizeof(block_q5_0), true};
        case GGML_TYPE_Q5_1: return {"q5_1", QK5_1, sizeof(block_q5_1), true};
        case GGML_TYPE_Q8_0: return {"q8_0", QK8_0, sizeof(block_q8_0), true};
        default:             return {"unsupported", 1, 0, false};
    }
}

// Stream-ordered allocations on the stream of the call in flight, so scratch stays valid inside graph capture.
struct stream_pool : ggml_cuda_pool {
    void * alloc(size_t size, size_t * actual_size) override {
        void * ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size, inference_fattn_current_stream));
        *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t) override {
        CUDA_CHECK(cudaFreeAsync(ptr, inference_fattn_current_stream));
    }
};

template <typename src_t>
__global__ void convert_to_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t ne00,
        const int64_t ne01, const int64_t ne0203, const uint3 ne02, const int64_t s01, const int64_t s02,
        const int64_t s03) {
    const int64_t i00 = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i00 >= ne00) {
        return;
    }
    const src_t * x = (const src_t *) vx;
    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t) i0203, ne02);
            const int64_t ix = dm.x*s03 + dm.y*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<half>(x[ix]);
        }
    }
}

template <typename src_t>
void convert_to_f16_nc(const void * vx, half * y, const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne03, const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const dim3 blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1)/CUDA_DEQUANTIZE_BLOCK_SIZE,
        (int) std::min(ne01, (int64_t) 65535), (int) std::min(ne0203, (int64_t) 65535));
    convert_to_f16<src_t><<<blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(
        vx, y, ne00, ne01, ne0203, init_fastdiv_values(ne02), s01, s02, s03);
}

template <typename src_t>
void convert_to_f16_cont(const void * vx, half * y, const int64_t k, cudaStream_t stream) {
    convert_to_f16_nc<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

} // namespace

int64_t ggml_blck_size(enum ggml_type type) { return traits(type).blck_size; }
size_t ggml_type_size(enum ggml_type type) { return traits(type).type_size; }
const char * ggml_type_name(enum ggml_type type) { return traits(type).name; }
bool ggml_is_quantized(enum ggml_type type) { return traits(type).quantized; }
size_t ggml_element_size(const struct ggml_tensor * tensor) { return ggml_type_size(tensor->type); }
int64_t ggml_nelements(const struct ggml_tensor * t) { return t->ne[0]*t->ne[1]*t->ne[2]*t->ne[3]; }
int64_t ggml_nrows(const struct ggml_tensor * t) { return t->ne[1]*t->ne[2]*t->ne[3]; }

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }
    const size_t blck_size = ggml_blck_size(tensor->type);
    size_t nbytes = blck_size == 1 ? ggml_type_size(tensor->type) : tensor->ne[0]*tensor->nb[0]/blck_size;
    for (int i = blck_size == 1 ? 0 : 1; i < GGML_MAX_DIMS; ++i) {
        nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
    }
    return nbytes;
}

bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor) == ggml_nelements(tensor)*ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

void ggml_log_internal(enum ggml_log_level level, const char * format, ...) {
    // a CONT line continues the previous message, so it shares that message's level
    thread_local ggml_log_level last = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) {
        last = level;
    }
    if (last < GGML_LOG_LEVEL_WARN) {
        return;
    }
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "inference-fattn: %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    abort();
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1;
    (void) cudaGetDevice(&id);
    fprintf(stderr, "inference-fattn CUDA error: %s\n  device %d, in %s at %s:%d\n  %s\n", msg, id, func, file, line,
        stmt);
    abort();
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info info = {};
        CUDA_CHECK(cudaGetDeviceCount(&info.physical_device_count));
        info.device_count = std::min(info.physical_device_count, GGML_CUDA_MAX_DEVICES);
        for (int id = 0; id < info.device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & dev = info.devices[id];
            dev.cc = 100*prop.major + 10*prop.minor;
            dev.nsm = prop.multiProcessorCount;
            dev.smpb = prop.sharedMemPerBlock;
            dev.smpbo = prop.sharedMemPerBlockOptin;
            dev.warp_size = prop.warpSize;
            dev.total_vram = prop.totalGlobalMem;
            dev.physical_device = id;
            dev.physical_share_count = 1;
        }
        return info;
    }();
    return info;
}

void ggml_cuda_set_device(int device) {
    int current;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::make_unique<stream_pool>();
}

// entry.cu keeps one context per thread and device for the process lifetime; the streams are candle's.
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return convert_to_f16_cont<float>;
        case GGML_TYPE_BF16: return convert_to_f16_cont<nv_bfloat16>;
        default:             return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return convert_to_f16_nc<float>;
        case GGML_TYPE_BF16: return convert_to_f16_nc<nv_bfloat16>;
        default:             return nullptr;
    }
}
