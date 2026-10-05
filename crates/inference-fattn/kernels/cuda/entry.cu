// C entry points: wrap raw device pointers in ggml tensor descriptors and run fattn on the caller's stream.

#include "fattn-common.cuh"
#include "fattn.cuh"

extern thread_local cudaStream_t inference_fattn_current_stream;

struct inference_fattn_tensor {
    const void * data; // null when the operand is absent (mask, sinks)
    int32_t      type; // ggml_type
    int64_t      ne[4];
    int64_t      nb[4]; // bytes
};

// Operands in ggml_flash_attn_ext's layout (ggml.h); q is f32 or bf16, dst is [d_v, n_head, n_q, n_seq], contiguous.
struct inference_fattn_args {
    inference_fattn_tensor q, k, v, mask, sinks;
    void *  dst;
    int32_t dst_type; // ggml_type: F32 or BF16
    float   scale;
    float   max_bias;
    float   softcap;
    int32_t device;
    void *  stream; // cudaStream_t
    const fattn_paged_kv * paged; // null: dense K/V; otherwise k/v describe one block's head and row strides
};

namespace {

ggml_tensor descriptor(const inference_fattn_tensor & t) {
    ggml_tensor out = {};
    out.type = (ggml_type) t.type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out.ne[i] = t.ne[i];
        out.nb[i] = (size_t) t.nb[i];
    }
    out.data = const_cast<void *>(t.data);
    return out;
}

struct operands {
    ggml_tensor q, k, v, mask, sinks, dst;

    explicit operands(const inference_fattn_args & a) :
            q(descriptor(a.q)), k(descriptor(a.k)), v(descriptor(a.v)), mask(descriptor(a.mask)),
            sinks(descriptor(a.sinks)), dst{} {
        dst.type = (ggml_type) a.dst_type;
        dst.op = GGML_OP_FLASH_ATTN_EXT;
        dst.ne[0] = v.ne[0];
        dst.ne[1] = q.ne[2];
        dst.ne[2] = q.ne[1];
        dst.ne[3] = q.ne[3];
        dst.nb[0] = ggml_type_size(dst.type);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            dst.nb[i] = dst.nb[i - 1]*dst.ne[i - 1];
        }
        dst.data = a.dst;
        dst.src[0] = &q;
        dst.src[1] = &k;
        dst.src[2] = &v;
        dst.src[3] = a.mask.data ? &mask : nullptr;
        dst.src[4] = a.sinks.data ? &sinks : nullptr;
        const float params[3] = {a.scale, a.max_bias, a.softcap};
        memcpy(dst.op_params, params, sizeof(params));
        memcpy(dst.op_params + FATTN_OP_PARAMS_PAGED, &a.paged, sizeof(a.paged));
    }
};

ggml_backend_cuda_context & context(int device, cudaStream_t stream) {
    thread_local ggml_backend_cuda_context * contexts[GGML_CUDA_MAX_DEVICES] = {};
    if (contexts[device] == nullptr) {
        contexts[device] = new ggml_backend_cuda_context(device);
    }
    ggml_backend_cuda_context & ctx = *contexts[device];
    ctx.streams[device][ctx.curr_stream_no] = stream;
    return ctx;
}

} // namespace

extern "C" bool inference_fattn_supported(const inference_fattn_args * args) {
    operands ops(*args);
    return ggml_cuda_flash_attn_ext_supported(args->device, &ops.dst);
}

extern "C" int inference_fattn_forward(const inference_fattn_args * args) {
    operands ops(*args);
    cudaStream_t stream = (cudaStream_t) args->stream;
    inference_fattn_current_stream = stream;
    ggml_cuda_flash_attn_ext(context(args->device, stream), &ops.dst);
    inference_fattn_current_stream = nullptr;
    return (int) cudaGetLastError();
}
