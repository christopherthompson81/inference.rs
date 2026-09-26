use std::sync::OnceLock;

pub const MODULE: &str = "inference_layout";
pub const DEPTHWISE: &str = "depthwise_conv2d_f32";
pub const IM2COL: &str = "im2col_cols_last_f32";
pub const MASK_TO_BOX: &str = "mask_to_box_f32";
/// Threads per `mask_to_box_f32` block; also sizes its shared reduction arrays.
pub const MASK_TO_BOX_BLOCK: u32 = 256;
pub const MS_DEFORM_ATTN: &str = "ms_deform_attn_f32";

const SRC: &str = r#"
extern "C" __global__ void depthwise_conv2d_f32(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ y, int n, int C, int H, int W, int Ho, int Wo, int K, int S, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ox = i % Wo;
    int oy = (i / Wo) % Ho;
    int plane = i / (Wo * Ho);
    int c = plane % C;
    const float* xp = x + (long)plane * H * W;
    const float* wk = w + c * K * K;
    float acc = b[c];
    int iy0 = oy * S - P;
    int ix0 = ox * S - P;
    for (int ky = 0; ky < K; ++ky) {
        int iy = iy0 + ky;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < K; ++kx) {
            int ix = ix0 + kx;
            if (ix >= 0 && ix < W) acc += xp[iy * W + ix] * wk[ky * K + kx];
        }
    }
    y[i] = acc;
}

// cols[n][c*K*K + ky*K + kx][oy*Wo + ox], plus a trailing ones row per batch so the GEMM adds the bias
extern "C" __global__ void im2col_cols_last_f32(
    const float* __restrict__ x, float* __restrict__ cols, int rows_per_b, int n_rows,
    int C, int H, int W, int Ho, int Wo, int K, int S, int P) {
    int hw = Ho * Wo;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= hw) return;
    int ox = p % Wo;
    int oy = p / Wo;
    for (int row = blockIdx.y; row < n_rows; row += gridDim.y) {
        int n = row / rows_per_b;
        int r = row - n * rows_per_b;
        float v = 1.f;
        if (r != rows_per_b - 1) {
            int kx = r % K;
            int ky = (r / K) % K;
            int c = r / (K * K);
            int iy = oy * S - P + ky;
            int ix = ox * S - P + kx;
            v = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[(((long)n * C + c) * H + iy) * W + ix] : 0.f;
        }
        cols[(long)row * hw + p] = v;
    }
}

// one block of MASK_TO_BOX_BLOCK threads per (batch, query) row: block-reduce the bbox of pixels with logit > 0
extern "C" __global__ void mask_to_box_f32(const float* __restrict__ m, float* __restrict__ out, int H, int W) {
    __shared__ int red[4][MASK_TO_BOX_BLOCK];
    int row = blockIdx.x;
    int n = H * W;
    const float* mr = m + (long)row * n;
    int x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (mr[i] > 0.f) {
            int y = i / W, x = i - y * W;
            x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x); y1 = max(y1, y);
        }
    }
    int t = threadIdx.x;
    red[0][t] = x0; red[1][t] = y0; red[2][t] = x1; red[3][t] = y1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            red[0][t] = min(red[0][t], red[0][t + s]);
            red[1][t] = min(red[1][t], red[1][t + s]);
            red[2][t] = max(red[2][t], red[2][t + s]);
            red[3][t] = max(red[3][t], red[3][t + s]);
        }
        __syncthreads();
    }
    if (t == 0) {
        float* o = out + (long)row * 4;
        if (red[2][0] < 0) { o[0] = o[1] = o[2] = o[3] = 0.f; return; }
        float fx0 = (float)red[0][0] / W, fy0 = (float)red[1][0] / H;
        float fx1 = (float)(red[2][0] + 1) / W, fy1 = (float)(red[3][0] + 1) / H;
        o[0] = (fx0 + fx1) / 2.f; o[1] = (fy0 + fy1) / 2.f; o[2] = fx1 - fx0; o[3] = fy1 - fy0;
    }
}

// one thread per (b, q, head, channel), channel fastest so value rows are read coalesced; up to 4 levels
extern "C" __global__ void ms_deform_attn_f32(
    const float* __restrict__ v, const float* __restrict__ loc, const float* __restrict__ attn,
    float* __restrict__ out, int n, int S, int Q, int H, int D, int L, int P,
    int h0, int h1, int h2, int h3, int w0, int w1, int w2, int w3, int s0, int s1, int s2, int s3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int lh[4] = {h0, h1, h2, h3};
    const int lw[4] = {w0, w1, w2, w3};
    const int ls[4] = {s0, s1, s2, s3};
    int d = i % D;
    int bqh = i / D;
    int hh = bqh % H;
    int bi = bqh / H / Q;
    float acc = 0.f;
    for (int l = 0; l < L; ++l) {
        int Hl = lh[l], Wl = lw[l];
        for (int p = 0; p < P; ++p) {
            long sidx = ((long)bqh * L + l) * P + p;
            float a = attn[sidx];
            float x = loc[sidx * 2] * Wl - 0.5f;
            float y = loc[sidx * 2 + 1] * Hl - 0.5f;
            float x0 = floorf(x), y0 = floorf(y);
            float fx = x - x0, fy = y - y0;
            int xi = (int)x0, yi = (int)y0;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int dx = c & 1, dy = c >> 1;
                int xc = xi + dx, yc = yi + dy;
                if (xc < 0 || yc < 0 || xc >= Wl || yc >= Hl) continue;
                float w = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy);
                long row = ((long)bi * S + ls[l] + yc * Wl + xc) * H + hh;
                acc += a * w * v[row * D + d];
            }
        }
    }
    out[i] = acc;
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

/// All layout kernels, NVRTC-compiled once per process.
pub fn ptx() -> candle_core::Result<&'static str> {
    use candle_core::cuda_backend::cudarc::nvrtc;
    PTX.get_or_init(|| {
        nvrtc::compile_ptx(format!(
            "#define MASK_TO_BOX_BLOCK {MASK_TO_BOX_BLOCK}\n{SRC}"
        ))
        .map(|p| p.to_src())
        .map_err(|e| e.to_string())
    })
    .as_deref()
    .map_err(|e| candle_core::Error::Msg(format!("nvrtc layout kernels: {e}")))
}
