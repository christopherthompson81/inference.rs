// 3D Gaussian splatting for one pinhole camera, from gsplat 1.5.3 (Apache-2.0, see third_party/README.md): EWA
// projection with the antialiasing compensation, tile intersection with a radix sort, front-to-back rasterization of
// RGB plus z-depth with a per-Gaussian multiplicative mask (MaskGaussian), and their backward passes. Parameters come
// in raw (log scales, opacity logits, degree-0 SH) and gradients go back to them.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

constexpr float kGsAlphaThreshold = 1.f / 255.f;
constexpr float kGsMaxAlpha = 0.999f;
constexpr float kGsMinTransmittance = 1e-4f;
constexpr float kGsExtend = 3.33f;
constexpr float kGsShC0 = 0.28209479177387814f;
constexpr int kGsThreads = 256;
constexpr int kGsChannels = 4;

// column-major like glm: m[column][row]
template <int C, int R> struct GsMat {
  float m[C][R];
  __device__ float *operator[](int c) { return m[c]; }
  __device__ const float *operator[](int c) const { return m[c]; }
};
template <int N> struct GsVec {
  float v[N];
  __device__ float &operator[](int i) { return v[i]; }
  __device__ float operator[](int i) const { return v[i]; }
};
using gmat2 = GsMat<2, 2>;
using gmat3 = GsMat<3, 3>;
using gmat3x2 = GsMat<3, 2>;
using gvec2 = GsVec<2>;
using gvec3 = GsVec<3>;
using gvec4 = GsVec<4>;

template <int C, int R> __device__ GsMat<C, R> gs_zero() {
  GsMat<C, R> z;
#pragma unroll
  for (int c = 0; c < C; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r)
      z[c][r] = 0.f;
  return z;
}
template <int K, int R, int C> __device__ GsMat<C, R> operator*(const GsMat<K, R> &a, const GsMat<C, K> &b) {
  GsMat<C, R> o;
#pragma unroll
  for (int c = 0; c < C; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < K; ++k)
        s += a[k][r] * b[c][k];
      o[c][r] = s;
    }
  return o;
}
template <int C, int R> __device__ GsVec<R> operator*(const GsMat<C, R> &a, const GsVec<C> &x) {
  GsVec<R> o;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c)
      s += a[c][r] * x[c];
    o[r] = s;
  }
  return o;
}
template <int C, int R> __device__ GsMat<C, R> operator+(const GsMat<C, R> &a, const GsMat<C, R> &b) {
  GsMat<C, R> o;
#pragma unroll
  for (int c = 0; c < C; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r)
      o[c][r] = a[c][r] + b[c][r];
  return o;
}
template <int C, int R> __device__ GsMat<C, R> operator*(float s, const GsMat<C, R> &a) {
  GsMat<C, R> o;
#pragma unroll
  for (int c = 0; c < C; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r)
      o[c][r] = s * a[c][r];
  return o;
}
template <int C, int R> __device__ GsMat<R, C> gs_transpose(const GsMat<C, R> &a) {
  GsMat<R, C> o;
#pragma unroll
  for (int c = 0; c < C; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r)
      o[r][c] = a[c][r];
  return o;
}
// glm::outerProduct(c, r): column vector times row vector
template <int R, int C> __device__ GsMat<C, R> gs_outer(const GsVec<R> &c, const GsVec<C> &r) {
  GsMat<C, R> o;
#pragma unroll
  for (int i = 0; i < C; ++i)
#pragma unroll
    for (int j = 0; j < R; ++j)
      o[i][j] = c[j] * r[i];
  return o;
}
__device__ gmat2 gs_inverse(const gmat2 &a) {
  const float inv_det = 1.f / (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
  gmat2 o;
  o[0][0] = a[1][1] * inv_det;
  o[0][1] = -a[0][1] * inv_det;
  o[1][0] = -a[1][0] * inv_det;
  o[1][1] = a[0][0] * inv_det;
  return o;
}

// the world-to-camera rotation and translation of a row-major 4x4
__device__ void gs_viewmat(const float (&vm)[16], gmat3 &R, gvec3 &t) {
#pragma unroll
  for (int c = 0; c < 3; ++c)
#pragma unroll
    for (int r = 0; r < 3; ++r)
      R[c][r] = vm[r * 4 + c];
  t = {vm[3], vm[7], vm[11]};
}

__device__ gmat3 gs_quat_to_rotmat(gvec4 q) {
  const float inv = rsqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float w = q[0] * inv, x = q[1] * inv, y = q[2] * inv, z = q[3] * inv;
  const float x2 = x * x, y2 = y * y, z2 = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y,
              wz = w * z;
  return {{{1.f - 2.f * (y2 + z2), 2.f * (xy + wz), 2.f * (xz - wy)},
           {2.f * (xy - wz), 1.f - 2.f * (x2 + z2), 2.f * (yz + wx)},
           {2.f * (xz + wy), 2.f * (yz - wx), 1.f - 2.f * (x2 + y2)}}};
}

__device__ void gs_quat_to_rotmat_vjp(gvec4 q, const gmat3 &v_R, gvec4 &v_q) {
  const float inv = rsqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float w = q[0] * inv, x = q[1] * inv, y = q[2] * inv, z = q[3] * inv;
  const gvec4 vn = {
      2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) + z * (v_R[0][1] - v_R[1][0])),
      2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) + z * (v_R[0][2] + v_R[2][0]) +
             w * (v_R[1][2] - v_R[2][1])),
      2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) + z * (v_R[1][2] + v_R[2][1]) +
             w * (v_R[2][0] - v_R[0][2])),
      2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) - 2.f * z * (v_R[0][0] + v_R[1][1]) +
             w * (v_R[0][1] - v_R[1][0]))};
  const gvec4 qn = {w, x, y, z};
  const float d = vn[0] * qn[0] + vn[1] * qn[1] + vn[2] * qn[2] + vn[3] * qn[3];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v_q[i] += (vn[i] - d * qn[i]) * inv;
}

__device__ gmat3 gs_diag(gvec3 s) {
  gmat3 S = gs_zero<3, 3>();
  S[0][0] = s[0];
  S[1][1] = s[1];
  S[2][2] = s[2];
  return S;
}

// the perspective Jacobian with gsplat's frustum clamp of the tangent
struct GsPersp {
  float rz, rz2, tx, ty, lim_x_pos, lim_x_neg, lim_y_pos, lim_y_neg;
  gmat3x2 J;
};
__device__ GsPersp gs_persp(gvec3 p, float fx, float fy, float cx, float cy, uint32_t w, uint32_t h) {
  GsPersp o;
  const float tan_fovx = 0.5f * w / fx, tan_fovy = 0.5f * h / fy;
  o.lim_x_pos = (w - cx) / fx + 0.3f * tan_fovx;
  o.lim_x_neg = cx / fx + 0.3f * tan_fovx;
  o.lim_y_pos = (h - cy) / fy + 0.3f * tan_fovy;
  o.lim_y_neg = cy / fy + 0.3f * tan_fovy;
  o.rz = 1.f / p[2];
  o.rz2 = o.rz * o.rz;
  o.tx = p[2] * fminf(o.lim_x_pos, fmaxf(-o.lim_x_neg, p[0] * o.rz));
  o.ty = p[2] * fminf(o.lim_y_pos, fmaxf(-o.lim_y_neg, p[1] * o.rz));
  o.J = {{{fx * o.rz, 0.f}, {0.f, fy * o.rz}, {-fx * o.tx * o.rz2, -fy * o.ty * o.rz2}}};
  return o;
}

struct GsProjectArgs {
  const float *means, *quats, *log_scales, *logit_opacities, *sh0;
  float viewmat[16], fx, fy, cx, cy;
  uint32_t n, width, height;
  float eps2d, near_plane, far_plane, radius_clip;
};

// per Gaussian: radii (x, y), pixel mean, z, conic, the compensation, opacity * compensation, and (rgb, z) colors
__global__ void gs_project_fwd_kernel(GsProjectArgs a, int32_t *radii, float *means2d, float *depths, float *conics,
                                      float *compensations, float *opacities, float *colors) {
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= a.n)
    return;
  radii[2 * g] = 0;
  radii[2 * g + 1] = 0;
  gmat3 R;
  gvec3 t;
  gs_viewmat(a.viewmat, R, t);
  const gvec3 mean = {a.means[3 * g], a.means[3 * g + 1], a.means[3 * g + 2]};
  gvec3 mc = R * mean;
#pragma unroll
  for (int i = 0; i < 3; ++i)
    mc[i] += t[i];
  if (mc[2] < a.near_plane || mc[2] > a.far_plane)
    return;
  const gvec4 q = {a.quats[4 * g], a.quats[4 * g + 1], a.quats[4 * g + 2], a.quats[4 * g + 3]};
  const gvec3 s = {expf(a.log_scales[3 * g]), expf(a.log_scales[3 * g + 1]), expf(a.log_scales[3 * g + 2])};
  const gmat3 M = gs_quat_to_rotmat(q) * gs_diag(s);
  const gmat3 covar = M * gs_transpose(M);
  const gmat3 covar_c = R * covar * gs_transpose(R);
  const float fx = a.fx, cx = a.cx, fy = a.fy, cy = a.cy;
  const GsPersp pj = gs_persp(mc, fx, fy, cx, cy, a.width, a.height);
  gmat2 cov2d = pj.J * covar_c * gs_transpose(pj.J);
  const gvec2 mean2d = {fx * mc[0] * pj.rz + cx, fy * mc[1] * pj.rz + cy};
  const float det_orig = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
  cov2d[0][0] += a.eps2d;
  cov2d[1][1] += a.eps2d;
  const float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
  if (det <= 0.f)
    return;
  const float compensation = sqrtf(fmaxf(0.f, det_orig / det));
  const gmat2 conic = gs_inverse(cov2d);
  const float opacity = compensation / (1.f + expf(-a.logit_opacities[g]));
  if (opacity < kGsAlphaThreshold)
    return;
  const float extend = fminf(kGsExtend, sqrtf(2.f * __logf(opacity / kGsAlphaThreshold)));
  const float rx = ceilf(extend * sqrtf(cov2d[0][0])), ry = ceilf(extend * sqrtf(cov2d[1][1]));
  if (rx <= a.radius_clip && ry <= a.radius_clip)
    return;
  if (mean2d[0] + rx <= 0 || mean2d[0] - rx >= a.width || mean2d[1] + ry <= 0 || mean2d[1] - ry >= a.height)
    return;
  radii[2 * g] = static_cast<int32_t>(rx);
  radii[2 * g + 1] = static_cast<int32_t>(ry);
  means2d[2 * g] = mean2d[0];
  means2d[2 * g + 1] = mean2d[1];
  depths[g] = mc[2];
  conics[3 * g] = conic[0][0];
  conics[3 * g + 1] = conic[0][1];
  conics[3 * g + 2] = conic[1][1];
  compensations[g] = compensation;
  opacities[g] = opacity;
#pragma unroll
  for (int c = 0; c < 3; ++c)
    colors[kGsChannels * g + c] = fmaxf(kGsShC0 * a.sh0[3 * g + c] + 0.5f, 0.f);
  colors[kGsChannels * g + 3] = mc[2];
}

// gradients of the raw parameters from the rasterizer's gradients of means2d, conics, colors and opacities
__global__ void gs_project_bwd_kernel(GsProjectArgs a, const int32_t *radii, const float *conics,
                                      const float *compensations, const float *v_means2d, const float *v_conics,
                                      const float *v_colors, const float *v_opacities, float *v_means, float *v_quats,
                                      float *v_log_scales, float *v_logit_opacities, float *v_sh0) {
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= a.n)
    return;
  for (int i = 0; i < 3; ++i) {
    v_means[3 * g + i] = 0.f;
    v_log_scales[3 * g + i] = 0.f;
    v_sh0[3 * g + i] = 0.f;
  }
  for (int i = 0; i < 4; ++i)
    v_quats[4 * g + i] = 0.f;
  v_logit_opacities[g] = 0.f;
  if (radii[2 * g] <= 0 || radii[2 * g + 1] <= 0)
    return;
  const float *vc = v_colors + kGsChannels * g;
#pragma unroll
  for (int c = 0; c < 3; ++c)
    if (kGsShC0 * a.sh0[3 * g + c] + 0.5f >= 0.f)
      v_sh0[3 * g + c] = kGsShC0 * vc[c];
  // opacity = sigmoid(logit) * compensation
  const float sig = 1.f / (1.f + expf(-a.logit_opacities[g]));
  const float compensation = compensations[g];
  v_logit_opacities[g] = v_opacities[g] * compensation * sig * (1.f - sig);
  const float v_compensation = v_opacities[g] * sig;

  const gmat2 conic = {{{conics[3 * g], conics[3 * g + 1]}, {conics[3 * g + 1], conics[3 * g + 2]}}};
  const gmat2 v_conic = {
      {{v_conics[3 * g], v_conics[3 * g + 1] * .5f}, {v_conics[3 * g + 1] * .5f, v_conics[3 * g + 2]}}};
  // inverse vjp, then add_blur's
  gmat2 v_cov2d = -1.f * (conic * v_conic * conic);
  {
    const float det_conic = conic[0][0] * conic[1][1] - conic[0][1] * conic[1][0];
    const float v_sqr = v_compensation * 0.5f / (compensation + 1e-6f);
    const float one_minus = 1.f - compensation * compensation;
    v_cov2d[0][0] += v_sqr * (one_minus * conic[0][0] - a.eps2d * det_conic);
    v_cov2d[0][1] += v_sqr * (one_minus * conic[0][1]);
    v_cov2d[1][0] += v_sqr * (one_minus * conic[1][0]);
    v_cov2d[1][1] += v_sqr * (one_minus * conic[1][1] - a.eps2d * det_conic);
  }

  gmat3 R;
  gvec3 t;
  gs_viewmat(a.viewmat, R, t);
  const gvec3 mean = {a.means[3 * g], a.means[3 * g + 1], a.means[3 * g + 2]};
  const gvec4 q = {a.quats[4 * g], a.quats[4 * g + 1], a.quats[4 * g + 2], a.quats[4 * g + 3]};
  const gvec3 s = {expf(a.log_scales[3 * g]), expf(a.log_scales[3 * g + 1]), expf(a.log_scales[3 * g + 2])};
  const gmat3 rot = gs_quat_to_rotmat(q);
  const gmat3 M = rot * gs_diag(s);
  const gmat3 covar = M * gs_transpose(M);
  gvec3 mc = R * mean;
#pragma unroll
  for (int i = 0; i < 3; ++i)
    mc[i] += t[i];
  const gmat3 covar_c = R * covar * gs_transpose(R);

  const float fx = a.fx, cx = a.cx, fy = a.fy, cy = a.cy;
  const GsPersp pj = gs_persp(mc, fx, fy, cx, cy, a.width, a.height);
  const float x = mc[0], y = mc[1];
  const gvec2 v_mean2d = {v_means2d[2 * g], v_means2d[2 * g + 1]};
  const gmat3 v_covar_c = gs_transpose(pj.J) * v_cov2d * pj.J;
  gvec3 v_mc = {fx * pj.rz * v_mean2d[0], fy * pj.rz * v_mean2d[1],
                -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * pj.rz2};
  const float rz3 = pj.rz2 * pj.rz;
  const gmat3x2 v_J =
      v_cov2d * pj.J * gs_transpose(covar_c) + gs_transpose(v_cov2d) * pj.J * covar_c;
  if (x * pj.rz <= pj.lim_x_pos && x * pj.rz >= -pj.lim_x_neg)
    v_mc[0] += -fx * pj.rz2 * v_J[2][0];
  else
    v_mc[2] += -fx * rz3 * v_J[2][0] * pj.tx;
  if (y * pj.rz <= pj.lim_y_pos && y * pj.rz >= -pj.lim_y_neg)
    v_mc[1] += -fy * pj.rz2 * v_J[2][1];
  else
    v_mc[2] += -fy * rz3 * v_J[2][1] * pj.ty;
  v_mc[2] += -fx * pj.rz2 * v_J[0][0] - fy * pj.rz2 * v_J[1][1] + 2.f * fx * pj.tx * rz3 * v_J[2][0] +
             2.f * fy * pj.ty * rz3 * v_J[2][1];
  // the depth channel is the camera z
  v_mc[2] += vc[3];

  const gvec3 v_mean = gs_transpose(R) * v_mc;
  const gmat3 v_covar = gs_transpose(R) * v_covar_c * R;
  // covar = M M^T with M = rot * S
  const gmat3 v_M = (v_covar + gs_transpose(v_covar)) * M;
  const gmat3 v_rot = v_M * gs_diag(s);
  gvec4 v_q = {0.f, 0.f, 0.f, 0.f};
  gs_quat_to_rotmat_vjp(q, v_rot, v_q);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    v_means[3 * g + i] = v_mean[i];
    const float v_s = rot[i][0] * v_M[i][0] + rot[i][1] * v_M[i][1] + rot[i][2] * v_M[i][2];
    v_log_scales[3 * g + i] = v_s * s[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v_quats[4 * g + i] = v_q[i];
}

struct GsTiles {
  uint32_t tile_size, tile_width, tile_height, tile_bits;
};

__device__ void gs_tile_rect(const int32_t *radii, const float *means2d, uint32_t g, GsTiles t, uint2 &lo, uint2 &hi) {
  const float trx = radii[2 * g] / static_cast<float>(t.tile_size);
  const float try_ = radii[2 * g + 1] / static_cast<float>(t.tile_size);
  const float tx = means2d[2 * g] / static_cast<float>(t.tile_size);
  const float ty = means2d[2 * g + 1] / static_cast<float>(t.tile_size);
  lo.x = min(max(0, (uint32_t)floorf(tx - trx)), t.tile_width);
  lo.y = min(max(0, (uint32_t)floorf(ty - try_)), t.tile_height);
  hi.x = min(max(0, (uint32_t)ceilf(tx + trx)), t.tile_width);
  hi.y = min(max(0, (uint32_t)ceilf(ty + try_)), t.tile_height);
}

__global__ void gs_tiles_count_kernel(const int32_t *radii, const float *means2d, uint32_t n, GsTiles t,
                                      int64_t *tiles_per_gauss) {
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n)
    return;
  if (radii[2 * g] <= 0 || radii[2 * g + 1] <= 0) {
    tiles_per_gauss[g] = 0;
    return;
  }
  uint2 lo, hi;
  gs_tile_rect(radii, means2d, g, t, lo, hi);
  tiles_per_gauss[g] = static_cast<int64_t>((hi.y - lo.y) * (hi.x - lo.x));
}

// key: tile id above the float bits of the depth (positive, so they sort as integers)
__global__ void gs_tiles_emit_kernel(const int32_t *radii, const float *means2d, const float *depths,
                                     const int64_t *cum_tiles, uint32_t n, GsTiles t, int64_t *keys,
                                     int32_t *values) {
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n || radii[2 * g] <= 0 || radii[2 * g + 1] <= 0)
    return;
  uint2 lo, hi;
  gs_tile_rect(radii, means2d, g, t, lo, hi);
  const int64_t depth_bits = static_cast<uint32_t>(__float_as_int(depths[g]));
  int64_t at = g == 0 ? 0 : cum_tiles[g - 1];
  for (int64_t i = lo.y; i < hi.y; ++i)
    for (int64_t j = lo.x; j < hi.x; ++j) {
      keys[at] = ((i * t.tile_width + j) << 32) | depth_bits;
      values[at] = static_cast<int32_t>(g);
      ++at;
    }
}

// first intersection of each tile; tiles past the last one start at n_isects
__global__ void gs_tile_offsets_kernel(const int64_t *keys, uint32_t n_isects, uint32_t n_tiles, int32_t *offsets) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_isects)
    return;
  const int64_t tile = keys[i] >> 32;
  if (i == 0)
    for (int64_t k = 0; k <= tile; ++k)
      offsets[k] = 0;
  if (i == n_isects - 1)
    for (int64_t k = tile + 1; k < n_tiles; ++k)
      offsets[k] = static_cast<int32_t>(n_isects);
  if (i > 0) {
    const int64_t prev = keys[i - 1] >> 32;
    for (int64_t k = prev + 1; k <= tile; ++k)
      offsets[k] = static_cast<int32_t>(i);
  }
}

struct GsRasterArgs {
  const float *means2d, *conics, *colors, *opacities, *gate;
  const int32_t *offsets, *flatten_ids;
  uint32_t width, height, n_isects;
  GsTiles t;
};

__global__ void gs_raster_fwd_kernel(GsRasterArgs a, float *render, float *alphas, int32_t *last_ids) {
  auto block = cg::this_thread_block();
  const int32_t tile_id = blockIdx.y * a.t.tile_width + blockIdx.x;
  const uint32_t i = blockIdx.y * a.t.tile_size + threadIdx.y;
  const uint32_t j = blockIdx.x * a.t.tile_size + threadIdx.x;
  const float px = (float)j + 0.5f, py = (float)i + 0.5f;
  const int32_t pix = i * a.width + j;
  const bool inside = i < a.height && j < a.width;
  bool done = !inside;
  const int32_t range_start = a.offsets[tile_id];
  const int32_t range_end =
      tile_id == a.t.tile_width * a.t.tile_height - 1 ? a.n_isects : a.offsets[tile_id + 1];
  const uint32_t block_size = block.size();
  const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;
  extern __shared__ int gs_smem[];
  int32_t *id_batch = gs_smem;
  float3 *xy_opacity = reinterpret_cast<float3 *>(&id_batch[block_size]);
  float3 *conic_batch = &xy_opacity[block_size];
  float *gate_batch = reinterpret_cast<float *>(&conic_batch[block_size]);
  float T = 1.f;
  uint32_t cur_idx = 0;
  const uint32_t tr = block.thread_rank();
  float out[kGsChannels] = {0.f};
  for (uint32_t b = 0; b < num_batches; ++b) {
    if (__syncthreads_count(done) >= block_size)
      break;
    const uint32_t batch_start = range_start + block_size * b;
    const uint32_t idx = batch_start + tr;
    if (idx < range_end) {
      const int32_t g = a.flatten_ids[idx];
      id_batch[tr] = g;
      xy_opacity[tr] = {a.means2d[2 * g], a.means2d[2 * g + 1], a.opacities[g]};
      conic_batch[tr] = {a.conics[3 * g], a.conics[3 * g + 1], a.conics[3 * g + 2]};
      gate_batch[tr] = a.gate != nullptr ? a.gate[g] : 1.f;
    }
    block.sync();
    const uint32_t batch_size = min(block_size, range_end - batch_start);
    for (uint32_t k = 0; k < batch_size && !done; ++k) {
      const float3 conic = conic_batch[k];
      const float3 xyo = xy_opacity[k];
      const float dx = xyo.x - px, dy = xyo.y - py;
      const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
      // the mask scales the transmittance update, not the threshold test
      const float alpha = fminf(kGsMaxAlpha, xyo.z * __expf(-sigma));
      if (sigma < 0.f || alpha < kGsAlphaThreshold)
        continue;
      const float gm = gate_batch[k];
      const float next_T = T * (1.f - gm * alpha);
      if (next_T <= kGsMinTransmittance) {
        done = true;
        break;
      }
      const float vis = gm * alpha * T;
      const float *c = a.colors + kGsChannels * id_batch[k];
#pragma unroll
      for (int ch = 0; ch < kGsChannels; ++ch)
        out[ch] += c[ch] * vis;
      cur_idx = batch_start + k;
      T = next_T;
    }
  }
  if (inside) {
    alphas[pix] = 1.f - T;
#pragma unroll
    for (int ch = 0; ch < kGsChannels; ++ch)
      render[kGsChannels * pix + ch] = out[ch];
    last_ids[pix] = static_cast<int32_t>(cur_idx);
  }
}

struct GsRasterGrads {
  float *v_means2d, *v_means2d_abs, *v_conics, *v_colors, *v_opacities, *v_gate;
};

__global__ void gs_raster_bwd_kernel(GsRasterArgs a, const float *alphas, const int32_t *last_ids,
                                     const float *v_render, const float *v_alphas, GsRasterGrads o) {
  auto block = cg::this_thread_block();
  const int32_t tile_id = blockIdx.y * a.t.tile_width + blockIdx.x;
  const uint32_t i = blockIdx.y * a.t.tile_size + threadIdx.y;
  const uint32_t j = blockIdx.x * a.t.tile_size + threadIdx.x;
  const float px = (float)j + 0.5f, py = (float)i + 0.5f;
  const int32_t pix = min(i * a.width + j, a.width * a.height - 1);
  const bool inside = i < a.height && j < a.width;
  const int32_t range_start = a.offsets[tile_id];
  const int32_t range_end =
      tile_id == a.t.tile_width * a.t.tile_height - 1 ? a.n_isects : a.offsets[tile_id + 1];
  const uint32_t block_size = block.size();
  const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;
  extern __shared__ int gs_smem[];
  int32_t *id_batch = gs_smem;
  float3 *xy_opacity = reinterpret_cast<float3 *>(&id_batch[block_size]);
  float3 *conic_batch = &xy_opacity[block_size];
  float *rgbs_batch = reinterpret_cast<float *>(&conic_batch[block_size]);
  float *gate_batch = &rgbs_batch[block_size * kGsChannels];
  const float T_final = 1.f - alphas[pix];
  float T = T_final;
  float buffer[kGsChannels] = {0.f};
  const int32_t bin_final = inside ? last_ids[pix] : 0;
  float v_render_c[kGsChannels];
#pragma unroll
  for (int ch = 0; ch < kGsChannels; ++ch)
    v_render_c[ch] = v_render[kGsChannels * pix + ch];
  const float v_render_a = v_alphas[pix];
  const uint32_t tr = block.thread_rank();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
  for (uint32_t b = 0; b < num_batches; ++b) {
    block.sync();
    const int32_t batch_end = range_end - 1 - block_size * b;
    const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
    const int32_t idx = batch_end - tr;
    if (idx >= range_start) {
      const int32_t g = a.flatten_ids[idx];
      id_batch[tr] = g;
      xy_opacity[tr] = {a.means2d[2 * g], a.means2d[2 * g + 1], a.opacities[g]};
      conic_batch[tr] = {a.conics[3 * g], a.conics[3 * g + 1], a.conics[3 * g + 2]};
#pragma unroll
      for (int ch = 0; ch < kGsChannels; ++ch)
        rgbs_batch[tr * kGsChannels + ch] = a.colors[kGsChannels * g + ch];
      gate_batch[tr] = a.gate != nullptr ? a.gate[g] : 1.f;
    }
    block.sync();
    for (int32_t k = max(0, batch_end - warp_bin_final); k < batch_size; ++k) {
      bool valid = inside && batch_end - k <= bin_final;
      float alpha = 0.f, opac = 0.f, vis = 0.f, dx = 0.f, dy = 0.f;
      float3 conic = {0.f, 0.f, 0.f};
      if (valid) {
        conic = conic_batch[k];
        const float3 xyo = xy_opacity[k];
        opac = xyo.z;
        dx = xyo.x - px;
        dy = xyo.y - py;
        const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
        vis = __expf(-sigma);
        alpha = fminf(kGsMaxAlpha, opac * vis);
        if (sigma < 0.f || alpha < kGsAlphaThreshold)
          valid = false;
      }
      if (!warp.any(valid))
        continue;
      float v_rgb[kGsChannels] = {0.f};
      float3 v_conic = {0.f, 0.f, 0.f};
      float2 v_xy = {0.f, 0.f}, v_xy_abs = {0.f, 0.f};
      float v_opacity = 0.f, v_gm = 0.f;
      if (valid) {
        const float gm = gate_batch[k];
        const float ma = gm * alpha;
        const float ra = 1.f / (1.f - ma);
        T *= ra;
        const float fac = ma * T;
#pragma unroll
        for (int ch = 0; ch < kGsChannels; ++ch)
          v_rgb[ch] = fac * v_render_c[ch];
        float v_ma = 0.f;
#pragma unroll
        for (int ch = 0; ch < kGsChannels; ++ch)
          v_ma += (rgbs_batch[k * kGsChannels + ch] * T - buffer[ch] * ra) * v_render_c[ch];
        v_ma += T_final * ra * v_render_a;
        const float v_alpha = v_ma * gm;
        v_gm = v_ma * alpha;
        if (opac * vis <= kGsMaxAlpha) {
          const float v_sigma = -opac * vis * v_alpha;
          v_conic = {0.5f * v_sigma * dx * dx, v_sigma * dx * dy, 0.5f * v_sigma * dy * dy};
          v_xy = {v_sigma * (conic.x * dx + conic.y * dy), v_sigma * (conic.y * dx + conic.z * dy)};
          v_xy_abs = {fabsf(v_xy.x), fabsf(v_xy.y)};
          v_opacity = vis * v_alpha;
        }
#pragma unroll
        for (int ch = 0; ch < kGsChannels; ++ch)
          buffer[ch] += rgbs_batch[k * kGsChannels + ch] * fac;
      }
#pragma unroll
      for (int ch = 0; ch < kGsChannels; ++ch)
        v_rgb[ch] = cg::reduce(warp, v_rgb[ch], cg::plus<float>());
      v_conic.x = cg::reduce(warp, v_conic.x, cg::plus<float>());
      v_conic.y = cg::reduce(warp, v_conic.y, cg::plus<float>());
      v_conic.z = cg::reduce(warp, v_conic.z, cg::plus<float>());
      v_xy.x = cg::reduce(warp, v_xy.x, cg::plus<float>());
      v_xy.y = cg::reduce(warp, v_xy.y, cg::plus<float>());
      if (o.v_means2d_abs != nullptr) {
        v_xy_abs.x = cg::reduce(warp, v_xy_abs.x, cg::plus<float>());
        v_xy_abs.y = cg::reduce(warp, v_xy_abs.y, cg::plus<float>());
      }
      v_opacity = cg::reduce(warp, v_opacity, cg::plus<float>());
      v_gm = cg::reduce(warp, v_gm, cg::plus<float>());
      if (warp.thread_rank() == 0) {
        const int32_t g = id_batch[k];
#pragma unroll
        for (int ch = 0; ch < kGsChannels; ++ch)
          atomicAdd(o.v_colors + kGsChannels * g + ch, v_rgb[ch]);
        atomicAdd(o.v_conics + 3 * g, v_conic.x);
        atomicAdd(o.v_conics + 3 * g + 1, v_conic.y);
        atomicAdd(o.v_conics + 3 * g + 2, v_conic.z);
        atomicAdd(o.v_means2d + 2 * g, v_xy.x);
        atomicAdd(o.v_means2d + 2 * g + 1, v_xy.y);
        if (o.v_means2d_abs != nullptr) {
          atomicAdd(o.v_means2d_abs + 2 * g, v_xy_abs.x);
          atomicAdd(o.v_means2d_abs + 2 * g + 1, v_xy_abs.y);
        }
        atomicAdd(o.v_opacities + g, v_opacity);
        if (o.v_gate != nullptr)
          atomicAdd(o.v_gate + g, v_gm);
      }
    }
  }
}

static int gs_blocks(int64_t n) { return static_cast<int>((n + kGsThreads - 1) / kGsThreads); }

static GsTiles gs_tiles(uint32_t width, uint32_t height, uint32_t tile_size) {
  GsTiles t;
  t.tile_size = tile_size;
  t.tile_width = (width + tile_size - 1) / tile_size;
  t.tile_height = (height + tile_size - 1) / tile_size;
  t.tile_bits = 0;
  while ((1u << t.tile_bits) <= t.tile_width * t.tile_height)
    ++t.tile_bits;
  return t;
}

// viewmat (row-major 4x4 world-to-camera) and K (row-major 3x3) are host pointers
static GsProjectArgs gs_project_args(const void *means, const void *quats, const void *log_scales,
                                     const void *logit_opacities, const void *sh0, const float *viewmat,
                                     const float *K, int n, int width, int height, float eps2d) {
  GsProjectArgs a;
  a.means = static_cast<const float *>(means);
  a.quats = static_cast<const float *>(quats);
  a.log_scales = static_cast<const float *>(log_scales);
  a.logit_opacities = static_cast<const float *>(logit_opacities);
  a.sh0 = static_cast<const float *>(sh0);
  for (int i = 0; i < 16; ++i)
    a.viewmat[i] = viewmat[i];
  a.fx = K[0];
  a.cx = K[2];
  a.fy = K[4];
  a.cy = K[5];
  a.n = static_cast<uint32_t>(n);
  a.width = static_cast<uint32_t>(width);
  a.height = static_cast<uint32_t>(height);
  a.eps2d = eps2d;
  a.near_plane = 0.f;
  a.far_plane = 0.f;
  a.radius_clip = 0.f;
  return a;
}

extern "C" int gs_project_fwd(const void *means, const void *quats, const void *log_scales,
                              const void *logit_opacities, const void *sh0, const float *viewmat, const float *K,
                              int n, int width, int height, float eps2d, float near_plane, float far_plane,
                              float radius_clip, void *radii, void *means2d, void *depths, void *conics,
                              void *compensations, void *opacities, void *colors, int64_t stream) {
  GsProjectArgs a = gs_project_args(means, quats, log_scales, logit_opacities, sh0, viewmat, K, n, width, height,
                                    eps2d);
  a.near_plane = near_plane;
  a.far_plane = far_plane;
  a.radius_clip = radius_clip;
  if (n > 0)
    gs_project_fwd_kernel<<<gs_blocks(n), kGsThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        a, static_cast<int32_t *>(radii), static_cast<float *>(means2d), static_cast<float *>(depths),
        static_cast<float *>(conics), static_cast<float *>(compensations), static_cast<float *>(opacities),
        static_cast<float *>(colors));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int gs_project_bwd(const void *means, const void *quats, const void *log_scales,
                              const void *logit_opacities, const void *sh0, const float *viewmat, const float *K,
                              int n, int width, int height, float eps2d, const void *radii, const void *conics,
                              const void *compensations, const void *v_means2d, const void *v_conics,
                              const void *v_colors, const void *v_opacities, void *v_means, void *v_quats,
                              void *v_log_scales, void *v_logit_opacities, void *v_sh0, int64_t stream) {
  const GsProjectArgs a = gs_project_args(means, quats, log_scales, logit_opacities, sh0, viewmat, K, n, width,
                                          height, eps2d);
  if (n > 0)
    gs_project_bwd_kernel<<<gs_blocks(n), kGsThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        a, static_cast<const int32_t *>(radii), static_cast<const float *>(conics),
        static_cast<const float *>(compensations), static_cast<const float *>(v_means2d),
        static_cast<const float *>(v_conics), static_cast<const float *>(v_colors),
        static_cast<const float *>(v_opacities), static_cast<float *>(v_means), static_cast<float *>(v_quats),
        static_cast<float *>(v_log_scales), static_cast<float *>(v_logit_opacities), static_cast<float *>(v_sh0));
  return static_cast<int>(cudaGetLastError());
}

// scratch bytes for gs_tiles_count (n Gaussians) and gs_tiles_sort (n_isects intersections), whichever is larger
extern "C" int64_t gs_tiles_temp_bytes(int n, int64_t n_isects, int width, int height, int tile_size) {
  const GsTiles t = gs_tiles(width, height, tile_size);
  size_t scan = 0, sort = 0;
  cub::DeviceScan::InclusiveSum(nullptr, scan, static_cast<int64_t *>(nullptr), static_cast<int64_t *>(nullptr), n);
  cub::DeviceRadixSort::SortPairs(nullptr, sort, static_cast<int64_t *>(nullptr), static_cast<int64_t *>(nullptr),
                                  static_cast<int32_t *>(nullptr), static_cast<int32_t *>(nullptr),
                                  static_cast<int>(n_isects), 0, 32 + static_cast<int>(t.tile_bits));
  return static_cast<int64_t>(scan > sort ? scan : sort);
}

// tiles each Gaussian touches and their running sum (the last entry is the intersection count)
extern "C" int gs_tiles_count(const void *radii, const void *means2d, int n, int width, int height, int tile_size,
                              void *tiles_per_gauss, void *cum_tiles, void *temp, int64_t temp_bytes,
                              int64_t stream) {
  const GsTiles t = gs_tiles(width, height, tile_size);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (n == 0)
    return 0;
  gs_tiles_count_kernel<<<gs_blocks(n), kGsThreads, 0, s>>>(static_cast<const int32_t *>(radii),
                                                             static_cast<const float *>(means2d), n, t,
                                                             static_cast<int64_t *>(tiles_per_gauss));
  size_t bytes = static_cast<size_t>(temp_bytes);
  cub::DeviceScan::InclusiveSum(temp, bytes, static_cast<int64_t *>(tiles_per_gauss), static_cast<int64_t *>(cum_tiles),
                                n, s);
  return static_cast<int>(cudaGetLastError());
}

// sorted intersections (tile, then depth) and each tile's first one
extern "C" int gs_tiles_sort(const void *radii, const void *means2d, const void *depths, const void *cum_tiles, int n,
                             int width, int height, int tile_size, int64_t n_isects, void *keys, void *values,
                             void *keys_sorted, void *values_sorted, void *offsets, void *temp, int64_t temp_bytes,
                             int64_t stream) {
  const GsTiles t = gs_tiles(width, height, tile_size);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const uint32_t n_tiles = t.tile_width * t.tile_height;
  if (n_isects == 0) {
    cudaMemsetAsync(offsets, 0, n_tiles * sizeof(int32_t), s);
    return static_cast<int>(cudaGetLastError());
  }
  gs_tiles_emit_kernel<<<gs_blocks(n), kGsThreads, 0, s>>>(
      static_cast<const int32_t *>(radii), static_cast<const float *>(means2d), static_cast<const float *>(depths),
      static_cast<const int64_t *>(cum_tiles), n, t, static_cast<int64_t *>(keys), static_cast<int32_t *>(values));
  size_t bytes = static_cast<size_t>(temp_bytes);
  cub::DeviceRadixSort::SortPairs(temp, bytes, static_cast<int64_t *>(keys), static_cast<int64_t *>(keys_sorted),
                                  static_cast<int32_t *>(values), static_cast<int32_t *>(values_sorted),
                                  static_cast<int>(n_isects), 0, 32 + static_cast<int>(t.tile_bits), s);
  gs_tile_offsets_kernel<<<gs_blocks(n_isects), kGsThreads, 0, s>>>(static_cast<const int64_t *>(keys_sorted),
                                                                    static_cast<uint32_t>(n_isects), n_tiles,
                                                                    static_cast<int32_t *>(offsets));
  return static_cast<int>(cudaGetLastError());
}

static GsRasterArgs gs_raster_args(const void *means2d, const void *conics, const void *colors,
                                   const void *opacities, const void *gate, const void *offsets,
                                   const void *flatten_ids, int width, int height, int tile_size, int64_t n_isects) {
  return {static_cast<const float *>(means2d),
          static_cast<const float *>(conics),
          static_cast<const float *>(colors),
          static_cast<const float *>(opacities),
          static_cast<const float *>(gate),
          static_cast<const int32_t *>(offsets),
          static_cast<const int32_t *>(flatten_ids),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          static_cast<uint32_t>(n_isects),
          gs_tiles(width, height, tile_size)};
}

// (rgb, accumulated z) per pixel, alpha, and each pixel's last contributing intersection; gate may be null
extern "C" int gs_raster_fwd(const void *means2d, const void *conics, const void *colors, const void *opacities,
                             const void *gate, const void *offsets, const void *flatten_ids, int width, int height,
                             int tile_size, int64_t n_isects, void *render, void *alphas, void *last_ids,
                             int64_t stream) {
  const GsRasterArgs a = gs_raster_args(means2d, conics, colors, opacities, gate, offsets, flatten_ids, width, height,
                                        tile_size, n_isects);
  const dim3 grid(a.t.tile_width, a.t.tile_height), block(tile_size, tile_size);
  const size_t smem = tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3) + sizeof(float));
  gs_raster_fwd_kernel<<<grid, block, smem, reinterpret_cast<cudaStream_t>(stream)>>>(
      a, static_cast<float *>(render), static_cast<float *>(alphas), static_cast<int32_t *>(last_ids));
  return static_cast<int>(cudaGetLastError());
}

// accumulates into zeroed gradients; v_means2d_abs and v_gate may be null
extern "C" int gs_raster_bwd(const void *means2d, const void *conics, const void *colors, const void *opacities,
                             const void *gate, const void *offsets, const void *flatten_ids, int width, int height,
                             int tile_size, int64_t n_isects, const void *alphas, const void *last_ids,
                             const void *v_render, const void *v_alphas, void *v_means2d, void *v_means2d_abs,
                             void *v_conics, void *v_colors, void *v_opacities, void *v_gate, int64_t stream) {
  const GsRasterArgs a = gs_raster_args(means2d, conics, colors, opacities, gate, offsets, flatten_ids, width, height,
                                        tile_size, n_isects);
  const GsRasterGrads o = {static_cast<float *>(v_means2d), static_cast<float *>(v_means2d_abs),
                           static_cast<float *>(v_conics),  static_cast<float *>(v_colors),
                           static_cast<float *>(v_opacities), static_cast<float *>(v_gate)};
  const dim3 grid(a.t.tile_width, a.t.tile_height), block(tile_size, tile_size);
  const size_t smem =
      tile_size * tile_size * (sizeof(int32_t) + 2 * sizeof(float3) + (kGsChannels + 1) * sizeof(float));
  gs_raster_bwd_kernel<<<grid, block, smem, reinterpret_cast<cudaStream_t>(stream)>>>(
      a, static_cast<const float *>(alphas), static_cast<const int32_t *>(last_ids),
      static_cast<const float *>(v_render), static_cast<const float *>(v_alphas), o);
  return static_cast<int>(cudaGetLastError());
}
