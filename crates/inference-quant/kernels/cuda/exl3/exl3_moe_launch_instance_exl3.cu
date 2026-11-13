// exllamav3 fused decode MoE launcher (third_party/exllamav3), and our C ABI over it
#include "quant/exl3_moe_coop.cu"

// One MoE layer's routed experts for a few tokens: gate and up, the activation and down, each slot weighted by its
// routing weight and summed per token into `out`. Each projection's table holds its experts' trellis, suh and svh
// addresses, `experts` apiece. Scratch is sized for `slots_max` slots (>= bsz x topk) and `ctr` zeroed
struct Exl3MoeArgs
{
    const half* x;              // (bsz, hidden)
    const int64_t* sel;         // (bsz, topk)
    const half* rw;             // (bsz, topk)
    const int64_t* gate_tables;
    const int64_t* up_tables;
    const int64_t* down_tables;
    half* had_g;                // (slots_max, hidden)
    half* had_u;
    float* gu_g;                // (slots_max, inter)
    float* gu_u;
    half* act_out;              // (slots_max, inter)
    float* d_out;               // (slots_max, hidden)
    int* ctr;                   // exl3_moe_coop_ctr_len(slots_max, bsz, inter, hidden)
    float* out;                 // (bsz, hidden)
    int bsz;
    int topk;
    int hidden;
    int inter;
    int experts;
    int act;                    // MOE_COOP_ACT_*
    int gate_up_words;          // uint16 words a tile
    int down_words;
    int cb;
    int slots_max;
};

extern "C" int64_t exl3_moe_ctr_len(int slots_max, int bsz, int inter, int hidden)
{
    return exl3_moe_coop_ctr_len(slots_max, bsz, inter, hidden);
}

// Returns 0 when launched, -1 when the shapes or rates fall outside the kernels
extern "C" int launch_exl3_moe(const Exl3MoeArgs* a, cudaStream_t stream)
{
    if (a->hidden % 128 || a->inter % 128 || a->slots_max < a->bsz * a->topk || a->slots_max > 256) return -1;
    auto rate = [] (int words) { return (words % 16 == 0 || words % 16 == 8) ? words / 16.0f : -1.0f; };
    float k_gu = rate(a->gate_up_words);
    float k_d = rate(a->down_words);
    if (k_gu < 1 || k_gu > 8 || k_d < 1 || k_d > 8) return -1;
    if ((k_gu != (int) k_gu || k_d != (int) k_d) && a->cb != 2) return -1;

    MoeCoopParams p = {};
    p.x = a->x;
    p.x_stride = a->hidden;
    p.sel = a->sel;
    p.rw = a->rw;
    p.bsz = a->bsz;
    p.topk = a->topk;
    p.H = a->hidden;
    p.Hi = a->hidden;
    p.I = a->inter;
    p.Ho = a->hidden;
    p.H_out = a->hidden;
    p.min_expert = -1;
    p.max_expert = -1;
    const int n = a->experts;
    p.g_trellis = a->gate_tables;
    p.g_suh = a->gate_tables + n;
    p.g_svh = a->gate_tables + 2 * n;
    p.u_trellis = a->up_tables;
    p.u_suh = a->up_tables + n;
    p.u_svh = a->up_tables + 2 * n;
    p.d_trellis = a->down_tables;
    p.d_suh = a->down_tables + n;
    p.d_svh = a->down_tables + 2 * n;
    p.act = a->act;
    p.gated = true;
    p.had_g = a->had_g;
    p.had_u = a->had_u;
    p.gu_g = a->gu_g;
    p.gu_u = a->gu_u;
    p.gu_f32 = true;
    p.act_out = a->act_out;
    p.d_out = a->d_out;
    p.slots_max = a->slots_max;
    p.rows_max = a->bsz;
    p.ctr_a_len = a->slots_max * (a->inter / 128);
    p.ctr_b_len = a->bsz * (a->hidden / 128);
    p.ctr_a = a->ctr;
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.n_local = n;
    p.out = a->out;
    p.out_stride = a->hidden;

    int device;
    cudaGetDevice(&device);
    exl3_moe_coop_launch(p, k_gu, k_d, a->cb, device, stream);
    return 0;
}
