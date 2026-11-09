// a 2D convolution's shape and x's layout, as both its kernels read them
layout(push_constant) uniform PushConstants {
    uint batch;
    // a group's channels
    uint c_in;
    uint c_out;
    uint groups;
    uint i_h;
    uint i_w;
    uint k_h;
    uint k_w;
    uint o_h;
    uint o_w;
    uint pad_h;
    uint pad_w;
    uint stride;
    uint dilation;
    uint x_off;
    uint x_sb;
    uint x_sc;
    uint x_sh;
    uint x_sw;
    uint w_off;
} pc;
