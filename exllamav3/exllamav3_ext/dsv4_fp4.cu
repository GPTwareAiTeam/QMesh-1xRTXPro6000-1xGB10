#include "dsv4_fp4.cuh"
#include "graph.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

__global__ void fp4_positions_kernel(const int* positions, int* output, int rows, int m) {
    int j = threadIdx.x, b = blockIdx.x;
    if (j < rows) output[b * rows + j] = (positions[b] / m + j) * m;
}

void dsv4_fp4_positions(const at::Tensor& positions, at::Tensor output, int m, Graph* graph) {
    int batch = output.size(0), rows = output.size(1);
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    fp4_positions_kernel<<<batch, rows, 0, stream>>>(positions.data_ptr<int>(), output.data_ptr<int>(), rows, m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// E2M1 positive codes are 0,.5,1,1.5,2,3,4,6. Midpoints choose
// the code with an even low bit (round to nearest, ties to even).
__device__ unsigned fp4(float x) {
    bool neg = signbit(x);
    float a = fabsf(x);
    const float mid[] = {.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f};
    unsigned c = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i)
        if (a > mid[i] || (a == mid[i] && (i & 1))) c = i + 1;
    return c | (neg ? 8 : 0);
}
__device__ float fp4_value(unsigned c) {
    const float v[] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}

template<bool IDX> __device__ void quant_group(const half* in, unsigned char* out, unsigned char* s) {
    constexpr int G = IDX ? 32 : 16;
    float a = 0;
    for (int i = 0; i < G; ++i) a = fmaxf(a, fabsf(__half2float(in[i])));
    float scale;
    unsigned char bits;
    if (IDX) {
        float v = fmaxf(a, 6.f * 0x1p-126f) * (1.f / 6.f);
        unsigned u = __float_as_uint(v);
        int exp = (int)((u >> 23) & 255) + ((u & 0x7fffff) != 0);
        bits = (unsigned char) exp;
        scale = __uint_as_float((unsigned)exp << 23);
    } else {
        float v = fmaxf(a, 6.f * 0x1p-9f) / 6.f;
        __nv_fp8_e4m3 q(v);
        bits = q.__x;
        scale = (float)q;
    }
    *s = bits;
    for (int i = 0; i < G; i += 2) {
        float a0 = __fdiv_rn(__half2float(in[i]), scale), a1 = __fdiv_rn(__half2float(in[i+1]), scale);
        out[i/2] = fp4(a0) | (fp4(a1) << 4);
    }
}
template<bool IDX> __global__ void quant_kernel(const half* x, unsigned char* p, unsigned char* s, int rows, int width) {
    constexpr int G = IDX ? 32 : 16;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows * (width / G)) return;
    quant_group<IDX>(x + (size_t)g * G, p + (size_t)g * (G / 2), s + g);
}

template<bool IDX> __global__ void scatter_kernel(const half* stage, unsigned char* packed, unsigned char* scales,
        const int* bt, const int* positions, int position, int batch, int nw_max, int bt_stride, int m, int seq, int epp) {
    constexpr int width = IDX ? 128 : 512;
    constexpr int group = IDX ? 32 : 16;
    int job = blockIdx.y, w = blockIdx.x, g = threadIdx.x;
    int pos = positions ? positions[job] : position;
    int e0 = pos / m, n = (pos + seq) / m - e0;
    if (w >= n || g >= width / group) return;
    int e = e0 + w;
    size_t row = (size_t)bt[job * bt_stride + e / epp] * epp + e % epp;
    quant_group<IDX>(stage + ((size_t)job * nw_max + w) * width + g * group,
                     packed + row * (width / 2) + g * (group / 2), scales + row * (width / group) + g);
}
template<bool IDX> __global__ void dequant_kernel(const unsigned char* p, const unsigned char* s, half* out, int rows, int width) {
    constexpr int G = IDX ? 32 : 16;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    unsigned char b = p[i / 2];
    unsigned code = (i & 1) ? b >> 4 : b & 15;
    unsigned char sb = s[i / G];
    float scale;
    if (IDX) scale = __uint_as_float((unsigned)sb << 23);
    else { __nv_fp8_e4m3 q; q.__x = sb; scale = (float)q; }
    out[i] = __float2half_rn(fp4_value(code) * scale);
}

static void check(const at::Tensor& t, at::ScalarType type, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 2 && t.scalar_type() == type,
                name, " must be a contiguous 2D CUDA tensor of the expected dtype");
}
std::vector<at::Tensor> dsv4_fp4_quant(const at::Tensor& input, bool indexer) {
    check(input, at::kHalf, "input");
    int width = indexer ? 128 : 512;
    TORCH_CHECK(input.size(1) == width, "invalid input width");
    const c10::cuda::CUDAGuard guard(input.device());
    int rows = input.size(0), group = indexer ? 32 : 16;
    auto p = at::empty({rows, width / 2}, input.options().dtype(at::kByte));
    auto s = at::empty({rows, width / group}, input.options().dtype(at::kByte));
    int n = rows * width / group;
    if (n) {
        auto stream = at::cuda::getCurrentCUDAStream();
        if (indexer) quant_kernel<true><<<(n+127)/128,128,0,stream>>>((const half*)input.data_ptr(), p.data_ptr<unsigned char>(), s.data_ptr<unsigned char>(), rows, width);
        else quant_kernel<false><<<(n+127)/128,128,0,stream>>>((const half*)input.data_ptr(), p.data_ptr<unsigned char>(), s.data_ptr<unsigned char>(), rows, width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {p,s};
}
at::Tensor dsv4_fp4_dequant(const at::Tensor& packed, const at::Tensor& scales, bool indexer) {
    check(packed, at::kByte, "packed"); check(scales, at::kByte, "scales");
    int width = indexer ? 128 : 512, group = indexer ? 32 : 16;
    TORCH_CHECK(packed.size(1) == width/2 && scales.size(0) == packed.size(0) && scales.size(1) == width/group && packed.device() == scales.device(), "invalid FP4 shapes or devices");
    const c10::cuda::CUDAGuard guard(packed.device());
    int rows = packed.size(0), n = rows * width;
    auto out = at::empty({rows,width}, packed.options().dtype(at::kHalf));
    if (n) {
        auto stream = at::cuda::getCurrentCUDAStream();
        if (indexer) dequant_kernel<true><<<(n+255)/256,256,0,stream>>>(packed.data_ptr<unsigned char>(), scales.data_ptr<unsigned char>(), (half*)out.data_ptr(), rows, width);
        else dequant_kernel<false><<<(n+255)/256,256,0,stream>>>(packed.data_ptr<unsigned char>(), scales.data_ptr<unsigned char>(), (half*)out.data_ptr(), rows, width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

void dsv4_fp4_scatter_gr(const at::Tensor& stage, at::Tensor packed, at::Tensor scales,
    const at::Tensor& bt, int position, const c10::optional<at::Tensor>& positions,
    int m, int seq, int epp, bool indexer, Graph* graph) {
    int width = indexer ? 128 : 512, groups = indexer ? 4 : 32;
    TORCH_CHECK(stage.is_cuda() && stage.is_contiguous() && stage.scalar_type() == at::kHalf &&
                (stage.dim() == 2 || stage.dim() == 3) && stage.size(-1) == width, "invalid stage");
    check(packed, at::kByte, "packed"); check(scales, at::kByte, "scales");
    int batch = stage.dim() == 3 ? stage.size(0) : 1, nw = stage.size(-2);
    TORCH_CHECK(packed.device() == stage.device() && scales.device() == stage.device() &&
                packed.size(1) == width / 2 && scales.size(0) == packed.size(0) && scales.size(1) == groups &&
                bt.device() == stage.device() && bt.scalar_type() == at::kInt && bt.is_contiguous() &&
                bt.dim() == 2 && bt.size(0) >= batch && m > 0 && epp > 0 && seq >= 0, "invalid FP4 scatter arguments");
    if (positions) TORCH_CHECK(positions->device() == stage.device() && positions->scalar_type() == at::kInt &&
                               positions->is_contiguous() && positions->numel() >= batch, "invalid positions");
    int grid = positions ? seq / m + 1 : (position + seq) / m - position / m;
    TORCH_CHECK(grid <= nw && grid >= 0, "FP4 scatter staging too small");
    if (!grid || !batch || !seq) return;
    const c10::cuda::CUDAGuard guard(stage.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    const int* pos = positions ? positions->data_ptr<int>() : nullptr;
    if (indexer) scatter_kernel<true><<<dim3(grid,batch),32,0,stream>>>((const half*)stage.data_ptr(), packed.data_ptr<unsigned char>(), scales.data_ptr<unsigned char>(), bt.data_ptr<int>(), pos, position, batch, nw, bt.size(1), m, seq, epp);
    else scatter_kernel<false><<<dim3(grid,batch),32,0,stream>>>((const half*)stage.data_ptr(), packed.data_ptr<unsigned char>(), scales.data_ptr<unsigned char>(), bt.data_ptr<int>(), pos, position, batch, nw, bt.size(1), m, seq, epp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv4_fp4_scatter(const at::Tensor& stage, at::Tensor packed, at::Tensor scales,
    const at::Tensor& bt, int position, const c10::optional<at::Tensor>& positions,
    int m, int seq, int epp, bool indexer) {
    dsv4_fp4_scatter_gr(stage, packed, scales, bt, position, positions, m, seq, epp, indexer, nullptr);
}
