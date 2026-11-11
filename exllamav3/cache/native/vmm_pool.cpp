// Growable device pool on CUDA virtual memory management (driver API): reserve a virtual range for the maximum size
// once, map physical chunks into it on demand (grow) and unmap them again (shrink). The base address never moves, so
// kernels and captured CUDA graphs that bake it stay valid. Exposed to torch as a tensor view of the full range;
// only [0, mapped) may be touched.
#include <torch/extension.h>
#include <cuda.h>
#include <vector>

namespace {

#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(r_, &s_); \
    TORCH_CHECK(false, #x " failed: ", s_ ? s_ : "?"); } } while (0)

struct Pool {
    int device;
    CUdeviceptr base = 0;
    size_t reserved = 0, granule = 0, mapped = 0;
    std::vector<CUmemGenericAllocationHandle> chunks;     // one per granule, in address order
};

CUmemAllocationProp prop_for(int device)
{
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = device;
    return p;
}

}  // namespace

int64_t vmm_create(int64_t device, int64_t max_bytes)
{
    CU(cuInit(0));
    auto* p = new Pool();
    p->device = (int) device;
    CUmemAllocationProp prop = prop_for(p->device);
    CU(cuMemGetAllocationGranularity(&p->granule, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    p->reserved = (max_bytes + p->granule - 1) / p->granule * p->granule;
    CU(cuMemAddressReserve(&p->base, p->reserved, p->granule, 0, 0));
    return (int64_t) p;
}

// Map physical memory so that at least `bytes` from the base are backed. Returns the mapped size.
int64_t vmm_grow(int64_t h, int64_t bytes)
{
    auto* p = (Pool*) h;
    TORCH_CHECK((size_t) bytes <= p->reserved, "vmm_grow: ", bytes, " > reserved ", p->reserved);
    CUmemAllocationProp prop = prop_for(p->device);
    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = p->device;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    while (p->mapped < (size_t) bytes)
    {
        CUmemGenericAllocationHandle c;
        CU(cuMemCreate(&c, p->granule, &prop, 0));
        CU(cuMemMap(p->base + p->mapped, p->granule, 0, c, 0));
        CU(cuMemSetAccess(p->base + p->mapped, p->granule, &acc, 1));
        p->chunks.push_back(c);
        p->mapped += p->granule;
    }
    return (int64_t) p->mapped;
}

// Unmap and release physical memory down to `bytes` (rounded up to a granule). Caller guarantees nothing in flight
// reads the released tail (synchronize first). Returns the mapped size.
int64_t vmm_shrink(int64_t h, int64_t bytes)
{
    auto* p = (Pool*) h;
    size_t keep = ((size_t) bytes + p->granule - 1) / p->granule * p->granule;
    while (p->mapped > keep)
    {
        p->mapped -= p->granule;
        CU(cuMemUnmap(p->base + p->mapped, p->granule));
        CU(cuMemRelease(p->chunks.back()));
        p->chunks.pop_back();
    }
    return (int64_t) p->mapped;
}

at::Tensor vmm_view(int64_t h, int64_t nbytes)
{
    auto* p = (Pool*) h;
    TORCH_CHECK((size_t) nbytes <= p->reserved);
    return torch::from_blob((void*) p->base, {nbytes}, [](void*) {},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, p->device));
}

std::vector<int64_t> vmm_info(int64_t h)
{
    auto* p = (Pool*) h;
    return { (int64_t) p->base, (int64_t) p->reserved, (int64_t) p->mapped, (int64_t) p->granule };
}

void vmm_destroy(int64_t h)
{
    auto* p = (Pool*) h;
    vmm_shrink(h, 0);
    cuMemAddressFree(p->base, p->reserved);
    delete p;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vmm_create", &vmm_create);
    m.def("vmm_grow", &vmm_grow);
    m.def("vmm_shrink", &vmm_shrink);
    m.def("vmm_view", &vmm_view);
    m.def("vmm_info", &vmm_info);
    m.def("vmm_destroy", &vmm_destroy);
}
