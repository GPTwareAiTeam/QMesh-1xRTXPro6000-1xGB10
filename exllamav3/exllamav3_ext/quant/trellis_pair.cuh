#pragma once

// Paired routed-expert trellis layout (qmesh/pair_layout.py, notes/CoopKernel2.md §11). A per-tensor property: the
// pack's files carry a layout tag (exllamav3/util/trellis_layout.py) and every routed-expert entry point takes a
// `paired` flag for the launch; dense / attention / shared-expert linears are always plain and never reach a paired
// instance. Rule, the same everywhere: a trellis of a paired launch is in the paired layout iff its rate is integer
// 2, 3 or 4 bpw (pairable); every other rate stays plain.
//
// Plain layout of a (ks, nt, 16 K) int16 trellis: 16x16 tile (kb, nb) is TW = 8 K consecutive uint32 at word offset
// (kb * nt + nb) * TW. Paired: each pair of k-slices (2j, 2j + 1) is one region of 2 nt TW words, cut per chunk c of
// two adjacent n-tiles (2c, 2c + 1) into a block of 4 TW words; inside a block word (L, h, l) = L * 2 LOADS + h * LOADS
// + l holds plain word l * NL + L of slice 2j + h's chunk (NL lanes x LOADS words cover the chunk's 2 TW words:
// (LOADS, NL) = (1, 32) K2, (2, 24) K3, (2, 32) K4). The four plain tiles (2j, 2c), (2j, 2c + 1), (2j + 1, 2c),
// (2j + 1, 2c + 1) are exactly one paired block, so a reader staging whole 2 x 2 tile groups copies blocks and places
// words; single words come from pair_word()

namespace trellis_pair {

__host__ __device__ constexpr bool pairable(int bits, bool half) { return !half && bits >= 2 && bits <= 4; }
// k2 = 2 * bits + half (bits_k.cuh)
__host__ __device__ constexpr bool pairable_k2(int k2) { return k2 == 4 || k2 == 6 || k2 == 8; }

template <int bits> constexpr int loads() { return bits == 2 ? 1 : 2; }
template <int bits> constexpr int lanes() { return bits == 3 ? 24 : 32; }

// Plain word (tile parity e within the chunk, word w < TW) of a slice's chunk -> word offset inside the paired block,
// for slice parity h
template <int bits>
__host__ __device__ __forceinline__ int block_word(int h, int e, int w)
{
    constexpr int TW = 8 * bits;
    const int cw = e * TW + w;
    const int l = cw / lanes<bits>();
    const int L = cw % lanes<bits>();
    return L * 2 * loads<bits>() + h * loads<bits>() + l;
}

// Inverse: paired word pw (< 4 TW) of a block -> (h, e, w) of the plain tile word it holds
template <int bits>
__host__ __device__ __forceinline__ void block_unword(int pw, int& h, int& e, int& w)
{
    constexpr int TW = 8 * bits;
    const int L = pw / (2 * loads<bits>());
    h = (pw / loads<bits>()) & 1;
    const int l = pw % loads<bits>();
    const int cw = l * lanes<bits>() + L;
    e = cw / TW;
    w = cw % TW;
}

// Word offset in the whole paired tensor of plain word w of tile (kb, nb); nt = n-tiles of the tensor (even)
template <int bits>
__host__ __device__ __forceinline__ size_t pair_word(int kb, int nb, int nt, int w)
{
    constexpr int TW = 8 * bits;
    const size_t block = ((size_t) (kb >> 1) * (nt >> 1) + (nb >> 1)) * 4 * TW;
    return block + block_word<bits>(kb & 1, nb & 1, w);
}

}  // namespace trellis_pair
