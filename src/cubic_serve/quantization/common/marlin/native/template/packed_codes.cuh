#pragma once

#include "configuration.cuh"

namespace MARLIN_NAMESPACE_NAME {

// The persistent stream follows Marlin's MMA code order at its actual bit
// width. Only this register fragment uses the converter's 4/8-bit lanes.
template <int bits, int fragment_bits, int phase>
__device__ __forceinline__ int4 unpack_code_fragment(const unsigned int* src) {
  constexpr int codes = 128 / fragment_bits;
  constexpr int words = (codes * bits + 31) / 32;
  unsigned int packed[words + 1];
#pragma unroll
  for (int i = 0; i < words; ++i) packed[i] = src[i];
  packed[words] = 0;
  int4 result;
  auto dst = reinterpret_cast<unsigned int*>(&result);
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    unsigned int value = 0;
#pragma unroll
    for (int j = 0; j < 32 / fragment_bits; ++j) {
      const int offset = phase + (w * (32 / fragment_bits) + j) * bits;
      unsigned int code = packed[offset / 32] >> (offset % 32);
      if (offset % 32 + bits > 32)
        code |= packed[offset / 32 + 1] << (32 - offset % 32);
      code &= (1u << bits) - 1;
#if defined(CUBIC_MARLIN_INT_A8_FLOAT_SCALE)
      if constexpr (fragment_bits == 8) {
        // The native signed-byte MMA consumes sign-extended original codes.
        // W5-W7 remain compact until this register fragment is unpacked.
        code = (code - (1u << (bits - 1))) & 0xffu;
      } else
#endif
      {
        code += (1u << (fragment_bits - 1)) - (1u << (bits - 1));
      }
      value |= code << (j * fragment_bits);
    }
    dst[w] = value;
  }
  return result;
}

template <typename Config>
__device__ __forceinline__ int4 load_code_fragment(const int4* stage, int index) {
  constexpr int bits = Config::storage_bits;
  constexpr int fragment_bits = Config::b_type.size_bits();
  static_assert(bits >= 2 && bits <= fragment_bits);
  if constexpr (bits == fragment_bits) {
    return stage[index];
  } else {
    constexpr int codes = 128 / fragment_bits;
    const int bit = index * codes * bits;
    const auto src = reinterpret_cast<const unsigned int*>(stage) + bit / 32;
    // 32 codes always close a word. 16 codes at an odd width alternate
    // between two halfword phases; specialize those phases, not the device.
    if constexpr (codes * bits % 32 == 0)
      return unpack_code_fragment<bits, fragment_bits, 0>(src);
    else
      return bit % 32 ? unpack_code_fragment<bits, fragment_bits, 16>(src)
                      : unpack_code_fragment<bits, fragment_bits, 0>(src);
  }
}

}  // namespace MARLIN_NAMESPACE_NAME
