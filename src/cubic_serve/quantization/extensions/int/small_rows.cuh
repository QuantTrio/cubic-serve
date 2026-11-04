// INT owns signed-offset interpretation, zero correction and scale-dtype rounding.
#include <type_traits>
struct FormatContext { long long zero_offset; int groups; };
#ifdef CUBIC_ROW_RUNTIME_GEOMETRY
__device__ __forceinline__ FormatContext format_context(const std::uint8_t*, int,
                                                       RowGeometry geometry) {
    return {geometry.zero_offset, geometry.k / GI};
}
#else
__device__ __forceinline__ FormatContext format_context(const std::uint8_t*, int) {
    return {ZERO_OFFSET, INPUT_K / GI};
}
#endif
__device__ __forceinline__ FormatContext format_rebind_context(FormatContext context, int zero_offset) {
    context.zero_offset=zero_offset;return context;
}
#ifdef CUBIC_ROW_RUNTIME_GEOMETRY
__device__ __forceinline__ void format_copy(std::uint8_t* target, const std::uint8_t* blob,
                                           int column, int tile, int lane, RowGeometry geometry) {
    packed_group_copy<FORMAT_RAW_BYTES>(target, blob, column, tile, lane, geometry.k);
}
#else
__device__ __forceinline__ void format_copy(std::uint8_t* target, const std::uint8_t* blob,
                                           int column, int tile, int lane) {
    packed_group_copy<FORMAT_RAW_BYTES>(target, blob, column, tile, lane);
}
#endif
__device__ __forceinline__ void format_expand(float2* target, const std::uint8_t* raw,
    const std::uint8_t* blob, FormatContext context, int column, int tile, int lane) {
    if (lane >= TILE_K / GI) return;
    const int groups = context.groups;
    const int group = tile * (TILE_K / GI) + lane;
    float scale;
    if constexpr (SCALE_KIND == 0) scale = __half2float(reinterpret_cast<const __half*>(raw)[lane]);
    else if constexpr (SCALE_KIND == 1) scale = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(raw)[lane]);
    else scale = reinterpret_cast<const float*>(raw)[lane];
    float zero = 0;
    if constexpr (ASYMMETRIC) {
        const auto* zeros = reinterpret_cast<const unsigned*>(blob + context.zero_offset);
        const int bit = LEGACY ? (column / (32 / BITS)) * 32 + (column % (32 / BITS)) * BITS : column * BITS;
        const int word = bit / 32, shift = bit % 32;
        unsigned raw = zeros[std::int64_t(word) * groups + group] >> shift;
        if (shift + BITS > 32) raw |= zeros[std::int64_t(word + 1) * groups + group] << (32 - shift);
        zero = float(int(raw & ((1u << BITS) - 1)) - (1 << (BITS - 1)));
    }
    target[lane] = make_float2(scale, zero);
}
struct FormatLevels {};
__device__ __forceinline__ FormatLevels format_levels(float2, int) { return {}; }
__device__ __forceinline__ float format_decode(unsigned raw, FormatLevels, float2 metadata, int) {
    const float q = float(int(raw) - (1 << (BITS - 1))) - metadata.y;
    const float value = __fmul_rn(q, metadata.x);
    float weight;
    if constexpr (SCALE_KIND == 0) weight = __half2float(__float2half_rn(value));
    else if constexpr (SCALE_KIND == 1) weight = __bfloat162float(__float2bfloat16_rn(value));
    else weight = value;
    if constexpr (IS_BF16) return __bfloat162float(__float2bfloat16_rn(weight));
    else return __half2float(__float2half_rn(weight));
}
__device__ __forceinline__ void format_emit_lut(unsigned* target, const std::uint8_t* raw,
    const std::uint8_t* blob, FormatContext context, int column, int tile, int group, int pair) {
    float2 metadata;
    format_expand(&metadata, raw + group*FORMAT_RAW_BYTES, blob, context,
                  column, tile, 0);
    // Zero-point ownership uses the absolute checkpoint group, not the CTA's
    // local metadata index. Re-evaluate its packed field at that group.
    if constexpr (ASYMMETRIC) {
        const int groups = context.groups;
        const int absolute_group = tile*(TILE_K/GI) + group;
        const auto* zeros = reinterpret_cast<const unsigned*>(blob + context.zero_offset);
        const int bit = LEGACY ? (column/(32/BITS))*32 + (column%(32/BITS))*BITS : column*BITS;
        const int word = bit/32, shift = bit%32;
        unsigned value = zeros[std::int64_t(word)*groups + absolute_group] >> shift;
        if (shift + BITS > 32) value |= zeros[std::int64_t(word+1)*groups+absolute_group] << (32-shift);
        metadata.y = float(int(value & ((1u<<BITS)-1)) - (1<<(BITS-1)));
    }
    unsigned result = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float value = format_decode(pair*2+i, {}, metadata, 0);
        unsigned bits;
        if constexpr (IS_BF16) bits = __bfloat16_as_ushort(__float2bfloat16_rn(value));
        else bits = __half_as_ushort(__float2half_rn(value));
        result |= bits << (16*i);
    }
    target[pair] = result;
}
