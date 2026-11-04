// Cubic owns polynomial decoding and the activation-storage rounding boundary.
#include <type_traits>
struct FormatContext { float a, b, scale; uint4 curves; };
__device__ __forceinline__ FormatContext format_rebind_context(FormatContext context, int) {
    return context;
}
__device__ __forceinline__ FormatContext format_context_impl(const std::uint8_t* blob, int column, int descriptors) {
    const auto* descriptor = blob;
    #pragma unroll
    for (int part = 0; part < descriptors - 1; ++part)
        if (column >= *reinterpret_cast<const unsigned*>(descriptor)) descriptor += 32;
    FormatContext result = {};
    if constexpr (METADATA_VARIANT == 1) {
        const auto* globals = reinterpret_cast<const float*>(descriptor + 4);
        result.a = globals[0]; result.b = globals[1]; result.scale = globals[2];
    } else if constexpr (METADATA_VARIANT == 2)
        result.curves = *reinterpret_cast<const uint4*>(descriptor + 16);
    return result;
}
#ifdef CUBIC_ROW_RUNTIME_GEOMETRY
__device__ __forceinline__ FormatContext format_context(const std::uint8_t* blob, int column,
                                                       RowGeometry geometry) {
    return format_context_impl(blob, column, geometry.descriptors);
}
__device__ __forceinline__ void format_copy(std::uint8_t* target, const std::uint8_t* blob,
                                           int column, int tile, int lane, RowGeometry geometry) {
    packed_group_copy<FORMAT_RAW_BYTES>(target, blob + geometry.raw_offset,
                                       column, tile, lane, geometry.k);
}
#else
__device__ __forceinline__ FormatContext format_context(const std::uint8_t* blob, int column) {
    return format_context_impl(blob, column, DESCRIPTOR_COUNT);
}
__device__ __forceinline__ void format_copy(std::uint8_t* target, const std::uint8_t* blob,
                                           int column, int tile, int lane) {
    packed_group_copy<FORMAT_RAW_BYTES>(target, blob + RAW_OFFSET, column, tile, lane);
}
#endif
__device__ __forceinline__ void format_expand(float2* target, const std::uint8_t* raw,
    const std::uint8_t*, FormatContext context, int, int, int lane) {
    if (lane >= TILE_K / GI) return;
    float a, b, scale;
    if constexpr (METADATA_VARIANT == 1) {
        const unsigned packed = reinterpret_cast<const unsigned short*>(raw)[lane];
        const int ac = int((packed >> 8) & 15) - ((packed & 0x800) ? 16 : 0);
        const int bc = int(packed >> 12) - ((packed & 0x8000) ? 16 : 0);
        a = __half2float(__float2half_rn(__fadd_rn(1.f, __fmul_rn(float(ac), context.a))));
        b = __half2float(__float2half_rn(__fmul_rn(float(bc), context.b)));
        scale = __fmul_rn(float(static_cast<signed char>(packed & 255)), context.scale);
    } else if constexpr (METADATA_VARIANT == 2) {
        const unsigned packed = reinterpret_cast<const unsigned short*>(raw)[lane];
        const int curve = packed >> 14;
        const unsigned aa = curve < 2 ? context.curves.x : context.curves.y;
        const unsigned bb = curve < 2 ? context.curves.z : context.curves.w;
        a = __half2float(__ushort_as_half(aa >> (16 * (curve & 1))));
        b = __half2float(__ushort_as_half(bb >> (16 * (curve & 1))));
        scale = __half2float(__ushort_as_half((packed & 0x3fff) << 1));
    } else {
        const auto* entry = raw + lane * 8;
        scale = *reinterpret_cast<const float*>(entry);
        a = __half2float(*reinterpret_cast<const __half*>(entry + 4));
        b = __half2float(*reinterpret_cast<const __half*>(entry + 6));
    }
    const unsigned ab = __half_as_ushort(__float2half_rn(a)) |
                        (unsigned(__half_as_ushort(__float2half_rn(b))) << 16);
    target[lane] = make_float2(__uint_as_float(ab), scale);
}
constexpr int group_lanes = GI / ELEMENTS < 32 ? GI / ELEMENTS : 32;
constexpr bool warp_groups_aligned = GI % (32 * ELEMENTS) == 0 || (32 * ELEMENTS) % GI == 0;
constexpr int positive_codes = 1 << (BITS - 1);
constexpr int levels_per_lane = (positive_codes + group_lanes - 1) / group_lanes;
struct FormatLevels { float value[levels_per_lane]; };
__device__ __forceinline__ float format_level(int code, float2 metadata) {
    const unsigned ab = __float_as_uint(metadata.x);
    const float a = __half2float(__ushort_as_half(ab & 65535));
    const float b = __half2float(__ushort_as_half(ab >> 16));
    const float t = __fmaf_rn(float(code), RECIP_HI, __fmul_rn(float(code), RECIP_LO));
    const float c = __fsub_rn(__fsub_rn(1.f, a), b);
    const float v = __fmul_rn(t, __fadd_rn(a, __fmul_rn(t, __fadd_rn(b, __fmul_rn(t, c)))));
    const float result = __fmul_rn(v, metadata.y);
    if constexpr (IS_BF16) return __bfloat162float(__float2bfloat16_rn(result));
    else return __half2float(__float2half_rn(result));
}
__device__ __forceinline__ FormatLevels format_levels(float2 metadata, int lane) {
    FormatLevels result = {};
    if constexpr (!warp_groups_aligned) return result;
    #pragma unroll
    for (int i = 0; i < levels_per_lane; ++i)
        result.value[i] = format_level((lane % group_lanes) * levels_per_lane + i, metadata);
    return result;
}
__device__ __forceinline__ float format_decode(unsigned raw, FormatLevels levels, float2 metadata, int lane) {
    int q = int(raw ^ positive_codes) - positive_codes;
    if (q == -positive_codes) q = 0;
    const int absolute = q < 0 ? -q : q;
    if constexpr (!warp_groups_aligned) {
        const float value = format_level(absolute, metadata);
        return q < 0 ? -value : value;
    }
    const int owner = (lane / group_lanes) * group_lanes + absolute / levels_per_lane;
    // Dynamic indexing of even a two-element register array lowers to local
    // memory. Keep every shuffle unconditional and select register values.
    float value = __shfl_sync(0xffffffff, levels.value[0], owner);
    #pragma unroll
    for (int i = 1; i < levels_per_lane; ++i) {
        const float alternative = __shfl_sync(0xffffffff, levels.value[i], owner);
        if (absolute % levels_per_lane == i) value = alternative;
    }
    return q < 0 ? -value : value;
}
