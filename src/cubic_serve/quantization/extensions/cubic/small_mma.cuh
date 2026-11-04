// Finite normalized curves are format constants, not expanded group weights.
struct FormatContext { unsigned curve_base; float global_scale; };
struct FormatLevels {};
__device__ __forceinline__ FormatContext format_context(const std::uint8_t* blob, int column) {
    const auto* descriptor = blob;
    #pragma unroll
    for (int part = 0; part < DESCRIPTOR_COUNT - 1; ++part)
        if (column >= *reinterpret_cast<const unsigned*>(descriptor)) descriptor += 32;
    constexpr unsigned curves = METADATA_VARIANT == 1 ? 256 : 4;
    return {unsigned((descriptor - blob) / 32) * curves,
            METADATA_VARIANT == 1 ? *reinterpret_cast<const float*>(descriptor + 12) : 1.f};
}
__device__ __forceinline__ void format_copy(std::uint8_t* target, const std::uint8_t* blob,
                                           int column, int tile, int lane) {
    packed_group_copy<FORMAT_RAW_BYTES>(target, blob + RAW_OFFSET, column, tile, lane);
}
__device__ __forceinline__ void format_emit_lut(unsigned* target, const std::uint8_t* raw,
    const std::uint8_t* blob, FormatContext context, int, int, int group, int pair) {
    const unsigned packed = reinterpret_cast<const unsigned short*>(raw)[group];
    const unsigned curve = METADATA_VARIANT == 1 ? packed >> 8 : packed >> 14;
    const float scale = METADATA_VARIANT == 1
        ? __fmul_rn(float(static_cast<signed char>(packed & 255)), context.global_scale)
        : __half2float(__ushort_as_half((packed & 0x3fff) << 1));
    const auto* table = reinterpret_cast<const float*>(blob + RAW_OFFSET +
        std::int64_t(OUTPUT_N) * (INPUT_K / GI) * FORMAT_RAW_BYTES);
    unsigned result = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned code = (pair * 2 + i) ^ (1u << (BITS - 1));
        const float value = __fmul_rn(__ldg(table +
            (context.curve_base + curve) * (1u << BITS) + code), scale);
        unsigned bits;
        if constexpr (IS_BF16) bits = __bfloat16_as_ushort(__float2bfloat16_rn(value));
        else bits = __half_as_ushort(__float2half_rn(value));
        result |= bits << (16 * i);
    }
    target[pair] = result;
}
