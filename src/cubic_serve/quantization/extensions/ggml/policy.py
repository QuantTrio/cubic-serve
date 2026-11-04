"""One portable baseline tile; no model, topology or device tuning table."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

from cubic_serve.quantization.common.parameters import PackedLinearParameters, PackedMoEParameters

# Q8_1 activation arithmetic opts into the existing independently audited
# local route/shared weighting followed by one TP completion. WNA16 retains
# its original rounding boundaries until that composition is qualified.
LOCAL_WEIGHTED_TP_COMPLETION = True
LOCAL_WEIGHTED_TP_COMPLETION_MODES = ("wna8",)

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.binding import PackedOperatorBinding


@dataclass(frozen=True)
class GGMLLinearParameters(PackedLinearParameters):
    generic_geometry: ClassVar[bool] = False
    fixed_block_n: int = 64
    fixed_block_k: int = 64
    fixed_stages: int = 1
    embedding_block_k: int = 256
    vector_rows: int = -1


@dataclass(frozen=True)
class GGMLMoEParameters(PackedMoEParameters):
    generic_geometry: ClassVar[bool] = False
    fixed_block_n: int = 64
    fixed_block_k: int = 64
    fixed_stages: int = 1
    vector_rows: int = -1


PARAMETERS = {"linear": GGMLLinearParameters, "moe": GGMLMoEParameters}
FORBIDDEN_OVERRIDES: dict[str, tuple[str, ...]] = {}


def bind_arithmetic(
    operators: PackedLinearOperators, x: Tensor, weight: QuantizedMatrix,
    params: GGMLLinearParameters | GGMLMoEParameters,
    bind: Callable[[str, int | None], PackedOperatorBinding], *, default: str = "auto",
    routed: bool = False, experts: int = 1, input_top_k: int = 1,
    routing_top_k: int = 1, group: tuple[QuantizedMatrix, ...] = (),
) -> PackedOperatorBinding:
    """Declare the selected reduction; only audit anchors may force its replay.

    Normal binding keeps its existing selector and native kernel. The ordered
    Q8 grid has the original MMVQ FMA/XOR tree, so it shares the vector recipe.
    MMQ has a distinct scale/reduction schedule and an explicit route-row tile.
    No runtime replay reads this metadata or the audit context.
    """
    from cubic_serve.quantization.common.small_rows_cost import (
        audited_row_recipe,
        record_row_recipe,
    )

    key = ("ggml.row-arithmetic.v1", routed, x.dtype, weight.encoding,
           weight.shape, weight.partitions, experts, input_top_k, routing_top_k,
           tuple((w.encoding, w.shape, w.partitions) for w in group), params)
    selected = audited_row_recipe(key, x.shape[0])
    tile = 8 if routed else 16
    recipes = {"vector": ("ggml-mmvq-v1", 0, 0)}
    if not routed:
        recipes["mma"] = ("ggml-full32-v1", 8, 8)
    algorithm = default
    matrix_rows = None
    if selected is not None:
        if (len(selected) == 3 and selected[0] == "ggml-mmq-v1"
                and type(selected[1]) is int and selected[1] in (tile, 2 * tile)
                and type(selected[2]) is int and selected[2] == 0):
            algorithm, matrix_rows = "matrix", selected[1]
        else:
            modes = [mode for mode, recipe in recipes.items() if selected == recipe]
            if len(modes) != 1:
                raise ValueError(
                    "GGML audit recipe differs from the declared reduction and row tile"
                )
            algorithm = modes[0]
    result = bind(algorithm, matrix_rows)
    operator = result._launches[-1].operator_id
    if operator == "cubic.cuda.ggml.llama.mmvq.v1":
        recipe = recipes["vector"]
    elif operator == "cubic.cuda.ggml.llama.mmq.v1":
        if result.row_tile_size not in (tile, 2 * tile):
            raise ValueError("GGML matrix binding lacks its actual compiled row tile")
        recipe = ("ggml-mmq-v1", result.row_tile_size, 0)
    elif operator == "cubic.cuda.ggml.q8_mma.v1" and not routed:
        recipe = recipes["mma"]
    else:
        raise ValueError("GGML binding lacks a declared arithmetic recipe")
    if selected is not None and recipe != selected:
        raise ValueError("GGML audit binding did not retain the requested reduction")
    record_row_recipe(operators, key, x.shape[0], recipe)
    return result


def validate(operation: str, values: object) -> tuple[tuple[str, object], ...]:
    if not isinstance(values, dict):
        raise ValueError("GGML parameters must be an object")
    defaults = asdict(PARAMETERS[operation]())
    if "vector_rows" in values:
        if type(values["vector_rows"]) is not int or values["vector_rows"] not in (-1, 0, 1, 4, 16):
            raise ValueError("invalid GGML vector row tile")
        defaults["vector_rows"] = values["vector_rows"]
    row_tile = max(0, defaults["vector_rows"])
    defaults["fixed_block_m"] = row_tile or 16
    defaults["fixed_block_n"] = 4 if row_tile else 64
    defaults["fixed_block_k"] = 256 if row_tile else 64
    if any(key not in defaults or type(value) is not type(defaults[key])
           or value != defaults[key] for key, value in values.items()):
        raise ValueError("GGML portable baseline does not implement alternate tile controls")
    return tuple(sorted(defaults.items()))


def select_vector_rows(requested: int, *, input_rows: int, input_top_k: int = 1,
                       routing_top_k: int | None = None) -> int:
    """Low-row vector baseline; larger batches retain the matrix baseline.

    Dense work below one 16-row matrix tile uses the vector primitive. Routed
    work has sparse expert grids and uses the four-column vector CTA only up
    to four source rows; larger routes retain the shared matrix baseline.
    This is a conservative geometry rule, not a claimed tuned crossover.
    Explicit tiles remain available for qualification; serving never probes.
    """
    if requested != -1:
        return requested
    source_rows = input_rows * input_top_k
    if routing_top_k is not None:
        source_rows = (source_rows + routing_top_k - 1) // routing_top_k
    return 1 if source_rows <= (16 if routing_top_k is None else 4) else 0


def supports_compute_mode(encoding: QuantizationScheme, operation: str, mode: str) -> bool:
    from .checkpoint import column_offset, kind

    return (mode == "wna8" and operation in ("linear", "moe")
            and int(kind(encoding)) in (8, 14, 20, 21, 23)
            and column_offset(encoding) == 0)


def routed_output_rows_per_warp(
    *, n: int, routes: int, sms: int, max_threads_per_sm: int,
) -> int:
    """Reuse four output rows only when the grid supplies two thread waves.

    The vector CTA has four warps. Four accumulators halve its CTA count
    relative to two; an underfilled grid loses latency hiding. One resident
    thread wave plus headroom for a partially occupied bucket protects small
    projections without model, expert-layout or device-generation tables.
    This does not change the per-output dot or XOR reduction order.
    """
    import math

    if min(n, routes, sms, max_threads_per_sm) <= 0:
        raise ValueError("GGML vector reuse requires positive geometry/resources")
    four_row_ctas = math.ceil(n / 16) * routes
    return 4 if four_row_ctas * 128 >= 2 * sms * max_threads_per_sm else 2


def matrix_costs(*, m: int, n: int, k: int, row_bytes: int,
                 sms: int, l2_bytes: int) -> tuple[float, float]:
    """Compare normalized serial work, not model names or public concurrency.

    The vector kernel reads each original weight row once per activation;
    MMQ shares decoded rows across a 16-row tile. The latter has a longer
    dependent load/scale/MMA chain, especially below one SM work wave.
    Coefficients conservatively fit paired projection measurements; a 50%
    margin protects the crossover and neighboring small projections. These
    are relative work units, never reported as measured execution times.
    """
    if min(m, n, k, row_bytes, sms, l2_bytes) <= 0:
        raise ValueError("GGML matrix pricing requires positive geometry/resources")
    import math

    # An isolated weight set beyond L2 loses the vector kernel's repeated-read
    # cache benefit. Treating smaller sets as cached is conservative in a model.
    cache_penalty = 4.0 if n * row_bytes > l2_bytes else 1.0
    vector = math.ceil(n / 4) * m / sms * cache_penalty
    matrix = 96.0 * max(1.0, math.ceil(n / 64) * math.ceil(m / 16) / sms)
    return k * vector, k * matrix


def select_matrix(*, m: int, n: int, k: int, row_bytes: int,
                  sms: int, l2_bytes: int) -> bool:
    if m == 1:
        return False
    vector, matrix = matrix_costs(m=m, n=n, k=k, row_bytes=row_bytes,
                                  sms=sms, l2_bytes=l2_bytes)
    return matrix * 1.5 < vector


def select_register_matrix_transport(
    *, m: int, n: int, k: int, row_tile: int, sms: int,
    shared_resident: int, register_resident: int,
    register_group_work: float = 3.4, register_wave_work: float = 0.6,
    shared_tile_work: float = 22.0, shared_wave_work: float = 6.65,
    margin: float = 1.1,
) -> bool:
    """Price dense Q8 matrix transport without changing its arithmetic.

    Register transport pays per original 32-value group; shared transport
    prepares complete 256-value tiles, including a partial final tile. Actual
    compiled-function residency and SM count determine the supplied CTA waves.
    The fitted coefficients are relative work units, not predicted microseconds.
    Paired projections and held-out shapes support a 10% conservative margin;
    this rule does not assert device qualification or cover routed expert grids.
    Both paths use the original codes, scale storage and ordered FP32 FMA.
    """
    import math

    if min(m, n, k, row_tile, sms, shared_resident, register_resident) <= 0 or k % 32:
        raise ValueError("GGML transport pricing requires positive complete geometry/resources")
    coefficients = (register_group_work, register_wave_work, shared_tile_work, shared_wave_work)
    if any(not math.isfinite(c) or c < 0 for c in coefficients):
        raise ValueError("GGML transport work must be finite and nonnegative")
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("GGML transport margin must be finite and at least one")
    tiles = math.ceil(m / row_tile) * math.ceil(n / 64)
    register = k / 32 * (
        register_group_work
        + register_wave_work * max(1.0, tiles / (sms * register_resident))
    )
    shared = math.ceil(k / 256) * (
        shared_tile_work + shared_wave_work * max(1.0, tiles / (sms * shared_resident))
    )
    return margin * register < shared


def prefer_wider_matrix_rows(
    *, m: int, n: int, k: int, row_bytes: int, experts: int,
    mean_rows: float | None, sms: int, l2_bytes: int,
    baseline_rows: int, wider_rows: int, baseline_resident: int,
    wider_resident: int, baseline_local: int, wider_local: int,
    dot_byte_work: float = 0.02415, margin: float = 1.1,
) -> bool:
    """Price original-block reuse only beyond the complete weight-owner L2 set.

    Packed reads and integer-dot work remain distinct: halving weight traffic
    cannot erase the dot. Actual compiled residency prices incomplete waves;
    local arrays pay their explicit per-thread read/write transport. The routed
    tile bound uses the existing smooth nonempty-owner plus mean/tile prior,
    while device counts remain authoritative at execution. This rule never
    chooses vector versus matrix or changes a checkpoint expert owner.

    One nonnegative byte/dot ratio fits paired original-format operators. It is
    a conservative ordering rule, not a latency predictor or device qualification.
    Cache-resident weights, extra dense padding, new spills, insufficient work,
    or reduced resident row storage retain the existing baseline.
    """
    import math

    if min(m, n, k, row_bytes, experts, sms, l2_bytes, baseline_rows,
           wider_rows, baseline_resident, wider_resident) <= 0:
        raise ValueError("GGML row reuse requires positive geometry/resources")
    if min(baseline_local, wider_local) < 0:
        raise ValueError("GGML local storage cannot be negative")
    if (not math.isfinite(dot_byte_work) or dot_byte_work < 0
            or not math.isfinite(margin) or margin < 1):
        raise ValueError("GGML row reuse work and margin must be finite and nonnegative")
    if mean_rows is not None and (not math.isfinite(mean_rows) or mean_rows <= 0):
        return False
    if (wider_rows != 2 * baseline_rows or experts * n * row_bytes <= l2_bytes
            or wider_local > baseline_local
            or wider_resident * wider_rows < baseline_resident * baseline_rows):
        return False
    if mean_rows is None:
        if experts != 1:
            return False
        if math.ceil(m / wider_rows) * wider_rows > math.ceil(m / baseline_rows) * baseline_rows:
            return False

    def work(rows: int, resident: int, local: int) -> tuple[float, float]:
        tiles = (math.ceil(m / rows) if mean_rows is None else
                 experts * (-math.expm1(-mean_rows) + mean_rows / rows))
        columns = math.ceil(n / 64)
        ctas = tiles * columns
        wave = ctas / (sms * resident)
        traffic = tiles * n * row_bytes + ctas * 128 * local * 2
        dots = tiles * rows * columns * 64 * k
        return traffic * math.ceil(wave) / wave + dot_byte_work * dots, ctas

    baseline, _ = work(baseline_rows, baseline_resident, baseline_local)
    wider, ctas = work(wider_rows, wider_resident, wider_local)
    return ctas >= sms * wider_resident and margin * wider < baseline


def select_pipelined_q8_matrix(
    *, m: int, n: int, k: int, sms: int, matrix_shared: bool,
    matrix_resident: int, pipeline_resident: int,
    shared_base: float = 12.880152518473334,
    shared_wave_work: float = 3.16461460255499,
    register_group_work: float = 1.0024498362418026,
    pipeline_base: float = 6.34335674506664,
    pipeline_stage_work: float = 0.4072517711815218,
    pipeline_wave_work: float = 1.394937371207912,
    margin: float = 1.1,
) -> bool:
    """Compare original MMQ with the format's pipelined full-group reduction.

    Shared MMQ pays complete resident CTA waves per 256-value tile; its
    register variant pays per 32-value group. The pipeline pays its prologue,
    512-value stages and fractional supplied waves. A partially filled device
    still pays for one CTA on each active SM. Actual compiled residency keeps
    resource pressure in the price. No runtime timing, model or SKU key exists.

    Nonnegative relative work coefficients fit eight matched core observations;
    four held-out geometries preserve every selected improvement. Their units
    are comparative, not serving-time predictions or hardware qualification.
    A 10% margin protects the measured near-crossover shared path and efficient
    register paths. The selected reduction is declared by bind_arithmetic.
    """
    import math

    if min(m, n, k, sms, matrix_resident, pipeline_resident) <= 0 or k % 32:
        raise ValueError("Q8 pipeline pricing requires positive complete geometry/resources")
    coefficients = (shared_base, shared_wave_work, register_group_work,
                    pipeline_base, pipeline_stage_work, pipeline_wave_work)
    if any(not math.isfinite(c) or c < 0 for c in coefficients):
        raise ValueError("Q8 pipeline work must be finite and nonnegative")
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("Q8 pipeline margin must be finite and at least one")
    if k <= 512:
        return False
    wave = math.ceil(n / 64) * math.ceil(m / 16) / (sms * matrix_resident)
    matrix = (shared_base + shared_wave_work * math.ceil(k / 256) * math.ceil(wave)
              if matrix_shared else register_group_work * (k / 32) * max(1.0, wave))
    stages = math.ceil(k / 512)
    supplied = math.ceil(n / 16) * math.ceil(m / 8) / (sms * pipeline_resident)
    pipeline = (pipeline_base + pipeline_stage_work * stages
                + pipeline_wave_work * stages * max(1 / pipeline_resident, supplied))
    return margin * pipeline < matrix


def select_routed_matrix(*, mean_rows: float | None, experts: int, n: int, k: int,
                         row_bytes: int, sms: int, l2_bytes: int,
                         decode_reuse_work: float = 5.0) -> bool:
    """Price reuse within original experts, including partial 8-row tiles.

    The existing routing prior supplies rank-local expected rows, not traffic
    remapping. The smooth Poisson nonempty-owner estimate plus lambda/8
    conservatively overestimates tile work. Encoded bytes per logical value
    price the loader/scale traffic: direct Q8 dots need more reuse before
    MMA pays off than lower-bit IQ dots. A 50% margin protects neighboring
    sparse workloads and the short final K tile. Coefficients fit paired
    complete-block measurements; these are work units, never microseconds.
    Byte traffic alone underprices repeated compressed-code decoding. A squared
    Poisson repeated-owner feature adds that arithmetic work separately from
    the cache penalty. Its conservative fitted coefficient preserves sparse
    vector work and leaves uncompressed Q8 pricing unchanged. It is a smooth
    regression feature, not a measured time or an exact route distribution.
    Device counts remain authoritative and original expert owners never move.
    """
    import math

    if mean_rows is None or not math.isfinite(mean_rows) or mean_rows <= 0:
        return False
    if min(experts, n, k, row_bytes, sms, l2_bytes) <= 0:
        raise ValueError("GGML routed pricing requires positive geometry/resources")
    if not math.isfinite(decode_reuse_work) or decode_reuse_work < 0:
        raise ValueError("GGML routed decode work must be finite and nonnegative")
    penalty = 4.0 if experts * n * row_bytes > l2_bytes else 1.0
    repeated = max(0.0, -math.expm1(-mean_rows) - mean_rows * math.exp(-mean_rows))
    expansion = max(0.0, k / row_bytes - 1.0)
    decode = decode_reuse_work * expansion * repeated**2
    vector = math.ceil(n / 4) * experts * mean_rows / sms * (penalty + decode)
    tiles = experts * (-math.expm1(-mean_rows) + mean_rows / 8)
    loader_cost = 40.0 + 64.0 * row_bytes / k
    matrix = loader_cost * max(1.0, math.ceil(n / 64) * tiles / sms)
    return 1.5 * matrix < vector
