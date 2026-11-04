"""Sampled CPU-FP64 audit of every unique real Qwen3.8 decode GEMM shape.

The CUDA candidate always computes the complete ``M x N`` output from the
complete real ``N x K`` checkpoint weight. The CPU oracle samples fixed batch
rows and output channels, but every sampled dot product covers all ``K`` terms.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from cubic_serve.guardrail.linear_artifact import (
    LINEAR_ARTIFACT_SEMANTICS,
    BF16LinearArtifactManifest,
    ReleaseBF16LinearArtifact,
    verify_bf16_linear_artifact_launch,
)
from cubic_serve.models import FullBF16WeightPolicy, seal_full_bf16_weight_set
from cubic_serve.ops.cuda.linear import BF16_LINEAR_OPERATOR_ID, BF16LinearPrimitive

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_DECODE_WEIGHT = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)|"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
_FP32_U, _FP64_U, _BF16_U = 2.0**-24, 2.0**-53, 2.0**-8
_DEFAULT_M_VALUES = (1, 2, 4, 8, 16, 32, 64)


def _manifest(primitive: BF16LinearPrimitive) -> BF16LinearArtifactManifest:
    identity = primitive.project_artifact()
    approval = ReleaseBF16LinearArtifact(
        "release:qwen3.8-linear-sampled-witness",
        BF16_LINEAR_OPERATOR_ID,
        identity.artifact_digest,
        identity.launch_specialization_digest,
        LINEAR_ARTIFACT_SEMANTICS,
    )
    return BF16LinearArtifactManifest(approval.release_id, (approval,))


def _indices(length: int, count: int) -> tuple[int, ...]:
    count = min(length, count)
    if count == 1:
        return (0,)
    return tuple(index * (length - 1) // (count - 1) for index in range(count))


def _m_values(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("M values must be comma-separated integers") from error
    if not result or len(set(result)) != len(result) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be unique positive integers")
    return result


def _index_description(indices: tuple[int, ...], length: int) -> object:
    if indices == tuple(range(length)):
        return {"all": True, "count": length}
    return indices


def _decode_shapes(
    checkpoint: Path, weight_map: dict[str, str]
) -> tuple[tuple[tuple[int, int], tuple[str, ...]], ...]:
    names = sorted(name for name in weight_map if _DECODE_WEIGHT.fullmatch(name))
    if "lm_head.weight" in weight_map:
        names.append("lm_head.weight")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_shard[weight_map[name]].append(name)
    by_shape: dict[tuple[int, int], list[str]] = defaultdict(list)
    for shard, shard_names in sorted(by_shard.items()):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            for name in shard_names:
                shape = tuple(source.get_slice(name).get_shape())
                if len(shape) != 2 or min(shape) <= 0:
                    raise RuntimeError(f"decode weight {name!r} is not a non-empty matrix")
                by_shape[(shape[0], shape[1])].append(name)
    return tuple((shape, tuple(sorted(items))) for shape, items in sorted(by_shape.items()))


def _sampled_fp64(
    x: torch.Tensor,
    weight: torch.Tensor,
    actual: torch.Tensor,
    input_rows: tuple[int, ...],
    output_rows: tuple[int, ...],
) -> tuple[int, int, float, float]:
    x64 = x[list(input_rows)].to(torch.float64)
    weight64 = weight[list(output_rows)].to(torch.float64)
    anchor = torch.mm(x64, weight64.t())
    sum_abs = torch.mm(x64.abs(), weight64.abs().t())
    k = x.shape[1]
    gamma32 = k * _FP32_U / (1.0 - k * _FP32_U)
    gamma64 = k * _FP64_U / (1.0 - k * _FP64_U)
    allowed = sum_abs.div(1.0 - gamma64)
    allowed.mul_((gamma32 + gamma64) * (1.0 + _BF16_U))
    allowed.add_(anchor.abs(), alpha=_BF16_U)
    observed = actual[list(input_rows)][:, list(output_rows)].to(torch.float64)
    difference = observed.sub(anchor).abs()
    finite = torch.isfinite(observed) & torch.isfinite(anchor) & torch.isfinite(allowed)
    violations = ~finite | (difference > allowed)
    return (
        int(violations.sum().item()),
        int((~finite).sum().item()),
        float(difference.masked_fill(~finite, 0).max().item()),
        float(allowed.masked_fill(~finite, 0).max().item()),
    )


def _emit(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1511)
    parser.add_argument("--weight")
    parser.add_argument("--m-values", type=_m_values, default=_DEFAULT_M_VALUES)
    parser.add_argument("--sample-input-rows", type=int, default=2)
    parser.add_argument("--sample-output-rows", type=int, default=8)
    args = parser.parse_args()
    if min(args.sample_input_rows, args.sample_output_rows) <= 0:
        parser.error("sample row counts must be positive")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shapes = _decode_shapes(args.checkpoint, weight_map)
    if args.weight is not None:
        selected = tuple(
            (shape, (args.weight,)) for shape, names in shapes if args.weight in names
        )
        if len(selected) != 1:
            parser.error("--weight must name exactly one enumerated decode GEMM")
        shapes = selected
    occurrence_count = sum(len(names) for _shape, names in shapes)
    _emit(
        {
            "kind": "coverage",
            "checkpoint": str(args.checkpoint.resolve()),
            "decode_gemm_occurrences": occurrence_count,
            "unique_shapes": len(shapes),
            "m_values": args.m_values,
            "candidate_scope": "complete MxNxK for one real representative per unique shape",
            "oracle_scope": "fixed sampled M rows x N rows; complete K reduction",
            "seed": args.seed,
        }
    )

    total_violations = 0
    full_oracle = True
    for shape_ordinal, ((n, k), names) in enumerate(shapes):
        shape_violations = 0
        weight_name = names[0]
        shard = args.checkpoint / weight_map[weight_name]
        with safe_open(shard, framework="pt", device="cpu") as source:
            weight_cpu = source.get_tensor(weight_name)
            if weight_cpu.dtype is not torch.bfloat16 or tuple(weight_cpu.shape) != (n, k):
                raise RuntimeError(f"checkpoint weight {weight_name!r} changed after header scan")
            weight = weight_cpu.to(args.device)
            named = ((weight_name, weight),)
            sealed = seal_full_bf16_weight_set(
                lambda values=named: values,
                policy=FullBF16WeightPolicy(expected_names=(weight_name,)),
            )
            for m in args.m_values:
                seed = args.seed + shape_ordinal * 100 + m
                generator = torch.Generator(device="cpu").manual_seed(seed)
                x_cpu = torch.randn((m, k), dtype=torch.bfloat16, generator=generator)
                x = x_cpu.to(args.device)
                output = torch.empty((m, n), dtype=torch.bfloat16, device=args.device)
                primitive = BF16LinearPrimitive(x, weight, output)
                structural = verify_bf16_linear_artifact_launch(
                    primitive=primitive,
                    x=x,
                    weight=weight,
                    output=output,
                    manifest=_manifest(primitive),
                )
                actual = output.detach().to(device="cpu", copy=True)
                input_rows = _indices(m, args.sample_input_rows)
                output_rows = _indices(n, args.sample_output_rows)
                full_oracle &= len(input_rows) == m and len(output_rows) == n
                violations, nonfinite, observed, allowed = _sampled_fp64(
                    x_cpu, weight_cpu, actual, input_rows, output_rows
                )
                total_violations += violations
                shape_violations += violations
                _emit(
                    {
                        "kind": "shape_result",
                        "status": "passed" if violations == 0 else "failed",
                        "shape_ordinal": shape_ordinal,
                        "shape_occurrences": len(names),
                        "representative_weight": weight_name,
                        "weight_bytes_digest": sealed.evidence.tensors[0].bytes_digest,
                        "m": m,
                        "n": n,
                        "k": k,
                        "seed": seed,
                        "candidate_full_shape": (m, n, k),
                        "candidate_output_elements": m * n,
                        "oracle_input_rows": _index_description(input_rows, m),
                        "oracle_output_rows": _index_description(output_rows, n),
                        "oracle_sampled_elements": len(input_rows) * len(output_rows),
                        "oracle_input_row_coverage": len(input_rows) / m,
                        "oracle_output_row_coverage": len(output_rows) / n,
                        "oracle_reduction_coverage": 1.0,
                        "out_of_bound_count": violations,
                        "nonfinite_count": nonfinite,
                        "max_observed_error": observed,
                        "max_allowed_error": allowed,
                        "artifact_digest": primitive.project_artifact().artifact_digest,
                        "structural_witness_digest": structural.witness_digest,
                    }
                )
                del actual, output, primitive, x, x_cpu
            del sealed, named, weight
        _emit(
            {
                "kind": "shape_summary",
                "status": "passed" if shape_violations == 0 else "failed",
                "shape_ordinal": shape_ordinal,
                "n": n,
                "k": k,
                "m_values": args.m_values,
                "shape_occurrences": len(names),
                "out_of_bound_count": shape_violations,
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
    _emit(
        {
            "kind": "summary",
            "status": "passed" if total_violations == 0 else "failed",
            "unique_shapes": len(shapes),
            "decode_gemm_occurrences": occurrence_count,
            "shape_m_cases": len(shapes) * len(args.m_values),
            "total_out_of_bound_count": total_violations,
            "claim": (
                "full-element CPU-FP64 enclosure"
                if full_oracle
                else "sampled CPU-FP64 enclosure only; not a full-element numerical audit"
            ),
        }
    )
    return 0 if total_violations == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
