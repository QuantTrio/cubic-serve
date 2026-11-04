"""Independent scalar format probes and eager multi-row/MTP identity evidence.

This tool does not qualify CUDA Graph execution or report serving throughput.
Run with PYTHONPATH=src. Payload or execution errors are distinct from numerical
assertions; an incomplete harness is never recorded as a floating-point pass.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import fields
from pathlib import Path

import torch

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.models.builtin import parse_builtin_model_config_json
from cubic_serve.models.quantization import WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedCheckpoint
from cubic_serve.models.qwen3_5.weights import logical_quantized_weights as dense_weights
from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cubic_serve.models.qwen3_5_moe.weights import logical_quantized_weights as moe_weights
from cubic_serve.ops.eager.quantization import QuantizedMatrix, decode_matrix
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope
from cubic_serve.runtime.quantized_reference import load_quantized_reference


def scalar_weight(payload, encoding, shape, partitions, row, column):
    """Scalar arithmetic from serialized codes, independent of tensor decoder math."""
    fmt = encoding.format
    if fmt is WeightFormat.AWQ:
        lane = (0, 4, 1, 5, 2, 6, 3, 7)[row % 8]
        code = (int(payload["qweight"][column, row // 8]) >> (lane * 4)) & 15
        zero = (int(payload["qzeros"][column // encoding.group_in, row // 8]) >> (lane * 4)) & 15
        scale = float(payload["scales"][column // encoding.group_in, row])
        return (code - zero) * scale, abs(scale)
    if fmt is WeightFormat.FP8:
        scale = float(payload["weight_scale"][row, 0])
        return float(payload["weight"][row, column]) * scale, abs(scale)
    if fmt is WeightFormat.NVFP4:
        code = (int(payload["weight_packed"][row, column // 2]) >> (4 * (column % 2))) & 15
        scale = float(payload["weight_scale"][row, column // 16]) / float(
            payload["weight_global_scale"][0]
        )
        value = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[code & 7]
        return value * (-1 if code & 8 else 1) * scale, abs(scale)
    bit = column * encoding.bits
    data = bytes(payload["weight_packed"][row, bit // 8 : bit // 8 + 2].tolist())
    code = (int.from_bytes(data, "little") >> (bit % 8)) & ((1 << encoding.bits) - 1)
    if encoding.bits == 1:
        code = code * 2 - 1
    elif code == 1 << (encoding.bits - 1):
        code = 0
    elif code >= 1 << (encoding.bits - 1):
        code -= 1 << encoding.bits
    part, offset = 0, 0
    for index, count in enumerate(partitions or (shape[0],)):
        if row < offset + count:
            part = index
            break
        offset += count
    r, c = row // encoding.group_out, column // encoding.group_in

    def half(value):
        return struct.unpack("e", struct.pack("e", value))[0]

    if encoding.metadata == "int8-scale-int4-ab-global-fp32":
        scale = int(payload["weight_scale"][r, c]) * float(payload["weight_scale_global"][part])
        ab = int(payload["weight_ab"][r, c])
        a, b = ab & 15, ab >> 4
        a, b = a - 16 if a >= 8 else a, b - 16 if b >= 8 else b

        # Exporter coefficient storage rounds through FP32 then FP16.
        def f32(value):
            return struct.unpack("f", struct.pack("f", value))[0]

        a = half(f32(1.0 + f32(a * float(payload["weight_a_global"][part]))))
        b = half(f32(b * float(payload["weight_b_global"][part])))
    elif encoding.metadata == "e5m9-scale-curve2-tensor-fp16":
        metadata = int(payload["weight_metadata"][r, c])
        scale = struct.unpack("e", struct.pack("H", (metadata & 0x3FFF) << 1))[0]
        curve = metadata >> 14
        a, b = (
            float(payload["weight_curve_a"][part, curve]),
            float(payload["weight_curve_b"][part, curve]),
        )
    else:
        scale, a, b = (
            float(payload[key][r, c]) for key in ("weight_scale", "weight_a", "weight_b")
        )
    t = abs(code) / max(1, (1 << (encoding.bits - 1)) - 1)
    value = (
        code * scale
        if encoding.bits == 1
        else math.copysign(1.0, code) * scale * t * (a + t * (b + t * (1 - a - b)))
    )
    return value, abs(scale) * (1 + abs(a) + abs(b) + abs(1 - a - b))


def scalar_probes(args):
    root = Path(args.model)
    config = parse_builtin_model_config_json(
        (root / "config.json").read_bytes(), compute_dtype=args.dtype
    )
    logical = moe_weights(config) if isinstance(config, Qwen3_5MoeConfig) else dense_weights(config)
    plan = QuantizedCheckpoint.admit(root, logical, verify_payloads=False)
    selected = {}
    for weight in plan.weights:
        if weight.encoding is not None:
            key = (
                weight.logical.partition,
                weight.logical.operation,
                weight.encoding,
                len(weight.logical.output_partitions),
            )
            selected.setdefault(key, weight)
    reports = []
    for weight in selected.values():
        payload = plan.read(weight)
        logical = weight.logical
        matrix = QuantizedMatrix(payload, weight.encoding, logical.shape, logical.output_partitions)
        alignment = math.lcm(
            weight.encoding.group_out, 8 if weight.encoding.format is WeightFormat.AWQ else 1
        )
        n, k = logical.shape
        rows = sorted(
            {
                0,
                n // 2,
                n - 1,
                *[
                    sum(logical.output_partitions[:i])
                    for i in range(1, len(logical.output_partitions))
                ],
            }
        )
        columns = sorted(
            {
                0,
                1,
                k // 2,
                k - 1,
                min(k - 1, weight.encoding.group_in - 1),
                min(k - 1, weight.encoding.group_in),
            }
        )
        maximum, probes = 0.0, 0
        for row in rows:
            start = row // alignment * alignment
            part = matrix.rows(start, start + alignment)
            actual = decode_matrix(
                QuantizedMatrix(
                    {key: value.to(args.device) for key, value in part.payload.items()},
                    part.encoding,
                    part.shape,
                    part.partitions,
                )
            ).cpu()
            for column in columns:
                expected, scale_bound = scalar_weight(
                    payload, weight.encoding, logical.shape, logical.output_partitions, row, column
                )
                error = abs(float(actual[row - start, column]) - expected)
                # At most a small fixed chain of FP32 scalar operations; this
                # is a format probe bound, never a GEMM reduction allowance.
                bound = 16 * 2**-23 * max(scale_bound, abs(expected), 1e-30)
                if not math.isfinite(error) or error > bound:
                    raise AssertionError(
                        f"scalar format mismatch {logical.name}[{row},{column}]: {error} > {bound}"
                    )
                maximum = max(maximum, error)
                probes += 1
        reports.append(
            dict(
                weight=logical.name,
                encoding=str(weight.encoding),
                probes=probes,
                max_abs_error=maximum,
            )
        )
        print(json.dumps(reports[-1]), flush=True)
    return dict(inventory=plan.inventory.inventory_digest, scalar_probes=reports)


def snapshot(step, row):
    values = {
        "logits": step.logits[row : row + 1].cpu().clone(),
        "hidden": step.hidden_states[row : row + 1].cpu().clone(),
    }
    layers = step.cache.layers if hasattr(step.cache, "layers") else (step.cache.layer,)
    for index, layer in enumerate(layers):
        for field in fields(layer):
            value = getattr(layer, field.name)
            if value is not None:
                values[f"state.{index}.{field.name}"] = value[row : row + 1].cpu().clone()
    return values


def equal(actual, expected, label):
    if actual.keys() != expected.keys():
        raise RuntimeError("audit harness snapshot fields differ")
    for name, value in actual.items():
        if not torch.equal(value, expected[name]):
            error = (value.float() - expected[name].float()).abs().max().item()
            raise AssertionError(f"{label} {name}: eager row identity differs; max_abs={error}")


def eager_rows(args):
    if args.mode == "native":
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
        from cubic_serve.runtime.quantized import load_quantized_model

        model, plan = load_quantized_model(
            args.model,
            device=args.device,
            dtype=args.dtype,
            operators=PackedLinearOperators(OperatorConfig(activation_dtype=args.activation_dtype)),
        )
    else:
        model, plan = load_quantized_reference(args.model, device=args.device, dtype=args.dtype)
    references, reports = {}, []
    with torch.inference_mode(), ReferenceExecutionContext():
        for count in (1, *[n for n in args.rows if n != 1]):
            for permutation in (False, True):
                corpus = [100 + (i % 2) * 100 for i in range(count)]
                if permutation:
                    corpus.reverse()
                # Count=1 runs both corpus members independently.
                if count == 1 and permutation:
                    corpus = [200]
                ids = torch.tensor([corpus], dtype=torch.int64, device=args.device).t().contiguous()
                cache = None
                for step_index in range(2):
                    start = time.monotonic()
                    result = model.forward(ids + step_index, cache=cache)
                    cache = result.cache
                    for row, token in enumerate(corpus):
                        key = (token, step_index)
                        current = snapshot(result, row)
                        if not all(bool(v.isfinite().all()) for v in current.values()):
                            raise AssertionError("nonfinite eager target output/state")
                        if count == 1:
                            references[key] = current
                        else:
                            equal(
                                current,
                                references[key],
                                f"M{count} permutation={permutation} row={row}",
                            )
                    positions = torch.full(
                        (3, count, 1), step_index + 1, dtype=torch.int64, device=args.device
                    )
                    mtp = model.mtp_forward(
                        ids + step_index + 1, result.hidden_states, position_ids=positions
                    )
                    for row, token in enumerate(corpus):
                        key = (token, step_index, "mtp")
                        current = snapshot(mtp, row)
                        if not all(bool(v.isfinite().all()) for v in current.values()):
                            raise AssertionError("nonfinite eager MTP output/state")
                        if count == 1:
                            references[key] = current
                        else:
                            equal(current, references[key], f"M{count} MTP row={row}")
                    reports.append(
                        dict(
                            rows=count,
                            permutation=permutation,
                            step=step_index,
                            seconds=time.monotonic() - start,
                            target_and_mtp="bit-exact" if count != 1 else "reference",
                        )
                    )
                    print(json.dumps(reports[-1]), flush=True)
    return dict(
        inventory=plan.inventory.inventory_digest,
        eager_identity=reports,
        excludes=[
            "CUDA Graph",
            "padding poison",
            "distributed collectives",
            "speculative algorithm",
            "serving throughput",
        ],
    )


def native_oracle(args):
    from cubic_serve.guardrail.quantization import QuantizedCausalReplay, QuantizedOperatorObserver
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.eager.quantization import QuantizedReferenceOperators
    from cubic_serve.runtime.quantized import load_quantized_model

    native = PackedLinearOperators(OperatorConfig(activation_dtype=args.activation_dtype))
    reference = QuantizedReferenceOperators(128)
    model, plan = load_quantized_model(
        args.model, device=args.device, dtype=args.dtype, operators=native
    )
    reports = []
    with torch.inference_mode(), ReferenceExecutionContext():
        for count in args.rows:
            ids = torch.tensor([[100 + (i % 2) * 100] for i in range(count)], device=args.device)
            model.default_operators = native
            prefix = model.forward(ids)
            for kind in ("target", "mtp"):

                def execute(kind=kind, ids=ids, prefix=prefix, count=count):
                    if kind == "target":
                        return model.forward(ids + 1, cache=prefix.cache)
                    positions = torch.ones((3, count, 1), dtype=torch.int64, device=args.device)
                    return model.mtp_forward(ids + 1, prefix.hidden_states, position_ids=positions)

                observer = QuantizedOperatorObserver(native)
                model.default_operators = observer
                actual = execute()
                # No reference math or diagnostic host read occurred in the
                # observed forward. Single rank: its complete event schedule
                # is now frozen before unrestricted independent execution.
                torch.cuda.synchronize()
                model.default_operators = reference
                expected = execute()
                raw_delta = float((actual.logits.float() - expected.logits.float()).abs().max())
                del expected
                replay = QuantizedCausalReplay(observer.events, reference)
                model.default_operators = replay
                witnessed = execute()
                replay.finish()
                for row in range(count):
                    equal(
                        snapshot(witnessed, row), snapshot(actual, row), f"M{count} {kind} causal"
                    )
                record = dict(
                    rows=count,
                    kind=kind,
                    events=len(observer.events),
                    linear_events=len(replay.reports),
                    independent_logits_max_abs=raw_delta,
                    causal_outputs_and_state="bit-exact",
                    linear_allowance=(
                        "full-output/full-K fp32-dot-and-format-scales-plus-output-rne"
                    ),
                    max_linear_error=max(r["max_abs_error"] for r in replay.reports),
                )
                reports.append(record)
                print(json.dumps(record), flush=True)
                model.default_operators = native
                del replay, observer, actual, witnessed
            del prefix
    return dict(
        inventory=plan.inventory.inventory_digest,
        native_oracle=reports,
        excludes=[
            "CUDA Graph",
            "distributed collectives",
            "speculative acceptance algorithm",
            "serving throughput",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation-dtype", choices=("auto", "fp32"), default="auto")
    parser.add_argument(
        "--mode", choices=("scalar", "eager", "native", "native-oracle"), default="scalar"
    )
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 8, 33])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = dict(
        model=args.model,
        dtype=args.dtype,
        activation_dtype=args.activation_dtype,
        mode=args.mode,
        status="harness_failure",
    )
    code = 1
    try:
        configure_reference_math()
        operation = (
            scalar_probes
            if args.mode == "scalar"
            else native_oracle
            if args.mode == "native-oracle"
            else eager_rows
        )
        with operator_strategy_scope(
            OperatorConfig(activation_dtype=args.activation_dtype), device=args.device, model=""
        ):
            report.update(operation(args))
        report["status"], code = "pass", 0
    except AssertionError as error:
        report.update(status="numerical_failure", error=str(error))
        raise
    except (torch.OutOfMemoryError, torch.cuda.CudaError) as error:
        report.update(status="execution_failure", error=str(error))
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
