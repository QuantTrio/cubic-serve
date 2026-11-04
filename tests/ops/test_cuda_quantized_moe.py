import os
import unittest

import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.models.quantization import ActivationFormat, QuantizationScheme, WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive


def storage_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@unittest.skipUnless(os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1", "explicit CUDA audit required")
class PackedMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_reference_math()
        torch.set_num_threads(1)

    def test_compact_route_metadata_ownership_and_poison(self):
        import triton

        from cubic_serve.ops.cuda.moe.fused import _compact_local_routes

        for experts, size in ((3, 1), (127, 31), (512, 160), (513, 256), (512, 257)):
            first = 7
            for mode in ("spread", "concentrated", "inactive"):
                logical = [first + (j % experts) for j in range(size)]
                if mode == "concentrated":
                    logical = [first] * size
                if mode == "inactive":
                    logical = [-1] * size
                else:
                    logical[::7] = [-1] * len(logical[::7])
                    logical[1::11] = [first + experts] * len(logical[1::11])
                # Changing the physical order must preserve the original ordinal
                # of each owned route, including holes between logical rows.
                for values in (logical, logical[::-1]):
                    ids = torch.tensor(values, device="cuda", dtype=torch.int64)
                    expected = [j for e in range(first, first + experts)
                                for j, value in enumerate(values) if value == e]
                    counts_cpu = [values.count(e) for e in range(first, first + experts)]
                    for bm in (8, 16, 32):
                        storage = torch.full((4 * experts + size + 1,), -777,
                                             device="cuda", dtype=torch.int32)
                        counts, starts, blocks, spare = storage[:4 * experts].chunk(4)
                        total = storage[4 * experts:4 * experts + 1]
                        rows = storage[4 * experts + 1:]
                        args = (ids, counts, starts, blocks, total, rows,
                                first, experts, size, bm, triton.next_power_of_2(experts),
                                triton.next_power_of_2(size))
                        _compact_local_routes[(1,)](*args, num_warps=8)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            _compact_local_routes[(1,)](*args, num_warps=8)
                        storage.fill_(-777)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertEqual(counts.tolist(), counts_cpu)
                        self.assertEqual(starts.tolist(),
                                         [sum(counts_cpu[:e]) for e in range(experts)])
                        block_counts = [(n + bm - 1) // bm for n in counts_cpu]
                        self.assertEqual(blocks.tolist(),
                                         [sum(block_counts[:e]) for e in range(experts)])
                        self.assertEqual(total.item(), sum(block_counts))
                        self.assertEqual(rows[:len(expected)].tolist(), expected)
                        self.assertTrue(bool((rows[len(expected):] == -777).all()))
                        self.assertTrue(bool((spare == -777).all()))

    def matrix(self, n, k, fmt, dtype, index):
        device = "cuda:0"
        if fmt == "plain":
            return torch.randn(n, k, dtype=dtype, device=device) * 0.05
        if fmt == "awq":
            return QuantizedMatrix(
                {
                    "qweight": torch.randint(
                        -(2**31), 2**31 - 1, (k, n // 8), dtype=torch.int32, device=device
                    ),
                    "qzeros": torch.full(
                        (k // 32, n // 8), 0x77777777, dtype=torch.int32, device=device
                    ),
                    "scales": torch.full((k // 32, n), 0.03125, dtype=torch.float16, device=device),
                },
                QuantizationScheme(WeightFormat.AWQ, 4, 32),
                (n, k),
            )
        if fmt == "fp8":
            return QuantizedMatrix(
                {
                    "weight": torch.randn(n, k, device=device).to(torch.float8_e4m3fn),
                    "weight_scale": torch.full((n, 1), 0.125, device=device),
                },
                QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN),
                (n, k),
            )
        if fmt == "nvfp4":
            return QuantizedMatrix(
                {
                    "weight_packed": torch.randint(
                        0, 256, (n, k // 2), dtype=torch.uint8, device=device
                    ),
                    "weight_scale": ((torch.arange(n * (k // 16), device=device) + index) % 17 + 40)
                    .to(torch.uint8)
                    .view(torch.float8_e4m3fn)
                    .reshape(n, k // 16),
                    "weight_global_scale": torch.tensor([8.0 + index], device=device),
                    "input_global_scale": torch.tensor([2.0 + index * 0.5], device=device),
                },
                QuantizationScheme(
                    WeightFormat.NVFP4, 4, 16, activation=ActivationFormat.NVFP4_LOCAL
                ),
                (n, k),
            )
        return QuantizedMatrix(
            {
                "weight_packed": torch.randint(
                    0, 256, (n, k // 2), dtype=torch.uint8, device=device
                ),
                "weight_scale": torch.ones(n, k // 32, dtype=torch.int8, device=device),
                "weight_ab": torch.full((n, k // 32), 0x23, dtype=torch.uint8, device=device),
                "weight_scale_global": torch.tensor([0.125], device=device),
                "weight_a_global": torch.tensor([0.125], device=device),
                "weight_b_global": torch.tensor([0.0625], device=device),
            },
            QuantizationScheme(
                WeightFormat.CUBIC, 4, 32, metadata="int8-scale-int4-ab-global-fp32"
            ),
            (n, k),
        )

    def assert_swiglu_enclosure(self, gate, up, actual):
        # PTX ex2.approx's documented per-input bound, propagated through
        # each prescribed low-precision rounding boundary. No global atol.
        dtype = gate.dtype
        if dtype is torch.bfloat16 and torch.cuda.get_device_capability(gate.device)[0] < 9:
            dtype = torch.float16
        x = gate.cpu().to(dtype)
        argument = -x.abs() * torch.tensor(1.4426950408889634, dtype=dtype)
        exact = torch.exp2(argument.double())
        error = exact * 2.0 ** -(7.0 if dtype is torch.bfloat16 else 9.9)
        error += torch.finfo(dtype).tiny if dtype is torch.bfloat16 else 2.0**-24
        lo, hi = (exact - error).clamp_min(0), exact + error
        # The exponential itself is already a low-precision value; bracket
        # by adjacent representable values before the rounded denominator.
        lo = torch.nextafter(lo.to(dtype), torch.full_like(x, -float("inf"))).clamp_min(0)
        hi = torch.nextafter(hi.to(dtype), torch.full_like(x, float("inf")))
        dlo, dhi = (1 + lo).float(), (1 + hi).float()
        nlo, nhi = torch.where(x >= 0, 1.0, lo.float()), torch.where(x >= 0, 1.0, hi.float())
        slo, shi = (nlo / dhi).to(dtype).double(), (nhi / dlo).to(dtype).double()
        a, b = x.double() * slo, x.double() * shi
        a, b = (
            torch.minimum(a, b).to(dtype).to(gate.dtype),
            torch.maximum(a, b).to(dtype).to(gate.dtype),
        )
        a, b = (
            (a.double() * up.cpu().double()).to(gate.dtype),
            (b.double() * up.cpu().double()).to(gate.dtype),
        )
        lower, upper = torch.minimum(a, b), torch.maximum(a, b)
        self.assertTrue(bool(((actual.cpu() >= lower) & (actual.cpu() <= upper)).all()))

    def test_cubic_storage_layout_bit_widths_and_metadata(self):
        from unittest.mock import patch

        cases = [(bits, "int8-scale-int4-ab-global-fp32") for bits in range(1, 9)]
        cases += [(5, "float32-scale-float16-ab"), (7, "e5m9-scale-curve2-tensor-fp16")]
        for bits, metadata in cases:
            with self.subTest(bits=bits, metadata=metadata):

                def matrix(n, k, fmt, dtype, index, *, bits=bits, metadata=metadata):
                    device = "cuda:0"
                    groups = (n, (k + 31) // 32)
                    payload = {
                        "weight_packed": torch.randint(
                            0, 256, (n, (k * bits + 7) // 8), dtype=torch.uint8, device=device
                        )
                    }
                    if metadata.startswith("int8"):
                        payload.update(
                            weight_scale=torch.full(
                                groups, index + 1, dtype=torch.int8, device=device
                            ),
                            weight_ab=torch.full(groups, 0x23, dtype=torch.uint8, device=device),
                            weight_scale_global=torch.tensor([0.03125], device=device),
                            weight_a_global=torch.tensor([0.125], device=device),
                            weight_b_global=torch.tensor([0.0625], device=device),
                        )
                    elif metadata.startswith("float32"):
                        payload.update(
                            weight_scale=torch.full(groups, 0.03125, device=device),
                            weight_a=torch.full(groups, 0.75, dtype=torch.float16, device=device),
                            weight_b=torch.full(groups, 0.125, dtype=torch.float16, device=device),
                        )
                    else:
                        codes = torch.arange(n * groups[1], device=device).reshape(groups) % 4
                        payload.update(
                            weight_metadata=((codes << 14) | 0x1800).to(torch.uint16),
                            weight_curve_a=torch.tensor(
                                [[0.5, 0.75, 1.0, 1.25]], dtype=torch.float16, device=device
                            ),
                            weight_curve_b=torch.tensor(
                                [[0.125, 0.0, -0.125, 0.25]], dtype=torch.float16, device=device
                            ),
                        )
                    return QuantizedMatrix(
                        payload,
                        QuantizationScheme(WeightFormat.CUBIC, bits, 32, metadata=metadata),
                        (n, k),
                    )

                with patch.object(self, "matrix", matrix):
                    self.audit_routed_formats(("cubic",), 128, 64)

    def test_routed_formats_padding_and_cross_m(self):
        self.audit_routed_formats(("plain", "awq", "fp8", "nvfp4", "cubic", "mixed"), 128, 64)

    def test_external_route_workspace_is_disjoint_and_bounded(self):
        for dtype in (torch.float16, torch.bfloat16):
            m, h, intermediate = 4, 128, 64
            x = torch.randn(m, h, device="cuda", dtype=dtype) * 0.05
            logits = torch.randn(m, 2, device="cuda", dtype=dtype)
            activity = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)
            active = torch.tensor([m], device="cuda", dtype=torch.int32)
            gate = tuple(self.matrix(intermediate, h, "plain", dtype, i) for i in range(2))
            up = tuple(self.matrix(intermediate, h, "plain", dtype, i) for i in range(2))
            down = tuple(self.matrix(h, intermediate, "plain", dtype, i) for i in range(2))
            operators = PackedLinearOperators()
            kwargs = dict(first_expert=0, top_k=1, operators=operators,
                          reduce_routes=lambda value: None)
            root = torch.full((m * h + 2,), 7.0, device="cuda", dtype=dtype)
            routes = root[1:-1].view(m, 1, h)
            outputs = [torch.empty_like(x), torch.empty_like(x)]
            blocks = [PackedRoutedExpertsPrimitive(
                x, logits, activity, active, gate, up, down, outputs[i],
                route_workspace=None if i == 0 else routes, **kwargs,
            ) for i in range(2)]
            for block in blocks:
                block.launch()
            self.assertTrue(storage_equal(outputs[0], outputs[1]))
            self.assertTrue(storage_equal(blocks[0].routes, routes))
            self.assertEqual(root[0].item(), 7.0)
            self.assertEqual(root[-1].item(), 7.0)
            self.assertEqual(blocks[1].routes.data_ptr(), routes.data_ptr())
            with self.assertRaisesRegex(ValueError, "aliases"):
                PackedRoutedExpertsPrimitive(
                    x, logits, activity, active, gate, up, down, outputs[1],
                    route_workspace=x.view(m, 1, h), **kwargs,
                )

    def test_lossless_marlin_scale_rebias(self):
        from cubic_serve.quantization.extensions.nvfp4.marlin import encode_scales

        scales = torch.arange(127, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        encoded = encode_scales(scales)
        decoded = (encoded.to(torch.int16) << 7).view(torch.float16)
        self.assertTrue(storage_equal(decoded, scales.half() * 128))
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
        levels = torch.cat((levels, -levels)).half()
        expected = levels[:, None] * scales.half()[None, :]
        actual = (((levels / 16384)[:, None] * decoded[None, :]).float() * 128).half()
        self.assertTrue(storage_equal(actual, expected))
        # Preserve the reference's scale domain; never clip small scales or
        # silently reinterpret negative/nonfinite checkpoint values.
        for code in (127, 129, 255):
            invalid = torch.tensor([code], dtype=torch.uint8, device="cuda")
            with self.assertRaises(ValueError):
                encode_scales(invalid.view(torch.float8_e4m3fn))

    def test_awq_marlin_grouped_padding_and_cross_m(self):
        self.audit_routed_formats(("awq",), 3072, 256)

    def test_marlin_original_half_scale_rounding(self):
        from cubic_serve.ops.cuda.native import CudaScalar
        from cubic_serve.quantization.common.marlin.compiler import compile_module

        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        fp8_scales = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).half() * 128
        scales = torch.cat(
            (
                bits[::251].view(torch.float16),
                fp8_scales,
                torch.tensor([0.0, -0.0, 2**-24, 2**-14, 65504.0, -65504.0], dtype=torch.float16),
            )
        )
        scales = scales[scales.isfinite()].contiguous()
        gpu_scales = scales.cuda()
        for dtype, tag in ((torch.float16, "kFloat16"), (torch.bfloat16, "kBFloat16")):
            values = bits.view(dtype)
            values = values[values.isfinite()].contiguous()
            count = values.numel()
            self.assertEqual(count % 4, 0)
            x = values.cuda()
            output = torch.empty(scales.numel(), count, dtype=dtype, device=x.device)
            source = f"""#define MARLIN_NAMESPACE_NAME scale_probe
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "marlin_template.h"
namespace scale_probe {{
extern "C" __global__ void scale_audit(
    const void* input, const half* scales, void* output, int nv, int ns) {{
  using D = MarlinScalarType<vllm::{tag}.id()>;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nv / 4 * ns) return;
  int row = index / (nv / 4), col = index % (nv / 4);
  typename D::FragB b;
  typename D::FragS s;
  reinterpret_cast<half*>(&s)[0] = scales[row];
  b[0] = reinterpret_cast<const typename D::scalar_t2*>(input)[col * 2];
  b[1] = reinterpret_cast<const typename D::scalar_t2*>(input)[col * 2 + 1];
  scale<vllm::{tag}.id()>(b, s, 0);
  reinterpret_cast<typename D::scalar_t2*>(output)[index * 2] = b[0];
  reinterpret_cast<typename D::scalar_t2*>(output)[index * 2 + 1] = b[1];
}}
}}"""
            module, entry = compile_module(source, x.device)
            launch = module.bind(
                entry,
                (
                    x,
                    gpu_scales,
                    output,
                    CudaScalar("i32", count),
                    CudaScalar("i32", scales.numel()),
                ),
                operator_id="cubic.cuda.audit.scale",
                grid=((count // 4 * scales.numel() + 255) // 256, 1, 1),
                block=(256, 1, 1),
                shared_bytes=0,
            )
            launch.launch()
            # Every finite FP16 input pattern; FP64 independently witnesses its
            # exact product. BF16 retains the original FP32 product boundary.
            wide = torch.float64 if dtype == torch.float16 else torch.float32
            expected = (values.to(wide)[None] * scales.to(wide)[:, None]).to(dtype)
            self.assertTrue(storage_equal(output.cpu(), expected))

    def test_nvfp4_marlin_padding_and_cross_m(self):
        self.audit_routed_formats(("nvfp4",), 3072, 256)

    def test_host_checkpoint_routed_formats(self):
        self.audit_routed_formats(
            ("plain", "awq", "fp8", "nvfp4", "cubic", "mixed"), 128, 64, "cpu"
        )

    def test_host_checkpoint_nvfp4_marlin(self):
        self.audit_routed_formats(("nvfp4",), 3072, 256, "cpu")

    def audit_routed_formats(self, formats, hidden, intermediate, checkpoint_device="cuda"):
        from cubic_serve.ops.strategies import OperatorConfig

        for dtype in (torch.bfloat16, torch.float16):
            for fmt in formats:
                with self.subTest(dtype=dtype, fmt=fmt):
                    torch.manual_seed(173)
                    e, h, i, topk = 3, hidden, intermediate, 3
                    gf, uf, df = ("awq", "fp8", "nvfp4") if fmt == "mixed" else (fmt,) * 3
                    gate = tuple(self.matrix(i, h, gf, dtype, j) for j in range(e))
                    up = tuple(self.matrix(i, h, uf, dtype, j + 1) for j in range(e))
                    down = tuple(self.matrix(h, i, df, dtype, j + 2) for j in range(e))
                    gate, up, down = (
                        tuple(
                            w.to(checkpoint_device) if isinstance(w, QuantizedMatrix) else w
                            for w in weights
                        )
                        for weights in (gate, up, down)
                    )
                    corpus = torch.randn(33, h, dtype=dtype, device="cuda:0") * 0.1
                    router = torch.randn(33, e, dtype=dtype, device="cuda:0")
                    first = None
                    config = OperatorConfig.from_dict(
                        {
                            "version": 1,
                            "overrides": {
                                "awq_moe": {
                                    "fixed_block_m": 16,
                                    "fixed_block_n": 32,
                                    "fixed_warps": 4,
                                    "fixed_stages": 1,
                                }
                            }
                            if fmt == "awq"
                            else {},
                        }
                    )
                    for cap, m in ((1, 1), (8, 8), (64, 33)):
                        x = torch.zeros(cap, h, dtype=dtype, device="cuda:0")
                        logits = torch.zeros(cap, e, dtype=dtype, device="cuda:0")
                        activity = torch.zeros(cap, dtype=torch.int32, device="cuda:0")
                        active = torch.tensor([m], dtype=torch.int32, device="cuda:0")
                        output = torch.empty_like(x)
                        op = PackedRoutedExpertsPrimitive(
                            x,
                            logits,
                            activity,
                            active,
                            gate,
                            up,
                            down,
                            output,
                            first_expert=0,
                            top_k=topk,
                            operators=PackedLinearOperators(config),
                        )
                        if fmt == "awq":
                            self.assertIsNone(op.projections[1][2])
                        elif fmt == "mixed":
                            self.assertIsNotNone(op.projections[1][2])
                        op.launch()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            op.launch()
                        x.fill_(float("nan"))
                        logits.fill_(float("nan"))
                        x[:m] = corpus[:m]
                        logits[:m] = router[:m]
                        activity[:m] = 1
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(bool((op.route_ids[m:] == -1).all()))
                        self.assertTrue(bool((op.route_weights[m:] == 0).all()))
                        self.assertTrue(bool((output[m:] == 0).all()))
                        ids = torch.arange(e, device="cuda:0").expand(m, -1)
                        probs = torch.softmax(logits[:m].float(), dim=-1).to(dtype)
                        self.assertTrue(storage_equal(op.route_ids[:m], ids))
                        self.assertTrue(storage_equal(op.route_weights[:m], probs))
                        packed = op.packed.reshape(cap, topk, 2 * i)
                        activated = op.activated.reshape(cap, topk, i)
                        expected = torch.zeros_like(output[:m])
                        for expert in range(e):
                            check_linear_enclosure(x[:m], gate[expert], packed[:m, expert, :i])
                            check_linear_enclosure(x[:m], up[expert], packed[:m, expert, i:])
                            self.assert_swiglu_enclosure(
                                packed[:m, expert, :i],
                                packed[:m, expert, i:],
                                activated[:m, expert],
                            )
                            check_linear_enclosure(
                                activated[:m, expert], down[expert], op.routes[:m, expert]
                            )
                            expected = expected + (
                                op.routes[:m, expert] * probs[:, expert, None]
                            ).to(dtype)
                        self.assertTrue(storage_equal(output[:m], expected))
                        current = (
                            output[:m].clone(),
                            op.route_weights[:m].clone(),
                            op.routes[:m].clone(),
                        )
                        if first is None:
                            first = current
                        else:
                            self.assertTrue(
                                all(
                                    storage_equal(a, b[: len(a)])
                                    for a, b in zip(first, current, strict=True)
                                )
                            )
                        first = current
                        baseline = tuple(
                            t[:m].clone() for t in (output, op.route_weights, op.routes)
                        )
                        rows = torch.arange(m - 1, -1, -1, device=x.device)
                        x[:m] = corpus[:m][rows]
                        logits[:m] = router[:m][rows]
                        graph.replay()
                        torch.cuda.synchronize()
                        for actual, original in zip(
                            (output, op.route_weights, op.routes), baseline, strict=True
                        ):
                            self.assertTrue(storage_equal(actual[:m][rows], original))
                        # A rank owns a fixed contiguous expert slice. Other
                        # routes contribute zero and never read rank-local payloads.
                        local = PackedRoutedExpertsPrimitive(
                            x,
                            logits,
                            activity,
                            active,
                            gate[1:],
                            up[1:],
                            down[1:],
                            output,
                            first_expert=1,
                            top_k=topk,
                            operators=PackedLinearOperators(),
                        )
                        local.launch()
                        torch.cuda.synchronize()
                        local_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(local_graph):
                            local.launch()
                        local_graph.replay()
                        torch.cuda.synchronize()
                        expected_local = torch.zeros_like(output[:m])
                        local_packed = local.packed.reshape(cap, topk, 2 * i)
                        local_activated = local.activated.reshape(cap, topk, i)
                        for expert in range(1, e):
                            # Removing non-owned experts can select a different
                            # native reduction plan. Witness each producing
                            # boundary on its actual operands instead of imposing
                            # bit identity with the full-owner launch.
                            check_linear_enclosure(
                                x[:m], gate[expert], local_packed[:m, expert, :i]
                            )
                            check_linear_enclosure(
                                x[:m], up[expert], local_packed[:m, expert, i:]
                            )
                            self.assert_swiglu_enclosure(
                                local_packed[:m, expert, :i],
                                local_packed[:m, expert, i:],
                                local_activated[:m, expert],
                            )
                            check_linear_enclosure(
                                local_activated[:m, expert], down[expert],
                                local.routes[:m, expert],
                            )
                            expected_local += (
                                local.routes[:m, expert] * baseline[1][rows, expert, None]
                            ).to(dtype)
                        self.assertTrue(storage_equal(output[:m], expected_local))
                        self.assertTrue(bool((output[m:] == 0).all()))


class PackedMoEConfigurationTests(unittest.TestCase):
    def test_grouped_projection_override_is_format_owned_and_boolean(self):
        from cubic_serve.ops.strategies import OperatorConfig

        for family in ("awq_moe", "fp8_moe", "nvfp4_moe", "cubic_moe"):
            config = OperatorConfig.from_dict(
                {"version": 1, "overrides": {family: {"grouped_projections": False}}}
            )
            self.assertFalse(
                config.resolve(
                    family, device="", model="", n=64, k=128, experts=3, top_k=2
                ).grouped_projections
            )
            with self.assertRaises(ValueError):
                OperatorConfig.from_dict(
                    {"version": 1, "overrides": {family: {"grouped_projections": 1}}}
                )

    def test_projection_and_expert_parameters_remain_independent(self):
        from cubic_serve.ops.strategies import OperatorConfig

        config = OperatorConfig.from_dict(
            {
                "version": 1,
                "overrides": {"cubic_moe": {"fixed_block_n": 64}},
            }
        )
        shape = dict(device="", model="", n=128, k=128, experts=3, top_k=2)
        for family in ("awq_moe", "fp8_moe"):
            self.assertEqual(
                config.resolve(family, **shape), OperatorConfig().resolve(family, **shape)
            )
        self.assertEqual(config.resolve("cubic_moe", **shape).fixed_block_n, 64)
        self.assertEqual(
            config.resolve("cubic_linear", device="", model="", n=128, k=128),
            OperatorConfig().resolve("cubic_linear", device="", model="", n=128, k=128),
        )
        self.assertEqual(config.resolve("nvfp4_moe", **shape).fixed_block_k, 16)
        with self.assertRaisesRegex(ValueError, "format boundary"):
            OperatorConfig.from_dict(
                {
                    "version": 1,
                    "overrides": {"nvfp4_moe": {"fixed_block_k": 32}},
                }
            )


class NVFP4ThresholdTests(unittest.TestCase):
    def test_exact_midpoint_rewrite_all_e4m3_scales(self):
        # Monotonicity confines a changed code to a comparison boundary. Probe
        # adjacent FP32 numerators on both sides of every E2M1 midpoint for
        # every positive finite E4M3 scale, including subnormal scales.
        scales = torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        products = scales[:, None] * thresholds
        self.assertTrue(torch.equal(products.double(), scales.double()[:, None] * thresholds))
        neighbors = (
            products.view(torch.int32)[:, :, None] + torch.arange(-32, 33, dtype=torch.int32)
        ).view(torch.float32)
        quotient = neighbors / scales[:, None, None]
        inclusive = torch.tensor([False, True, False, True, False, True, False])[None, :, None]
        old = torch.where(
            inclusive, quotient >= thresholds[None, :, None], quotient > thresholds[None, :, None]
        )
        new = torch.where(
            inclusive, neighbors >= products[:, :, None], neighbors > products[:, :, None]
        )
        self.assertTrue(torch.equal(old, new))


@unittest.skipUnless(os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1", "explicit CUDA audit required")
class RoutingLaunchTests(unittest.TestCase):
    def test_selected_routing_keeps_cross_m_ties_padding_and_rng(self):
        from cubic_serve.ops.cuda.moe.quantized import _bind_route_selection
        from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope
        from cubic_serve.quantization.common.binding import PackedOperatorBinding

        class FixedChoice(dict):
            def __init__(self, choice):
                self.choice = choice

            def __contains__(self, key):
                return True

            def __getitem__(self, key):
                return self.choice

        configure_reference_math()
        torch.set_num_threads(1)
        policy = OperatorConfig(activation_dtype="fp32", attention_dtype="fp32")
        with operator_strategy_scope(policy, device="", model=""):
            for dtype in (torch.float16, torch.bfloat16):
                generator = torch.Generator(device="cuda").manual_seed(831)
                common = torch.randn(127, device="cuda", dtype=dtype, generator=generator)
                common_result = None
                for rows in (1, 4, 32, 256):
                    logits = torch.randn(rows, 127, device="cuda", dtype=dtype, generator=generator)
                    logits[0].copy_(common)
                    activity = torch.ones(rows, device="cuda", dtype=torch.int32)
                    active = torch.tensor([rows], device="cuda", dtype=torch.int32)
                    targets, bindings = [], []
                    for choices in (FixedChoice(0), FixedChoice(1), {}):
                        outputs = (
                            torch.empty(rows, 7, device="cuda", dtype=torch.float32),
                            torch.empty(rows, 7, device="cuda", dtype=torch.int64),
                            torch.empty(rows, 7, device="cuda", dtype=torch.int64),
                            torch.empty(rows, 7, device="cuda", dtype=dtype),
                            torch.empty(31, device="cuda", dtype=torch.int32),
                            torch.empty(31, device="cuda", dtype=torch.int32),
                        )
                        rng = torch.cuda.get_rng_state().clone()
                        calls = _bind_route_selection(
                            logits, activity, active, *outputs, choices=choices
                        )
                        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
                        bindings.append(PackedOperatorBinding(outputs[3], calls))
                        targets.append(outputs)
                    for case in ("random", "permuted", "ties", "padding"):
                        if case == "permuted":
                            logits.copy_(logits.flip(0))
                        elif case == "ties":
                            logits.zero_()
                        elif case == "padding" and rows > 1:
                            # Explicit inactive holes, including a physical NaN row.
                            activity[1::2].zero_()
                            logits[1::2].fill_(float("nan"))
                        for binding in bindings:
                            binding.launch()
                        for actual in targets[1:]:
                            for expected, value in zip(targets[0], actual, strict=True):
                                self.assertTrue(storage_equal(expected, value), (rows, dtype, case))
                        if case == "random":
                            first = tuple(t[0].clone() for t in targets[0][:4])
                            if common_result is None:
                                common_result = first
                            else:
                                for a, b in zip(common_result, first, strict=True):
                                    self.assertTrue(storage_equal(a, b))
                            physical = tuple(t.clone() for t in targets[0][:4])
                        elif case == "permuted":
                            for a, b in zip(physical, targets[0][:4], strict=True):
                                self.assertTrue(storage_equal(a.flip(0), b))
                        elif case == "ties":
                            expected = torch.arange(7, device="cuda").expand(rows, -1)
                            self.assertTrue(torch.equal(targets[0][2], expected))
