"""Startup-bound candidate preparation; caller owns the enclosing graph lease."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native

if TYPE_CHECKING:
    from cubic_serve.generation.tp_sampling import ShardedSamplingWorkspace


class LocalCandidatePlan:
    """Preserve torch.topk and fuse its owned-row checks/publication.

    Native launches and buffers are bound before capture. Replay may change
    temperatures and explicit activity/count inputs, never their ownership.
    This operator does not capture graphs, grant readiness, or consume RNG.
    """

    def __init__(
        self,
        *,
        local_logits: Tensor,
        workspace: ShardedSamplingWorkspace,
        active_rows: int,
        visible_width: int,
        local_count: int,
        vocab_start: int,
        row_activity: Tensor | None = None,
        metadata: Tensor | None = None,
        all_random_rows: bool = False,
        token_mask: Tensor | None = None,
        masked_logits: Tensor | None = None,
        consume_logits: bool = False,
    ) -> None:
        dtype = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}.get(local_logits.dtype)
        if (
            dtype is None
            or local_logits.ndim != 2
            or local_logits.stride(1) != 1
            or not 0 < active_rows <= local_logits.shape[0]
            or not 0 < local_count <= visible_width <= local_logits.shape[1]
            or workspace.local_scores.dtype != local_logits.dtype
            or workspace.local_token_ids.dtype != torch.int64
            or workspace.local_scores.shape != workspace.local_token_ids.shape
            or workspace.local_scores.shape[0] < active_rows
            or workspace.local_scores.shape[1] < local_count
            or not workspace.local_scores.is_contiguous()
            or not workspace.local_token_ids.is_contiguous()
            or (
                row_activity is not None
                and (
                    row_activity.dtype != torch.bool
                    or row_activity.shape != (active_rows,)
                    or not row_activity.is_contiguous()
                )
            )
            or (
                metadata is not None
                and (
                    metadata.dtype != torch.int32
                    or metadata.numel() < 1
                    or not metadata.is_contiguous()
                )
            )
        ):
            raise ValueError("local candidate preparation has an invalid tensor ABI")
        tensors = (
            workspace.local_scores,
            workspace.local_token_ids,
            *((row_activity,) if row_activity is not None else ()),
            *((metadata,) if metadata is not None else ()),
        )
        if token_mask is not None:
            if (
                token_mask.dtype != torch.int32
                or token_mask.ndim != 2
                or token_mask.shape[0] != active_rows
                or token_mask.shape[1] * 32 < vocab_start + visible_width
                or not token_mask.is_contiguous()
                or token_mask.device != local_logits.device
            ):
                raise ValueError("grammar bitmask differs from local candidate ownership")
            tensors += (token_mask,)
        if local_logits.device.type != "cuda" or any(
            x.device != local_logits.device for x in tensors
        ):
            raise ValueError("local candidate preparation requires one CUDA device")
        if type(consume_logits) is not bool or consume_logits and token_mask is None:
            raise ValueError("logits consumption requires an explicit masked terminal consumer")
        if consume_logits and masked_logits is None:
            masked_logits = local_logits[:active_rows, :visible_width]
        if masked_logits is not None and (
            token_mask is None
            or masked_logits.shape != (active_rows, visible_width)
            or masked_logits.dtype != local_logits.dtype
            or masked_logits.device != local_logits.device
            or (not consume_logits and not masked_logits.is_contiguous())
            or (consume_logits and (
                masked_logits.data_ptr() != local_logits.data_ptr()
                or masked_logits.stride() != local_logits.stride()))
            or (not consume_logits and masked_logits.untyped_storage().data_ptr()
                == local_logits.untyped_storage().data_ptr())
            or any(
                masked_logits.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                for t in tensors
            )
        ):
            raise ValueError("masked logits scratch differs from the non-aliased candidate ABI")
        self.logits, self.capacity = local_logits, active_rows
        self.width, self.count = visible_width, local_count
        self.outputs = (workspace.local_scores, workspace.local_token_ids)
        self.activity, self.metadata = row_activity, metadata
        self.vocab_start, self.all_random = vocab_start, all_random_rows
        self.token_mask = token_mask
        self.masked_logits = (
            None
            if token_mask is None
            else masked_logits if masked_logits is not None else torch.empty(
                (active_rows, visible_width), dtype=local_logits.dtype, device=local_logits.device
            )
        )
        self.evidence: dict[str, Any] | None = None
        self.temperatures = torch.empty(active_rows, device=local_logits.device)
        self.top_scores = torch.empty((active_rows, local_count),
                                      dtype=local_logits.dtype, device=local_logits.device)
        self.top_ids = torch.empty_like(self.top_scores, dtype=torch.int64)
        tiles = (visible_width + 4095) // 4096
        partial_scores = torch.empty((active_rows, tiles), device=local_logits.device)
        partial_ids = torch.empty_like(partial_scores, dtype=torch.int32)
        partial_finite = torch.empty_like(partial_ids)
        source = Path(__file__).with_suffix('.cu').read_bytes()
        common = (partial_scores, partial_ids, partial_finite)
        calls = []
        for phase, arguments, block in (
            (
                0,
                (
                    local_logits,
                    self.temperatures,
                    row_activity,
                    metadata,
                    *common,
                    CudaScalar("i32", visible_width),
                    CudaScalar("i32", local_logits.stride(0)),
                    CudaScalar("i32", tiles),
                    CudaScalar("i32", int(all_random_rows)),
                    *(
                        (
                            token_mask,
                            self.masked_logits,
                            CudaScalar("i32", token_mask.shape[1]),
                            CudaScalar("i64", vocab_start),
                            CudaScalar("i32", self.masked_logits.stride(0)),
                        )
                        if token_mask is not None
                        else ()
                    ),
                ),
                256,
            ),
            (
                1,
                (
                    self.top_scores,
                    self.top_ids,
                    self.temperatures,
                    row_activity,
                    metadata,
                    *common,
                    workspace.local_scores,
                    workspace.local_token_ids,
                    CudaScalar("i32", tiles),
                    CudaScalar("i32", local_count),
                    CudaScalar("i32", workspace.local_scores.shape[1]),
                    CudaScalar("i64", vocab_start),
                    CudaScalar("i32", int(all_random_rows)),
                ),
                32,
            ),
        ):
            prefix = (
                f"#define SCORE_TYPE {dtype}\n#define PHASE {phase}\n"
                f"#define HAS_GRAMMAR {int(token_mask is not None)}\n"
                f"#define CONSUME_LOGITS {int(consume_logits)}\n"
            ).encode()
            module, entry = compile_native(
                (("entry.cu", prefix + source),), local_logits.device, "local_candidates"
            )
            calls.append(
                module.bind(
                    entry,
                    arguments,
                    operator_id=f"sampling.local_candidates.phase{phase}",
                    grid=(active_rows, tiles if phase == 0 else 1, 1),
                    block=(block, 1, 1),
                )
            )
        self.calls = tuple(calls)

    def launch(self, *, temperatures: Tensor) -> None:
        if (temperatures.dtype != torch.float32
                or temperatures.device != self.temperatures.device
                or temperatures.shape != self.temperatures.shape
                or not temperatures.is_contiguous()):
            raise ValueError("local candidate temperatures differ from the bound ABI")
        self.temperatures.copy_(temperatures)
        # Stable native operands, same selection algorithm and candidate ties.
        if self.token_mask is None:
            torch.topk(
                self.logits[: self.capacity, : self.width],
                self.count,
                dim=-1,
                out=(self.top_scores, self.top_ids),
            )
            self.calls[0].launch()
        else:
            self.calls[0].launch()
            assert self.masked_logits is not None
            torch.topk(self.masked_logits, self.count, dim=-1, out=(self.top_scores, self.top_ids))
        self.calls[1].launch()

    def qualify(self) -> None:
        """Independent CPU boundary witness, outside any observed forward."""
        tensors = (self.logits, *self.outputs,
                   *((self.activity,) if self.activity is not None else ()),
                   *((self.metadata,) if self.metadata is not None else ()))
        # Startup qualification borrows externally owned storage. Keep its exact
        # backup off device: a complete vocabulary-sized logits clone must not
        # compete with the graph/runtime allocations being qualified.
        saved = tuple(t.detach().to(device="cpu", copy=True) for t in tensors)
        rng = torch.cuda.get_rng_state(self.logits.device)
        cases = []
        try:
            columns = torch.arange(self.width)
            for count, tag in [(n, tag) for n in sorted({1, max(1, self.capacity // 2),
                                                        self.capacity})
                               for tag in ('random', 'greedy_ties', 'nonfinite')]:
                cpu = torch.full(self.logits.shape, float('nan'), dtype=self.logits.dtype)
                for row in range(count):
                    cpu[row, :self.width] = ((columns + row * 7) % 31 - 15).float()
                    if tag == 'greedy_ties':
                        cpu[row, 0] = cpu[row, self.width - 1] = 32
                if tag == 'nonfinite':
                    cpu[0, 0] = float('inf')
                self.logits.copy_(cpu)
                temperatures = torch.full_like(self.temperatures,
                                               0. if tag == 'greedy_ties' else .7)
                if self.metadata is not None:
                    self.metadata[0] = count
                if self.activity is not None:
                    self.activity.copy_(torch.arange(self.capacity,
                                                    device=self.logits.device) < count)
                self.launch(temperatures=temperatures)
                # The complete operator has returned; synchronize before any
                # independent reference or assertion, never in a callback.
                torch.cuda.synchronize(self.logits.device)
                actual_scores, actual_ids = (x[:self.capacity].cpu() for x in self.outputs)
                producer_scores, producer_ids = self.top_scores.cpu(), self.top_ids.cpu()
                for row in range(self.capacity):
                    scores, ids = actual_scores[row], actual_ids[row]
                    owned = row < count or (self.metadata is None and self.activity is None)
                    finite = bool(torch.isfinite(cpu[row, :self.width]).all())
                    if not owned or not finite or (self.all_random and tag == 'greedy_ties'):
                        expected = -2 if owned and not finite else -1
                        if (not bool(torch.isnan(scores).all())
                                or not bool((ids[:self.count] == expected).all())
                                or not bool((ids[self.count:] == -1).all())):
                            raise RuntimeError('local candidate sentinel boundary audit failed')
                        continue
                    if tag == 'greedy_ties':
                        value, index = cpu[row, :self.width].float().max(0)
                        if (float(scores[0]) != float(value)
                                or int(ids[0]) != int(index) + self.vocab_start
                                or not bool(torch.isnan(scores[1:]).all())
                                or not bool((ids[1:] == -1).all())):
                            raise RuntimeError('local candidate greedy boundary audit failed')
                    else:
                        selected = producer_ids[row]
                        original = cpu[row, :self.width]
                        chosen = original[selected]
                        if (not torch.equal(chosen, producer_scores[row])
                                or selected.unique().numel() != self.count
                                or bool((chosen[1:] > chosen[:-1]).any())
                                or int((original > chosen[-1]).sum()) >= self.count
                                or not torch.equal(scores[:self.count], chosen)
                                or not torch.equal(ids[:self.count], selected + self.vocab_start)
                                or not bool(torch.isnan(scores[self.count:]).all())
                                or not bool((ids[self.count:] == -1).all())):
                            raise RuntimeError('local candidate selection boundary audit failed')
                cases.append((count, tag))
            if not torch.equal(torch.cuda.get_rng_state(self.logits.device), rng):
                raise RuntimeError('local candidate preparation advanced RNG')
            self.evidence = dict(oracle='independent CPU finite/argmax and original-input '
                                       'top-k selection boundary; unchanged top-k producer',
                                 cases=cases, unchanged_rng=True,
                                 launch_signatures=[call.signature for call in self.calls])
        finally:
            for tensor, original in zip(tensors, saved, strict=True):
                tensor.copy_(original)
            torch.cuda.synchronize(self.logits.device)
