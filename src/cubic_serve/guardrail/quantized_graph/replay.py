from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.distributed as dist

from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.ops.eager.quantization import (
    QuantizedMatrix,
    QuantizedReferenceOperators,
    decode_matrix,
)

from . import linear as linear_witness
from .attention_bounds import enclose
from .linear import ProducerMismatch, same
from .linear import Replay as Base


def reference_model_for_token_domain(model, vocabulary_size, *, operators=None):
    """Keep rank-local math shapes while validating global request token IDs.

    Embedding operators still mask the declared lexical shard and the head
    still produces that shard. The local graph's vocabulary width describes
    storage; it must not shrink the token domain of the original model rule.
    """
    if type(vocabulary_size) is not int or vocabulary_size < model.config.graph.text.vocabulary_size:
        raise ValueError("reference token domain cannot shrink the declared vocabulary")
    config = replace(model.config, graph=replace(model.config.graph,
        text=replace(model.config.graph.text, vocabulary_size=vocabulary_size)))
    return type(model)(config=config, weights=model.weights,
                      recurrent_state_dtype=model.recurrent_state_dtype,
                      operators=model.default_operators if operators is None else operators)


class AttentionReplay(Base):
    def __init__(self, recorder, names):
        super().__init__(recorder, names)
        self.attention_events = recorder.attention_events
        self.attention_meta = recorder.attention_meta
        self.cache_reads = getattr(recorder, "cache_reads", [None] * len(self.attention_events))
        self.attention_used = 0
        self.attention_proofs = []

    def full_attention(self, query, key, value, **options):
        i = self.attention_used
        if i >= len(self.attention_events):
            raise RuntimeError("attention event schedule exhausted")
        q, k, v, lengths, y = self.attention_events[i]
        scale, splits, block_n = self.attention_meta[i]
        if not torch.equal(lengths, lengths[:1].expand_as(lengths)):
            raise RuntimeError("batched witness currently requires equal context lengths")
        batch = len(lengths)
        n = int(lengths[0])
        k = k[:, :n]
        v = v[:, :n]
        snapshot = self.cache_reads[i]
        producers = [
            (key, k.transpose(1, 2), "key"),
            (value, v.transpose(1, 2), "value"),
        ]
        if snapshot is None:
            producers.insert(0, (query, q, "query"))
        for a, b, name in producers:
            if not same(a, b):
                raise ProducerMismatch("unwitnessed attention " + name)
        if not same(
            options["query_ordinals"], torch.full((batch, 1), n - 1, device=query.device)
        ) or not same(
            options["key_ordinals"], torch.arange(n, device=query.device)[None].expand(batch, -1)
        ):
            raise RuntimeError("attention causal ordinal schedule differs")
        if snapshot is not None:
            from cubic_serve.ops.eager.kv_cache import (
                cache_attention_reference, witness_cache_query, witness_cache_read,
            )
            from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

            encoding, window, codes, values, table = snapshot
            witness_cache_query(query, q, encoding=encoding, window=window)
            visible = [witness_cache_read(
                key[row].transpose(0, 1), value[row].transpose(0, 1), key_pages=codes,
                value_pages=values, encoding=encoding, window=window, table=table,
                layer=0, row=row,
            ) for row in range(batch)]
            expected = cache_attention_reference(
                query, key, value,
                storage=KVCacheStorageSpec(encoding.key.spec, encoding.value.spec,
                                           encoding.enable_rotation),
                **options,
            )
            k = torch.stack([item[0] for item in visible])
            v = torch.stack([item[1] for item in visible])
        else:
            expected = super().full_attention(query, key, value, **options)
        proofs = [
            enclose(
                q[j : j + 1],
                k[j : j + 1],
                v[j : j + 1],
                y[j : j + 1],
                scale=scale,
                splits=splits,
                block_n=block_n,
                value_rotation=(snapshot is not None and encoding.enable_rotation
                                and encoding.value.spec.dtype.quantized),
            )
            for j in range(batch)
        ]
        proof = {
            "batch": batch,
            "outside_enclosure": sum(p["outside_enclosure"] for p in proofs),
            "rows": proofs,
        }
        proof["independent_staged_max_abs"] = float(
            (expected.reshape_as(y).float() - y.float()).abs().max()
        )
        if proof["outside_enclosure"]:
            raise AssertionError("paged attention exceeds actual-input interval " + str(proof))
        self.attention_proofs.append(proof)
        self.attention_used += 1
        return y[:, :, None, :].clone()


WarmReplay = AttentionReplay


def ordered_sum(x, group):
    values = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(values, x.contiguous(), group=group)
    total = torch.zeros_like(x, dtype=torch.float32)
    for value in values:
        total = total + value.float()
    return total.to(x.dtype)


class Reference(QuantizedReferenceOperators):
    def __init__(self, row_weights, span, group, *, row_groups=None):
        super().__init__(128)
        self.row_weights = row_weights
        self.span = span
        self.group = group
        self.row_groups = {} if row_groups is None else row_groups

    def embedding(self, ids, w, *, dtype):
        owns = (ids >= self.span.start) & (ids < self.span.stop)
        local = torch.where(owns, ids - self.span.start, 0)
        y = super().embedding(local, w, dtype=dtype) * owns[..., None]
        return ordered_sum(y, self.group)

    def linear(self, x, w):
        row = id(w) in self.row_weights
        group = self.row_groups.get(id(w), self.group)
        if (
            row
            and isinstance(w, QuantizedMatrix)
            and (w.encoding.activation is ActivationFormat.FP8_TOKEN)
        ):
            rows = x.reshape(-1, x.shape[-1])
            maximum = rows.float().abs().amax(-1)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            scales = (maximum.cpu().double() / 448).float().clamp_min(1 / (448 * 512))
            codes = (
                (rows.cpu().double() / scales.double()[:, None])
                .float()
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            activation = (codes.float() * scales[:, None]).to(x.device).reshape_as(x)
            from cubic_serve.ops.eager.linear import linear

            outputs = []
            for start in range(0, w.shape[0], self.output_rows):
                weight = decode_matrix(
                    w.rows(start, min(start + self.output_rows, w.shape[0])).to(x.device)
                )
                outputs.append(linear(activation, weight, output_dtype=x.dtype))
            y = torch.cat(outputs, -1)
        else:
            y = super().linear(x, w)
        return ordered_sum(y, group) if row else y


class Replay(WarmReplay):
    def __init__(self, recorder, names, tp, span):
        super().__init__(recorder, names)
        self.tp = tp
        self.span = span
        self.scales = {}
        for wid, events in recorder.events.items():
            for i, (_w, _x, y) in enumerate(events):
                scale = tp.expert_scales.get((wid, i), tp.weight_scales.get(wid))
                if scale is not None:
                    self.scales[wid, y.data_ptr()] = scale

    def embedding(self, ids, w, *, dtype):
        owns = (ids >= self.span.start) & (ids < self.span.stop)
        local = torch.where(owns, ids - self.span.start, 0)
        if self.immutable_embedding_group is not None:
            if self.embedding_used >= len(self.embedding_events):
                raise RuntimeError("immutable embedding observation schedule exhausted")
            weight, source, observed = self.embedding_events[self.embedding_used]
            if weight is not w or not same(ids, source):
                raise AssertionError("immutable embedding producer IDs differ")
            # Independent original-shard decode and ordered sum, after forward.
            y = QuantizedReferenceOperators.embedding(self, local, w, dtype=dtype)
            expected = ordered_sum(y * owns[..., None], self.immutable_embedding_group)
            if not same(expected, observed):
                raise AssertionError("immutable embedding differs from original distributed rows")
            self.embedding_used += 1
            return observed.clone().reshape_as(y)
        y = super().embedding(local, w, dtype=dtype)
        if self.tp.embedding_collective is None:
            return y * owns[..., None]
        source, out = self.tp.embedding_collective
        if not same(y * owns[..., None], source):
            raise AssertionError("embedding mask producer differs")
        return out.clone().reshape_as(y)

    def linear(self, x, w):
        before = set(self.used)
        y = super().linear(x, w)
        key = (self.used - before).pop()
        return self.tp.global_outputs.get(key, y).clone().reshape_as(y)

    @contextmanager
    def scale_witness(self):
        original = linear_witness.check_linear_enclosure

        def check(x, w, y, *, int8_token_scales=None):
            return original(
                x, w, y, fp8_token_scales=self.scales.get((id(w), y.data_ptr())),
                int8_token_scales=int8_token_scales,
            )

        with patch.object(linear_witness, "check_linear_enclosure", check):
            yield
