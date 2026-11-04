"""Wide slot offsets with a few physically backed pages, not a huge allocation."""
import os
import gc

import pytest
import torch

from cubic_serve.runtime.cache.shared_pool import SharedCacheField,SharedCacheLayout
from cubic_serve.runtime.cache.virtual_memory import VirtualAllocation,virtual_memory_supported
from cubic_serve.executor.runner import ReferenceExecutionContext,configure_reference_math

pytestmark=pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA')!='1',reason='explicit CUDA address proof')
SOURCE,DESTINATION=1025,1026


@pytest.fixture
def wide_fields():
    retained=[]
    def make(fields):
        device=torch.device('cuda',torch.cuda.current_device())
        if not virtual_memory_supported(device):pytest.skip('sparse virtual address witness requires CUDA VMM')
        stride=32<<20
        base=SharedCacheLayout(tuple(fields))
        layout=SharedCacheLayout((*fields,SharedCacheField('gap',torch.uint8,(stride-base.entry_bytes,))))
        assert layout.entry_bytes==stride
        owner=VirtualAllocation((DESTINATION+1)*stride,device)
        for index in (0,1,2,SOURCE,DESTINATION):owner.ensure_range(index*stride,base.entry_bytes)
        raw=owner.tensor(((DESTINATION+1)*stride,),torch.uint8)
        views=layout.views(raw,DESTINATION+1)
        views.pop('gap')
        for value in views.values():
            for index in (1,2,SOURCE,DESTINATION):value[index].fill_(13)
        owner.seal()
        assert owner.backed_bytes<=5*owner.granularity
        retained.append((owner,raw,views))
        return views
    yield make
    retained.clear();gc.collect()


def _metadata():
    source=torch.tensor([SOURCE],device='cuda',dtype=torch.int32)
    destination=torch.tensor([DESTINATION],device='cuda',dtype=torch.int32)
    live=torch.ones(1,device='cuda',dtype=torch.int32)
    return source,destination,live


def _capture(call):
    call();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):call()
        graph.replay();torch.cuda.synchronize()
    finally:graph.reset()


def _protected(fields):
    return {name:value[[1,2]].clone() for name,value in fields.items()}


def _check_protected(fields,before):
    for name,value in fields.items():assert torch.equal(value[[1,2]],before[name]),name


def test_wide_ngram_history_retains_exact_integer_bits_and_disjoint_low_slots(wide_fields):
    from cubic_serve.ops.cuda.embedding.ngram import bind_ngram_hash
    from cubic_serve.ops.eager.ngram import hash_indices
    configure_reference_math()
    fields=wide_fields((SharedCacheField('history',torch.int64,(2,)),))
    history=fields['history'];history[SOURCE]=torch.tensor([2**54+1,2**61+3],device='cuda')
    token=torch.tensor([7],device='cuda',dtype=torch.int64)
    multipliers=torch.tensor([2**62+57,1000000007,97],device='cuda',dtype=torch.int64)
    vocab=torch.tensor([11,13,17,19],device='cuda',dtype=torch.int64)
    offsets=torch.tensor([0,11,24,41],device='cuda',dtype=torch.int64)
    out=torch.empty(1,4,device='cuda',dtype=torch.int64)
    src,dst,live=_metadata();before=_protected(fields)
    with ReferenceExecutionContext():
        expected,next_history=hash_indices(token.cpu().view(1,1),history[SOURCE].cpu().view(1,2),
            multipliers.cpu(),vocab.cpu(),offsets.cpu(),eos_token_id=31)
    binding=bind_ngram_hash(token,history,src,dst,live,live,multipliers,vocab,offsets,out,eos_token_id=31)
    _capture(binding.launch)
    assert torch.equal(out.cpu(),expected[0])
    assert torch.equal(history[DESTINATION].cpu(),next_history[0])
    _check_protected(fields,before)


def test_wide_dilated_convolution_reads_and_checkpoints_the_owned_slot(wide_fields):
    from cubic_serve.ops.cuda.pointwise.dilated_conv import bind_dilated_depthwise_conv
    from cubic_serve.guardrail.quantized_graph.bounds import check,scaled_silu_bounds
    configure_reference_math()
    fields=wide_fields((SharedCacheField('history',torch.bfloat16,(2,8)),))
    history=fields['history'];history[SOURCE]=torch.arange(1,17,device='cuda').view(2,8)
    x=torch.arange(8,device='cuda',dtype=torch.bfloat16).view(1,8)
    weight=torch.zeros(8,1,3,device='cuda',dtype=x.dtype);weight[:,0,0]=1
    out=torch.empty_like(x);src,dst,live=_metadata();before=_protected(fields)
    expected=torch.cat((history[SOURCE,1:].cpu(),x.cpu()))
    interval=scaled_silu_bounds(history[SOURCE,0].cpu(),divisor=1.,activation_dtype=torch.bfloat16)
    binding=bind_dilated_depthwise_conv(x,history,src,dst,live,live,weight,out,dilation=1)
    _capture(binding.launch)
    assert check(out[0].cpu(),*interval)["outside_enclosure"]==0
    assert torch.equal(history[DESTINATION].cpu(),expected)
    _check_protected(fields,before)


def test_wide_qsa_history_preserves_exact_positions_and_group_mean(wide_fields):
    from cubic_serve.ops.cuda.attention.sparse_indexer import bind_causal_key_compression
    configure_reference_math()
    fields=wide_fields((SharedCacheField('keys',torch.bfloat16,(4,8)),
                        SharedCacheField('positions',torch.int64,(4,3))))
    history,positions=fields['keys'],fields['positions']
    history[SOURCE]=torch.arange(32,device='cuda').view(4,8)
    positions[SOURCE]=torch.arange(12,device='cuda').view(4,3)+2**54+1
    key=-torch.arange(8,device='cuda',dtype=torch.bfloat16).view(1,8)
    current=torch.tensor([[2**54+17,2**54+18,2**54+19]],device='cuda',dtype=torch.int64)
    logical=torch.tensor([3],device='cuda',dtype=torch.int64)
    pooled=torch.empty_like(key);group=torch.empty_like(current);complete=torch.empty(1,device='cuda',dtype=torch.int32)
    src,dst,live=_metadata();before=_protected(fields)
    expected=torch.cat((history[SOURCE,1:].cpu(),key.cpu()))
    expected_positions=torch.cat((positions[SOURCE,1:].cpu(),current.cpu()))
    binding=bind_causal_key_compression(key,current,logical,history,positions,src,dst,live,live,pooled,group,complete)
    _capture(binding.launch)
    assert torch.equal(history[DESTINATION].cpu(),expected)
    assert torch.equal(positions[DESTINATION].cpu(),expected_positions)
    assert torch.equal(pooled.cpu(),expected.float().mean(0).to(key.dtype)[None])
    assert torch.equal(group.cpu(),expected_positions[:1]) and complete.item()
    _check_protected(fields,before)


def test_wide_pending_and_feedback_publish_use_declared_owner_pitch(wide_fields):
    from cubic_serve.executor.draft_graph import pending_copy,publish_feedback,prompt_provider_feedback
    fields=wide_fields((SharedCacheField('scores',torch.float32,(2,17)),
        SharedCacheField('ids',torch.int64,(2,17)),SharedCacheField('tokens',torch.int64,(1,)),
        SharedCacheField('feedback',torch.bfloat16,(259,))))
    ctx=torch.tensor([1,0,DESTINATION],device='cuda',dtype=torch.int32)
    scores=torch.arange(34,device='cuda',dtype=torch.float32).view(2,1,17)
    ids=torch.arange(34,device='cuda',dtype=torch.int64).view(2,1,17)+2**54+1
    tokens=torch.tensor([2**54+3],device='cuda',dtype=torch.int64)
    ps,pi,pt=fields['scores'].permute(1,0,2),fields['ids'].permute(1,0,2),fields['tokens'][:,0]
    kw=dict(SCORE_OWNER=ps.stride(1),SCORE_TP=ps.stride(0),ID_OWNER=pi.stride(1),
            ID_TP=pi.stride(0),TOKEN_OWNER=pt.stride(0))
    before=_protected(fields)
    _capture(lambda:pending_copy[(1,)](ctx,scores,ids,tokens,ps,pi,pt,2,1,DESTINATION+1,2,17,32,False,**kw))
    assert torch.equal(fields['scores'][DESTINATION],scores[:,0])
    assert torch.equal(fields['ids'][DESTINATION],ids[:,0]) and pt[DESTINATION]==tokens[0]
    loaded_scores=torch.empty_like(scores);loaded_ids=torch.empty_like(ids);loaded_tokens=torch.empty_like(tokens)
    pending_copy[(1,)](ctx,loaded_scores,loaded_ids,loaded_tokens,ps,pi,pt,2,1,DESTINATION+1,2,17,32,True,**kw)
    assert torch.equal(loaded_scores,scores) and torch.equal(loaded_ids,ids) and torch.equal(loaded_tokens,tokens)
    src,dst,live=_metadata();feedback=torch.arange(259,device='cuda').view(1,259).to(torch.bfloat16)
    _capture(lambda:publish_feedback[(1,2)](live,dst,feedback,fields['feedback'],259,1,fields['feedback'].stride(0),256))
    assert torch.equal(fields['feedback'][DESTINATION],feedback[0])
    starts=torch.zeros_like(live);out=torch.empty_like(feedback)
    _capture(lambda:prompt_provider_feedback[(1,2)](live,starts,live,dst,feedback,fields['feedback'],out,
        1,1,259,256,False,DURABLE_STRIDE=fields['feedback'].stride(0)))
    assert torch.equal(out,feedback)
    _check_protected(fields,before)
