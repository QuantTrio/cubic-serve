"""Startup-captured target-first round for a declared-depth hidden-conditioned draft.

The target review, prefix repair, provider realignment, head-only proposal and
compact result transfer share one graph. The provider supplies its bound body
and projection; this executor has no model, checkpoint-format or device-name
selection. Its carried q/P and root checkpoint follow executor.draft's independent
pipelined reference. Segmentation and explicit logical activity remain separate.

Construction is pre-readiness only. The caller owns numerical qualification,
portfolio registration and the exclusive topology-wide lease through input fill,
replay, durable state and output consumption. This primitive alone grants neither
readiness nor a speculative numerical receipt. It does not replace the physical
M1 lower-bound bucket. No graph is captured from run().

Algorithm reference: T-Crypt/ninfer-4090 b9a6b5c,
src/targets/qwen3_6/impl/runtime/mtp_impl.h; pinned vLLM llm_base_proposer.py
32b05add and SGLang eagle_worker_common.py c6e822c2 preserve accepted feedback and
provider-cache alignment. Cubic preserves its own rounded state boundaries.
"""
from __future__ import annotations

from cubic_serve.ops.cuda.collective import all_reduce, broadcast, barrier
from cubic_serve.executor.readiness import execution_ready

import math
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import triton
import triton.language as tl
from cubic_serve.ops.cuda.sampling.speculative import verify_one_draft_candidates
from cubic_serve.executor.state_transfer import CapturedStateTransfer, owner_copy
from cubic_serve.runtime.cache.quantization import slot_storage_contiguous

@triton.jit
def align_meta(ctx,dec,tm,ids,mask,pos,ps,pd,off,lens,seg,act,maps,table,
               M:tl.constexpr,T:tl.constexpr,CS:tl.constexpr,TS:tl.constexpr,
               AX:tl.constexpr,L:tl.constexpr,START:tl.constexpr,STOP:tl.constexpr,
               PAGE:tl.constexpr,SCRATCH_PAGE:tl.constexpr,SCRATCH_STATE:tl.constexpr,
               VOCAB:tl.constexpr,BLOCK:tl.constexpr,DEPTH:tl.constexpr=1,C:tl.constexpr=0,
               checkpoint_ctx=None,CHECKPOINT_CS:tl.constexpr=0,PADDING_ROWS:tl.constexpr=0,
               PADDING_PAGES:tl.constexpr=0):
    row=tl.program_id(0);r=row//(DEPTH+1);j=row%(DEPTH+1)
    n=tl.load(ctx);a=tl.load(dec+r*2,r<n,-1);bonus=tl.load(dec+r*2+1,r<n,-1)
    owned=(r<n)&(a>=0)&(a<=DEPTH)&(bonus>=0)&(bonus<VOCAB)
    live=owned&(j<=a)
    owner=tl.load(ctx+1+r*CS+1,r<n,0)
    q=tl.load(ctx+1+r*CS+2+j,(r<n)&(j<DEPTH),0)
    rootpos=tl.load(ctx+1+r*CS,r<n,0)
    cachepos=tl.where(live,rootpos+j+1,0)
    token=tl.where(j<a,q,bonus)
    valid=live&(token>=0)&(token<VOCAB)
    local=(token>=START)&(token<STOP)&valid
    tl.store(ids+row,tl.where(local,token-START,0));tl.store(mask+row,local)
    for axis in tl.static_range(3):
        x0=tl.load(tm+r*(DEPTH+1)*TS+AX+axis,r<n,0)
        x1=tl.load(tm+(r*(DEPTH+1)+1)*TS+AX+axis,r<n,0)
        tl.store(pos+axis*M+row,tl.where(live,x0+(x1-x0)*(j+1),0))
    padding_page=SCRATCH_PAGE+(row%PADDING_PAGES if PADDING_PAGES else row)
    page_index=cachepos//PAGE
    page=tl.load(ctx+1+r*CS+2+DEPTH+page_index,mask=live,other=padding_page)
    page=tl.where(live,page,padding_page)
    tl.store(ps+row,page);tl.store(pd+row,page)
    tl.store(off+row,cachepos%PAGE);tl.store(lens+row,tl.where(live,cachepos+1,1))
    tl.store(seg+row,tl.where(live&(j==0),a+1,0));tl.store(act+row,live)
    padding=SCRATCH_STATE+row
    if PADDING_ROWS:
        padding=SCRATCH_STATE+row%PADDING_ROWS
    private=SCRATCH_STATE+row
    if CHECKPOINT_CS:
        private=tl.load(checkpoint_ctx+1+r*CHECKPOINT_CS+3+j,
                        (r<n)&(j<DEPTH),padding)
    source=tl.where(live,owner,padding)
    dest=tl.where(live,tl.where(j==a,owner,private),padding)
    layers=tl.arange(0,triton.next_power_of_2(L))
    for field in tl.static_range(len(maps)):
        tl.store(maps[field]+layers*M+row,source*L+layers,layers<L)
        tl.store(maps[field]+(L+layers)*M+row,dest*L+layers,layers<L)
    cols=tl.arange(0,BLOCK)
    for start in range(tl.cdiv(T,BLOCK)):
        column=start*BLOCK+cols
        v=tl.load(ctx+1+r*CS+2+DEPTH+column,mask=live&(column<T),other=padding_page)
        tl.store(table+row*T+column,v,column<T)

@triton.jit
def align_feedback(ctx,dec,target,out,H:tl.constexpr,CS:tl.constexpr,B:tl.constexpr,DEPTH:tl.constexpr=1):
    row=tl.program_id(0);r=row//(DEPTH+1);j=row%(DEPTH+1);h=tl.program_id(1)*B+tl.arange(0,B)
    n=tl.load(ctx);a=tl.load(dec+r*2,r<n,-1);bonus=tl.load(dec+r*2+1,r<n,-1)
    live=(r<n)&(a>=0)&(a<=DEPTH)&(j<=a)&(bonus>=0)
    source=(DEPTH+1)*r+tl.where(j<a,j,a)
    v=tl.load(target+source*H+h,(h<H)&live,0)
    tl.store(out+row*H+h,v,h<H)

@triton.jit
def clear_new_pages(activity,offsets,pages,field,W:tl.constexpr,B:tl.constexpr,
                    STRIDE:tl.constexpr):
    row=tl.program_id(0);x=tl.program_id(1)*B+tl.arange(0,B)
    clear=(tl.load(activity+row)!=0)&(tl.load(offsets+row)==0)
    page=tl.load(pages+row).to(tl.int64)
    tl.store(field+page*STRIDE+x,0,clear&(x<W))

@triton.jit
def select_head(ctx,dec,raw,norm,hraw,hnorm,ord0,ordout,valid,
                H:tl.constexpr,N:tl.constexpr,CS:tl.constexpr,B:tl.constexpr,DEPTH:tl.constexpr=1):
    r=tl.program_id(0);x=tl.program_id(1)*B+tl.arange(0,B)
    a=tl.load(dec+r*2);live=(r<tl.load(ctx))&(a>=0)&(a<=DEPTH)
    source=(DEPTH+1)*r+tl.maximum(a,0)
    v=tl.load(raw+source*H+x,live&(x<H),0);tl.store(hraw+r*H+x,v,x<H)
    v=tl.load(norm+source*N+x,live&(x<N),0);tl.store(hnorm+r*N+x,v,x<N)
    if tl.program_id(1)==0:
        tl.store(ordout+r,tl.load(ord0+r)+a+1)
        tl.store(valid+r,live)

@triton.jit
def pending_copy(ctx,scores,ids,tokens,pscores,pids,ptokens,CS:tl.constexpr,C:tl.constexpr,
                 OWNERS:tl.constexpr,TP:tl.constexpr,K:tl.constexpr,B:tl.constexpr,LOAD:tl.constexpr,
                 SCORE_OWNER:tl.constexpr=0,SCORE_TP:tl.constexpr=0,
                 ID_OWNER:tl.constexpr=0,ID_TP:tl.constexpr=0,TOKEN_OWNER:tl.constexpr=1):
    r=tl.program_id(0);x=tl.arange(0,B);n=tl.load(ctx);owner=tl.load(ctx+1+r*CS+1).to(tl.int64)
    for tp in tl.static_range(TP):
        if LOAD:
            s=tl.load(pscores+tp*(SCORE_TP if SCORE_TP else OWNERS*K)+owner*(SCORE_OWNER if SCORE_OWNER else K)+x,(r<n)&(x<K),float('nan'))
            i=tl.load(pids+tp*(ID_TP if ID_TP else OWNERS*K)+owner*(ID_OWNER if ID_OWNER else K)+x,(r<n)&(x<K),-1)
            tl.store(scores+(tp*C+r)*K+x,s,x<K);tl.store(ids+(tp*C+r)*K+x,i,x<K)
        else:
            s=tl.load(scores+(tp*C+r)*K+x,x<K,0);i=tl.load(ids+(tp*C+r)*K+x,x<K,-1)
            tl.store(pscores+tp*(SCORE_TP if SCORE_TP else OWNERS*K)+owner*(SCORE_OWNER if SCORE_OWNER else K)+x,s,(r<n)&(x<K))
            tl.store(pids+tp*(ID_TP if ID_TP else OWNERS*K)+owner*(ID_OWNER if ID_OWNER else K)+x,i,(r<n)&(x<K))
    if LOAD:
        tok=tl.load(ptokens+owner*TOKEN_OWNER,r<n,-1);tl.store(tokens+r,tok)
    else: tl.store(ptokens+owner*TOKEN_OWNER,tl.load(tokens+r),r<n)

@triton.jit
def publish_feedback(activity, owners, source, durable, H:tl.constexpr,
                     OWNER_STRIDE:tl.constexpr, DURABLE_STRIDE:tl.constexpr, B:tl.constexpr):
    row=tl.program_id(0); h=tl.program_id(1)*B+tl.arange(0,B)
    live=tl.load(activity+row)!=0
    owner=tl.load(owners+row*OWNER_STRIDE,live,0).to(tl.int64)
    value=tl.load(source+row*H+h,live&(h<H),0)
    tl.store(durable+owner*DURABLE_STRIDE+h,value,live&(h<H))


@triton.jit
def merge_prefix(ctx,decisions,out,C:tl.constexpr,DEPTH:tl.constexpr,B:tl.constexpr):
    r=tl.arange(0,B);live=(r<C)&(r<tl.load(ctx))
    accepted=tl.full((B,),DEPTH,tl.int64)
    bonus=tl.load(decisions+((DEPTH-1)*C+r)*2+1,r<C,-1)
    good=tl.full((B,),True,tl.int1)
    for d in tl.static_range(DEPTH-1,-1,-1):
        a=tl.load(decisions+(d*C+r)*2,r<C,-1)
        tok=tl.load(decisions+(d*C+r)*2+1,r<C,-1)
        accepted=tl.where(a==0,d,accepted);bonus=tl.where(a==0,tok,bonus)
        good=good&((a==0)|(a==1))&(tok>=0)
    tl.store(out+r*2,tl.where(live&good,accepted,-1),r<C)
    tl.store(out+r*2+1,tl.where(live&good,bonus,-1),r<C)

@triton.jit
def joint_result(ctx,dec,q,guard,out,C:tl.constexpr,CS:tl.constexpr,VOCAB:tl.constexpr,
                 B:tl.constexpr,DEPTH:tl.constexpr=1):
    r=tl.arange(0,B);live=(r<C)&(r<tl.load(ctx))
    a=tl.load(dec+r*2,r<C,-1);bonus=tl.load(dec+r*2+1,r<C,-1)
    good=(a>=0)&(a<=DEPTH)&(bonus>=0)&(bonus<VOCAB)&tl.load(guard)
    tl.store(out+r*(DEPTH+3),a,r<C);tl.store(out+r*(DEPTH+3)+1,bonus,r<C)
    for d in tl.static_range(DEPTH):
        nextq=tl.load(q+d*C+r,r<C,-1)
        good=good&(nextq>=0)&(nextq<VOCAB)
        tl.store(out+r*(DEPTH+3)+2+d,nextq,r<C)
    tl.store(out+r*(DEPTH+3)+2+DEPTH,good&live,r<C)

@triton.jit
def advance_round_context(ctx, result, metadata, ordinals, base_ordinals,
                          target_limits, provider_limits, ready,
                          C:tl.constexpr, M:tl.constexpr, CS:tl.constexpr,
                          TS:tl.constexpr, AX:tl.constexpr, PT:tl.constexpr,
                          PAGE:tl.constexpr, START:tl.constexpr, STOP:tl.constexpr,
                          VOCAB:tl.constexpr, DEPTH:tl.constexpr, B:tl.constexpr,
                          EMBED:tl.constexpr=True):
    """Advance an already-owned append transaction from its audited result.

    Limits describe actual owned cache and declared context, never table padding.
    A failed guard or insufficient next-round capacity leaves every input intact.
    This integer ingress operator does not choose owners or allocate/remap pages.
    """
    r=tl.program_id(0);owned=r<tl.load(ctx)
    a=tl.load(result+r*(DEPTH+3),owned,-1)
    bonus=tl.load(result+r*(DEPTH+3)+1,owned,-1)
    valid=owned&(a>=0)&(a<=DEPTH)&(bonus>=0)&(bonus<VOCAB)
    valid=valid&(tl.load(result+r*(DEPTH+3)+DEPTH+2,owned,0)!=0)
    j=tl.arange(0,B)
    q=tl.load(result+r*(DEPTH+3)+2+j,owned&(j<DEPTH),0)
    valid=valid&(tl.sum(((j<DEPTH)&((q<0)|(q>=VOCAB))).to(tl.int32),0)==0)
    delta=a+1;row=r*(DEPTH+1);root=row*TS
    provider_pos=tl.load(ctx+1+r*CS,owned,0)
    provider_end=provider_pos.to(tl.int64)+delta+2*DEPTH
    pos=tl.load(metadata+(row+j)*TS,owned&(j<DEPTH+1),0).to(tl.int64)+delta
    target_limit=tl.load(target_limits+r,owned,0)
    bad_position=(j<DEPTH+1)&((pos<0)|(pos>=target_limit)|(pos>=2147483647)
                                             |(pos//PAGE>=AX-PT))
    can_advance=valid&(tl.sum(bad_position.to(tl.int32),0)==0)
    can_advance=can_advance&(provider_end<tl.load(provider_limits+r,owned,0))
    can_advance=can_advance&(provider_pos>=0)&(provider_end<2147483647)
    for axis in tl.static_range(3):
        x0=tl.load(metadata+root+AX+axis,owned,0).to(tl.int64)
        x1=tl.load(metadata+root+TS+AX+axis,owned,0).to(tl.int64)
        old=tl.load(metadata+(row+j)*TS+AX+axis,owned&(j<DEPTH+1),0).to(tl.int64)
        following=old+delta*(x1-x0)
        bad_axis=(j<DEPTH+1)&((following<0)|(following>2147483647))
        can_advance=can_advance&(tl.sum(bad_axis.to(tl.int32),0)==0)
    ordinal=tl.load(base_ordinals+r,owned,0)
    can_advance=can_advance&(ordinal>=0)&(ordinal<=9223372036854775807-delta)
    tl.store(ready+r,can_advance)
    page=tl.load(metadata+(row+j)*TS+PT+pos//PAGE,
                 can_advance&(j<DEPTH+1),0)
    token=tl.where(j==0,bonus,tl.load(result+r*(DEPTH+3)+1+j,
                                  owned&(j>0)&(j<DEPTH+1),0))
    live=can_advance&(j<DEPTH+1)
    local=(token>=START)&(token<STOP)
    tl.store(metadata+(row+j)*TS,pos,live)
    tl.store(metadata+(row+j)*TS+1,page,live)
    tl.store(metadata+(row+j)*TS+2,page,live)
    tl.store(metadata+(row+j)*TS+3,pos%PAGE,live)
    tl.store(metadata+(row+j)*TS+4,pos+1,live)
    if EMBED:
        tl.store(metadata+(row+j)*TS+5,tl.where(local,token-START,0),live)
        tl.store(metadata+(row+j)*TS+6,local,live)
    for axis in tl.static_range(3):
        x0=tl.load(metadata+root+AX+axis,owned,0)
        x1=tl.load(metadata+root+TS+AX+axis,owned,0)
        old=tl.load(metadata+(row+j)*TS+AX+axis,owned&(j<DEPTH+1),0)
        tl.store(metadata+(row+j)*TS+AX+axis,old+delta*(x1-x0),live)
    tl.store(ctx+1+r*CS,provider_pos+delta,can_advance)
    tl.store(ctx+1+r*CS+2+j,q,can_advance&(j<DEPTH))
    tl.store(ordinals+r,ordinal+delta,can_advance)


@triton.jit
def autoregressive_meta(ctx,dec,tm,q,ids,mask,pos,ps,pd,off,lens,seg,act,maps,table,
                        C:tl.constexpr,T:tl.constexpr,CS:tl.constexpr,TS:tl.constexpr,
                        AX:tl.constexpr,L:tl.constexpr,START:tl.constexpr,STOP:tl.constexpr,
                        PAGE:tl.constexpr,SCRATCH_PAGE:tl.constexpr,SCRATCH_STATE:tl.constexpr,
                        VOCAB:tl.constexpr,DEPTH:tl.constexpr,STEP:tl.constexpr,BLOCK:tl.constexpr,
                        checkpoint_ctx=None,CHECKPOINT_CS:tl.constexpr=0,PADDING_ROWS:tl.constexpr=0,
               PADDING_PAGES:tl.constexpr=0):
    r=tl.program_id(0);n=tl.load(ctx);a=tl.load(dec+r*2)
    token=tl.load(q+r);live=(r<n)&(a>=0)&(a<=DEPTH)&(token>=0)&(token<VOCAB)
    owner=tl.load(ctx+1+r*CS+1).to(tl.int64);rootpos=tl.load(ctx+1+r*CS)
    cachepos=tl.where(live,rootpos+a+1+STEP,0)
    local=live&(token>=START)&(token<STOP)
    tl.store(ids+r,tl.where(local,token-START,0));tl.store(mask+r,local)
    for axis in tl.static_range(3):
        x0=tl.load(tm+r*(DEPTH+1)*TS+AX+axis,r<n,0)
        x1=tl.load(tm+(r*(DEPTH+1)+1)*TS+AX+axis,r<n,0)
        tl.store(pos+axis*C+r,tl.where(live,x0+(x1-x0)*(a+1+STEP),0))
    padding_page=SCRATCH_PAGE+(r%PADDING_PAGES if PADDING_PAGES else r)
    page=tl.load(ctx+1+r*CS+2+DEPTH+cachepos//PAGE,live,padding_page)
    tl.store(ps+r,page);tl.store(pd+r,page);tl.store(off+r,cachepos%PAGE)
    tl.store(lens+r,tl.where(live,cachepos+1,1));tl.store(seg+r,live);tl.store(act+r,live)
    private=SCRATCH_STATE+r*(DEPTH+1)
    source=tl.where(live,owner if STEP==1 else private+STEP-2,private)
    dest=private+STEP-1
    if CHECKPOINT_CS:
        padding=SCRATCH_STATE+r
        if PADDING_ROWS:
            padding=SCRATCH_STATE+r%PADDING_ROWS
        dest=tl.load(checkpoint_ctx+1+r*CHECKPOINT_CS+3+STEP-1,live,padding)
        if STEP==1:
            source=tl.where(live,owner,padding)
        else:
            source=tl.load(checkpoint_ctx+1+r*CHECKPOINT_CS+3+STEP-2,live,padding)
        dest=tl.where(live,dest,padding)
    layers=tl.arange(0,triton.next_power_of_2(L))
    for f in tl.static_range(len(maps)):
        tl.store(maps[f]+layers*C+r,source*L+layers,layers<L)
        tl.store(maps[f]+(L+layers)*C+r,dest*L+layers,layers<L)
    cols=tl.arange(0,BLOCK)
    for start in range(tl.cdiv(T,BLOCK)):
        column=start*BLOCK+cols
        v=tl.load(ctx+1+r*CS+2+DEPTH+column,live&(column<T),padding_page)
        tl.store(table+r*T+column,v,column<T)

@dataclass(slots=True)
class DraftPendingPool:
    """Bounded pending history and pre-bonus rollback for durable owners only."""

    depth: int
    owners: int
    metadata: list[Any]
    scores: torch.Tensor
    ids: torch.Tensor
    tokens: torch.Tensor
    backup: dict[str, torch.Tensor]

    @staticmethod
    def fields(provider):
        """Declare auxiliary bytes from the actual provider operand contract."""
        from cubic_serve.runtime.cache.shared_pool import SharedCacheField
        workspace = provider._sampling.workspace
        depth = provider._max_draft_tokens
        tp, _, candidates = workspace.gathered_scores.shape
        return (SharedCacheField("pending.scores",workspace.gathered_scores.dtype,(depth,tp,candidates)),
                SharedCacheField("pending.ids",workspace.gathered_token_ids.dtype,(depth,tp,candidates)),
                SharedCacheField("pending.tokens",torch.int64,(depth,)),
                *(SharedCacheField("pending.backup."+name,value.dtype,tuple(value.shape[1:]))
                  for name,value in provider.state_fields.items()))

    @classmethod
    def allocate(cls, provider, *, pricing_owner_capacity=None):
        workspace = provider._sampling.workspace
        if workspace is None:
            raise ValueError("draft pending pool requires native candidate storage")
        owners = provider.body.arena.spec.state_slot_capacity
        depth = provider._max_draft_tokens
        device = provider.body.hidden.device
        shape = (depth, workspace.gathered_scores.shape[0], owners,
                 workspace.gathered_scores.shape[2])
        storage = getattr(provider.body.arena, "shared_storage", None)
        if pricing_owner_capacity is not None:
            if (type(pricing_owner_capacity) is not int or pricing_owner_capacity < 1
                    or storage is None or storage.extra_state_fields):
                raise ValueError("pricing owner extent requires an ownerless bootstrap schema")
            owners = pricing_owner_capacity
            shape = (depth, workspace.gathered_scores.shape[0], owners,
                     workspace.gathered_scores.shape[2])
        # Bootstrap arenas declare no engine fields and keep their auxiliary
        # allocation private until final sizing. A declared engine schema is
        # complete: never allocate missing final-pool fields outside its budget.
        if storage is not None and storage.extra_state_fields:
            fields = storage.extra_states
            if any(field.name not in fields for field in cls.fields(provider)):
                raise ValueError("shared pending pool omits declared provider fields")
            return cls(depth, owners, [None] * owners,
                       fields["pending.scores"][:owners].permute(1, 2, 0, 3),
                       fields["pending.ids"][:owners].permute(1, 2, 0, 3),
                       fields["pending.tokens"][:owners].transpose(0, 1),
                       {name: fields["pending.backup." + name][:owners]
                        for name in provider.state_fields})
        return cls(
            depth, owners, [None] * owners,
            torch.empty(shape, device=device, dtype=workspace.gathered_scores.dtype),
            torch.empty(shape, device=device, dtype=workspace.gathered_token_ids.dtype),
            torch.empty((depth, owners), device=device, dtype=torch.int64),
            {name: torch.empty((owners, *field.shape[1:]), dtype=field.dtype, device=field.device)
             for name, field in provider.state_fields.items()},
        )

    def prefix(self, depth):
        """Alias finite proposal planes; ownership and rollback stay shared."""
        if type(depth) is not int or not 1 <= depth <= self.depth:
            raise ValueError("draft prefix exceeds its allocated history")
        if depth == self.depth:
            return self
        return DraftPendingPool(depth, self.owners, self.metadata,
                                self.scores[:depth], self.ids[:depth], self.tokens[:depth],
                                self.backup)

    def is_prefix_of(self, pool):
        return (self.owners == pool.owners and 1 <= self.depth <= pool.depth
                and self.metadata is pool.metadata and self.backup is pool.backup
                and all(own.is_set_to(full[:self.depth]) for own, full in (
                    (self.scores, pool.scores), (self.ids, pool.ids), (self.tokens, pool.tokens))))

    def restore(self, provider, rows):
        """Return to the unconsumed bonus boundary before another draft strategy."""
        for owner, row in {row.state_source_slot: row for row in rows}.items():
            if not 0 <= owner < self.owners:
                continue
            pending = self.metadata[owner]
            if pending is not None and pending[0] == row.request_id:
                for name, field in provider.state_fields.items():
                    field[owner].copy_(self.backup[name][owner])
            self.metadata[owner] = None


def _launch_body(stage, *, produce_logits=True):
    """Reuse sealed bodies while composing; legacy builders keep their ABI."""
    owned = getattr(stage, "captured_body_graph", None)
    if owned is None:
        return stage.launch() if produce_logits else stage.launch(produce_logits=False)
    graph = owned(produce_logits=produce_logits)
    if torch.cuda.is_current_stream_capturing():
        from cubic_serve.executor.resident_window import append_captured_graph
        append_captured_graph(graph)
    else:
        graph.replay()


def prepare_state_transfers(providers, targets, lifetime=None, *, qualification=None):
    """Bind optional row transport before final KV admission, for all providers.

    A field layout selects one complete-range cold decision. Empty/stateless
    providers use the same target/feedback contract without a fictitious state.
    The original direct transfer remains the qualified transport for layouts
    whose measured critical path favors DMA.
    """
    if not providers or not targets:
        return

    def signature(fields):
        return tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
                     for t in fields)

    existing = {}
    for target in targets.values():
        existing.update(getattr(target, "_state_transfer_instances", {}))
    provider_fields = []
    for provider in providers.values():
        fields = provider.state_fields
        provider_fields.append((fields, id(fields) if fields else 0))
    for target in targets.values():
        hidden = target._target_hidden_slots
        if hidden is None:
            raise ValueError("state transfer lost target feedback storage")
        # Target-only callers have no provider storage to transfer. Preserve
        # their existing direct-copy contract without allocating another pool.
        bindings = {0: None}
        for provider, identity in provider_fields:
            fields = (*target.body.arena.states.values(), *provider.values(), hidden)
            key = signature(fields)
            if key not in existing:
                maximum = min(target.body.arena.spec.state_slot_capacity,
                              *(t.shape[0] // 2 for t in fields))
                geometry = tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in fields)
                selected = []
                def select():
                    selected.append(CapturedStateTransfer.select(fields, maximum))
                    return selected[0] is not None
                if qualification is None:
                    select()
                    plan = selected[0]
                else:
                    enabled = qualification.run("state-transfer-choice", (geometry, maximum), select)
                    plan = (selected[0] if selected else
                            CapturedStateTransfer(fields, maximum) if enabled else None)
                if plan is not None:
                    if qualification is None:
                        plan.qualify()
                    else:
                        qualification.component(plan, (geometry, maximum))
                existing[key] = plan
                if plan is not None and lifetime is not None:
                    lifetime.callback(plan.close)
            bindings[identity] = existing[key]
        target._state_transfer_bindings = bindings
        target._state_transfer_instances = existing
        target._state_transfer_state_fields = target.body.arena.states
        target._state_transfer_hidden_slots = hidden


def capture_draft_rounds(providers, targets, *, dynamic_depth=False, lifetime=None, qualify_prompts=False, retained_rounds=None, qualification=None, enforce_capacity=True, pricing_owner_capacity=None, progress=None):
    """Price and retain the same declared-depth portfolio before readiness.

    Capability selection is structural. PP communication is owned by a stage's
    replay contract; a body without an inline PP contract keeps its existing
    sealed proposal/review graphs. No eager or runtime-capture path is added.
    """
    if pricing_owner_capacity is not None and (enforce_capacity or retained_rounds is not None):
        raise ValueError("pricing owner extent cannot bind a serving portfolio")
    if providers and targets and lifetime is not None:
        prepare_state_transfers(providers, targets, lifetime, qualification=qualification)
        capture_draft_prompt_windows(providers, targets, lifetime, qualify=qualify_prompts,
                                     reuse=retained_rounds is not None, qualification=qualification,
                                     progress=progress)
    if not providers:
        return {}
    first = providers[min(providers)]
    topology = getattr(first.body, "topology", None)
    if (topology is None or topology.pp_size != 1
        or not callable(getattr(first.body, "launch_projection", None))):
        return {}
    pool = (getattr(first, "_pending_draft_pool", None) if retained_rounds is not None else
            DraftPendingPool.allocate(first) if pricing_owner_capacity is None else
            DraftPendingPool.allocate(first, pricing_owner_capacity=pricing_owner_capacity))
    if pool is None or retained_rounds is not None and any(
            getattr(provider, "_pending_draft_pool", None) is not pool
            for provider in providers.values()):
        raise ValueError("retained draft rounds lost their common pending ownership")
    # Every declared dynamic depth shares the same bounded pool and body
    # portfolio; changing depth must not lose the complete-round composition.
    depths = tuple(range(1, pool.depth + 1)) if dynamic_depth else (pool.depth,)
    owner_limit=min(pool.owners,getattr(first,"_max_request_owners",pool.owners))
    depth_limits = {depth: owner_limit for depth in depths}
    storage = getattr(getattr(first.body, "arena", None), "shared_storage", None)
    if enforce_capacity and storage is not None:
        block = first.body.arena.layout.block_size_tokens
        limits = [storage.allocator.cold_owner_capacity(state_slots=depth + 1,
                  pages=(8 * (depth + 1) + block - 1) // block) for depth in depths]
        # PP1 was checked above: every rank binds the identical round schedule.
        if torch.distributed.is_initialized():
            common = torch.tensor(limits, dtype=torch.int64, device=first.body.hidden.device)
            all_reduce(common, op=torch.distributed.ReduceOp.MIN)
            limits = common.tolist()
        depth_limits = dict(zip(depths, limits, strict=True))
    expected = {}
    for depth in depths:
        for capacity, head in sorted(providers.items()):
            if capacity & (capacity - 1) or capacity > min(owner_limit, depth_limits[depth]):
                continue
            review = triton.next_power_of_2(capacity * (depth + 1))
            if review in targets and review in providers:
                expected[(review, depth)] = head
    if retained_rounds is not None:
        if set(retained_rounds) != set(expected) or any(
                round_.target is not targets[review] or round_.align is not providers[review]
                or round_.head is not expected[(review, depth)]
                or not round_.pool.is_prefix_of(pool) or round_.depth != depth
                for (review, depth), round_ in retained_rounds.items()):
            raise ValueError("retained draft round portfolio differs from startup-bound resources")
        return retained_rounds
    views = {depth: pool.prefix(depth) for depth in depths}
    rounds = {}
    if progress is not None:
        progress(0, len(expected), phase="rounds")
    for (review, depth), head in sorted(expected.items()):
        capacity = head.captured_m
        rounds[(review, depth)] = CapturedDraftRound(
            targets[review], providers[review], head, views[depth], depth=depth,
            retain_audit_graph=bool(getattr(qualification, "engineering", False)),
        )
        if lifetime is not None:
            for graph in rounds[(review, depth)]._review_graphs.values():
                lifetime.callback(graph.reset)
            lifetime.callback(rounds[(review, depth)].graph.reset)
            lifetime.callback(rounds[(review, depth)].candidate_graph.reset)
            lifetime.callback(rounds[(review, depth)].grammar_review_graph.reset)
            for graph in rounds[(review, depth)].target_constraint_graphs.values():
                lifetime.callback(graph.reset)
            for graphs in rounds[(review, depth)].grammar_proposal_graphs.values():
                for graph in graphs:
                    lifetime.callback(graph.reset)
            from cubic_serve.executor.resident_window import SharedResidentJournal
            from cubic_serve.runtime.continuous_batch import RESIDENT_STEP_CAPACITY
            elements = 64 + RESIDENT_STEP_CAPACITY * (capacity * (2 * depth + 8) + 6)
            journal = SharedResidentJournal(elements, targets[review].body.hidden.device)
            lifetime.callback(journal.close)
            round_ = rounds[(review, depth)]
            for graph in round_._forward_graphs.values():
                lifetime.callback(graph.reset)
            for candidates, attribute in ((False, "_resident_window"), (True, "_candidate_window")):
                round_._capture_candidates = candidates
                window = CapturedDraftWindow(round_, journal=journal)
                lifetime.callback(window.close)
                setattr(round_, attribute, window)
            if round_.budget is not None:
                round_._capture_budget = True
                for candidates, attribute in ((False, "_budget_resident_window"),
                                               (True, "_budget_candidate_window")):
                    round_._capture_candidates = candidates
                    window = CapturedDraftWindow(round_, journal=journal)
                    lifetime.callback(window.close)
                    setattr(round_, attribute, window)
                lifetime.callback(round_.budget_graph.reset)
                lifetime.callback(round_.budget_candidate_graph.reset)
                round_._capture_budget = False
            round_._capture_candidates = False
            round_.capture_constraint_continuation()
            lifetime.callback(round_.constraint_continuation_forward_graph.reset)
        if progress is not None:
            progress(len(rounds), len(expected), phase="rounds")
    if rounds:
        for provider in providers.values():
            provider._pending_draft_pool = pool
        for (review, depth), round_ in rounds.items():
            executor = targets[review]
            portfolio = getattr(executor, "_pipelined_draft_rounds", None)
            if portfolio is None:
                portfolio = executor._pipelined_draft_rounds = {}
            portfolio[depth] = round_
            executor._pipelined_draft_round = round_
    return rounds


class CapturedDraftRound:
    def __init__(self,target,align,head,pool,*,depth=1,head_projection=None,
                 retain_audit_graph=False):
        if type(retain_audit_graph) is not bool:
            raise ValueError("audit graph retention requires an explicit boolean")
        # Offline witnesses inspect the actual DAG and remove temporary
        # immutable observers before releasing its exclusive ownership.
        self._retain_audit_graph = retain_audit_graph
        if (type(depth) is not int or depth < 1
            or any(type(stage.captured_m) is not int or stage.captured_m <= 0
                or stage.captured_m & (stage.captured_m - 1)
                for stage in (target, align, head))
            or target.captured_m != align.captured_m
            or target.captured_m != triton.next_power_of_2((depth+1)*head.captured_m)
            or target._rank != align._rank or target._rank != head._rank):
            raise ValueError("draft round differs from its captured review/provider ABI")
        projection = head_projection or getattr(head.body, "launch_projection", None)
        if not callable(projection):
            raise ValueError("draft provider has no bound head-only projection")
        # Two sampling algorithms share one transaction implementation, storage
        # and owner lease. This finite startup portfolio preserves the native
        # argmax lower bound without forcing top-k work on greedy requests.
        self._capture_candidates = False
        self._capture_budget = False
        self._capture_review_only = False
        self._capture_target_constraints = False
        self._review_graphs = {}
        self._review_capture_open = True
        self._last_target_constraints = False
        self.target_constraints_qualified = False
        self.constraint_continuation_qualified = False
        self._constraint_forward_inflight = False
        self._constraint_continuation_binding = None
        self._last_sampling_is_greedy = True
        self._last_budget = False
        self._last_review_only = False
        self.grammar_qualified = False
        self.depth=depth;self.stride=depth+1;self.provider_lookahead_tokens=depth
        self.target,self.align,self.head,self.pool=target,align,head,pool
        self.M=target.captured_m;self.C=head.captured_m
        self.T=align.body.buffers.page_table.shape[1];self.CS=2+depth+self.T;self.commit_CS=3+depth
        self.target_T=target.body.buffers.page_table.shape[1]
        dev=target.body.hidden.device
        self.ctx=torch.zeros(1+self.C*self.CS,dtype=torch.int32,device=dev)
        self.host=torch.zeros_like(self.ctx,device='cpu',pin_memory=True)
        self.commit_ctx=torch.zeros(1+self.C*self.commit_CS,dtype=torch.int32,device=dev)
        self.commit_host=torch.zeros_like(self.commit_ctx,device='cpu',pin_memory=True)
        self.result=torch.empty((self.C,depth+3),dtype=torch.int64,device=dev)
        self.host_result=torch.empty_like(self.result,device='cpu',pin_memory=True)
        self.host_decisions=torch.empty((self.C, 2),dtype=torch.int64,device='cpu',pin_memory=True)
        self._last_transfer=None
        self.host_array=self.host.numpy()
        self.commit_host_array=self.commit_host.numpy()
        self.metadata_host=torch.empty_like(target._metadata_hosts[0],pin_memory=True)
        self.index_host=torch.empty(self.M,dtype=torch.int64,pin_memory=True)
        self.index_device=torch.empty_like(self.index_host,device=dev)
        self.param_hosts=tuple(torch.empty(self.C,dtype=x.dtype,pin_memory=True)
                               for x in (head._sampling.temperatures,head._sampling.top_ks,
                                         head._sampling.top_ps,head._sampling.seeds,head._sampling.ordinals))
        self.param_arrays=tuple(x.numpy() for x in self.param_hosts)
        self.active=torch.empty(self.C,dtype=torch.bool,device=dev)
        self.rows=tuple(torch.arange(self.C,device=dev,dtype=torch.int32)*self.stride+d for d in range(depth))
        self.decisions=torch.empty((self.C,2),device=dev,dtype=torch.int64)
        self.next_tokens=torch.empty((depth,self.C),device=dev,dtype=torch.int64)
        # Provider state after the pending root remains in its durable owner;
        # autoregressive proposals write only to lease-private checkpoints.
        self.pool_scores=pool.scores.unsqueeze(0) if pool.scores.ndim==3 else pool.scores
        self.pool_ids=pool.ids.unsqueeze(0) if pool.ids.ndim==3 else pool.ids
        self.pool_tokens=pool.tokens.unsqueeze(0) if pool.tokens.ndim==1 else pool.tokens
        if (self.pool_scores.shape[0]!=depth or self.pool_ids.shape!=self.pool_scores.shape
            or self.pool_tokens.shape!=(depth,pool.owners)
            or depth>head._verification_draft_tokens.shape[0]):
            raise ValueError("draft pending history differs from declared depth")
        self.valid=torch.empty((),device=dev,dtype=torch.bool)
        self.constraint_forward_valid=torch.ones((),device=dev,dtype=torch.bool)
        self.round_start=torch.cuda.Event(enable_timing=True,external=True);self.round_end=torch.cuda.Event(enable_timing=True,external=True)
        # CUDA graph event nodes require their original event objects to live.
        # Different K compositions may share this target executor.
        self.review_started=target._replay_started=torch.cuda.Event(enable_timing=True,external=True)
        self.review_finished=target._replay_finished=torch.cuda.Event(enable_timing=True,external=True)
        self.target_greedy=target._sampling.greedy_plan(target.slot.logits)
        self.head_greedy=head._sampling.greedy_plan(head.slot.logits)
        if self.target_greedy is None or self.head_greedy is None: raise RuntimeError('round sampler missing')
        self.budget = getattr(getattr(target._sampling, "_constraint_arena", None), "budget", None)
        self.budget_qualified = False
        self.budget_graph = self.budget_candidate_graph = None
        self.target_masked = next(p for p in target._sampling.masked_plans
                                  if p.local_logits is target.slot.logits)
        self.head_masked = next(p for p in head._sampling.masked_plans
                                if p.local_logits is head.slot.logits)
        self.target_masked.metadata[1].zero_()
        self.head_masked.metadata[1].zero_()
        if self.budget is not None:
            self.budget_review = self.budget.prefix(
                self.ctx, self.next_tokens, owners=self.C, depth=depth,
                vocabulary=target.vocab_size,
            )
            self.budget_heads = tuple(self.budget.prefix(
                self.ctx, self.next_tokens, owners=self.C, depth=depth,
                vocabulary=target.vocab_size, head_step=d,
            ) for d in range(depth))
            self.budget_commit = self.budget.commit(
                self.ctx, self.decisions, owners=self.C, depth=depth,
                vocabulary=target.vocab_size,
            )
        self.head_launch=projection
        self.head_greedy.metadata[1].zero_()
        self.target_greedy.metadata[1].zero_()
        self.scratch_state=align._draft_binding.executor._scratch_states[0]
        self.scratch_page=align._draft_binding.executor._scratch_pages[0]
        self.padding_pages=align.body.arena.spec.page_scratch_capacity_per_arena
        # Compile and prove capture identity before readiness, private scratch only.
        target.body.prepare_scratch(target.resource);align.body.prepare_scratch(align.body.resource)
        self.host[0]=0
        arr=self.host[1:].view(self.C,self.CS);arr[:,1]=torch.arange(self.C,dtype=torch.int32)
        for r in range(self.C): arr[r,2+depth:].fill_(self.padding_page(self.stride*r))
        cc=self.commit_host[1:].view(self.C,self.commit_CS);cc[:,1]=torch.arange(self.C,dtype=torch.int32);cc[:,2]=self.scratch_state+torch.arange(self.C)*self.stride
        for d in range(depth):cc[:,3+d]=target._scratch_states[0]+torch.arange(self.C)*self.stride+d
        self.commit_host[0]=0
        target._metadata_device[:,target._position_axes_slice].zero_()
        target._metadata_device[:,target._state_destination_column].copy_(torch.tensor(target._scratch_states,device=dev,dtype=torch.int32))
        target._metadata_device.zero_()
        target._metadata_device[:,target._state_destination_column].copy_(torch.tensor(target._scratch_states,device=dev,dtype=torch.int32))
        source=torch.tensor(target._scratch_states,device=dev,dtype=torch.int32)*target._local_state_layers.numel()
        offsets=target._local_state_layers[None,:]
        target._metadata_device[:,target._source_slots_slice].copy_(source[:,None]+offsets)
        target._metadata_device[:,target._destination_slots_slice].copy_(source[:,None]+offsets)
        target._metadata_device[:,1].copy_(torch.tensor(target._scratch_pages,device=dev,dtype=torch.int32))
        target._metadata_device[:,2].copy_(target._metadata_device[:,1])
        target._metadata_device[:,4].fill_(1)
        target._metadata_device[:,target._page_table_slice].copy_(target._metadata_device[:,1:2])
        self.metadata_host.copy_(target._metadata_device.cpu())
        self.index_host.copy_(torch.tensor(target._scratch_pages,dtype=torch.int64))
        for array,value in zip(self.param_arrays,(0.,1,1.,0,0),strict=True):array.fill(value)
        self.ctx.copy_(self.host);self.commit_ctx.copy_(self.commit_host)
        # delta q on token 0 permits a private startup fixture.
        head._verification_draft_tokens.zero_();head._verification_q_scores.fill_(float('nan'));head._verification_q_token_ids.fill_(-1)
        head._verification_q_scores[:depth,:, :,0]=0;head._verification_q_token_ids[:depth,:, :,0]=0
        head._sampling.temperatures.zero_();head._sampling.top_ks.fill_(1);head._sampling.top_ps.fill_(1);head._sampling.seeds.zero_();head._sampling.ordinals.zero_()
        target._sampling.temperatures.zero_();target._sampling.top_ks.fill_(1);target._sampling.top_ps.fill_(1);target._sampling.seeds.zero_();target._sampling.ordinals.zero_()
        for src,dst in [(head._sampling.ordinals,head._verification_ordinals[0]),(head._sampling.temperatures,head._verification_temperatures[0]),(head._sampling.top_ps,head._verification_top_ps[0]),(head._sampling.top_ks,head._verification_top_ks[0]),(head._sampling.seeds,head._verification_seeds[0])]:dst.copy_(src)
        # Mutually exclusive variants on this stream may reuse temporary
        # capture allocations. Review children get a separate pool: a parent
        # temporary can remain live across a nested review, whose own temporary
        # can remain live across a model child. Model graphs keep their pools.
        # Exported results/state are allocated before capture; the portfolio
        # lease covers replay through durable output consumption.
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._review_graph_pool = torch.cuda.graph_pool_handle()
        stream=target.slot.stream;stream.wait_stream(torch.cuda.current_stream(dev))
        self._forward_graphs = {}
        for budget in ((False, True) if self.budget is not None else (False,)):
            self._capture_budget = budget
            self.capture_forward_graph(stream)
            for candidates in (False, True):
                self._capture_candidates = candidates
                with torch.cuda.stream(stream):
                    self.body()
                    self.body()
                stream.synchronize()
                barrier()
                self.capture_review_graph(stream)
                graph = torch.cuda.CUDAGraph(keep_graph=self._retain_audit_graph)
                with torch.cuda.graph(graph, stream=stream, pool=self._graph_pool):
                    self.completion_body()
                if self._retain_audit_graph:
                    graph.instantiate()
                stream.synchronize()
                barrier()
                attribute = ("budget_" if budget else "") + (
                    "candidate_graph" if candidates else "graph")
                setattr(self, attribute, graph)
        self._capture_candidates = False
        self._capture_budget = False
        self._capture_review_only = True
        self._capture_candidates = True
        with torch.cuda.stream(stream):
            self.body()
            self.body()
        stream.synchronize()
        barrier()
        self.capture_review_graph(stream)
        self.grammar_review_graph = torch.cuda.CUDAGraph(keep_graph=self._retain_audit_graph)
        with torch.cuda.graph(self.grammar_review_graph, stream=stream, pool=self._graph_pool):
            self.completion_body()
        if self._retain_audit_graph:
            self.grammar_review_graph.instantiate()
        stream.synchronize()
        barrier()
        self.capture_grammar_proposals(stream)
        self._capture_review_only = False
        # Target-only constraints retain the actual unmasked q distribution.
        # Forward does not read a grammar mask. Seal it separately so the host
        # matcher can prepare the reachable-prefix masks while it is running.
        self._capture_target_constraints = True
        self.target_constraint_forward_graph = self._forward_graphs[False]
        self.target_constraint_graphs = {}
        for candidates in (False, True):
            self._capture_candidates = candidates
            with torch.cuda.stream(stream):
                self.body()
                self.body()
            stream.synchronize()
            barrier()
            self.capture_review_graph(stream)
            graph = torch.cuda.CUDAGraph(keep_graph=self._retain_audit_graph)
            with torch.cuda.graph(graph, stream=stream, pool=self._graph_pool):
                self.completion_body()
            if self._retain_audit_graph:
                graph.instantiate()
            stream.synchronize()
            barrier()
            self.target_constraint_graphs[candidates] = graph
        self._capture_target_constraints = False
        self._capture_candidates = False
        # All retained runtime storage has stable addresses, capture is never repeated.
        self.completed=torch.cuda.Event()
        self.ctx.zero_();self.commit_ctx.zero_()
        # Warmup/capture writes shared sampler metadata without going through
        # its cached host ingress. The next ordinary sealed replay must fill
        # that metadata even if its logical count matches an earlier replay.
        self.head_greedy.invalidate_metadata()
        self.target_greedy.invalidate_metadata()
        self.head_masked.invalidate_metadata()
        self.target_masked.invalidate_metadata()
        self._review_capture_open = False

    def capture_grammar_proposals(self, stream):
        """Seal finite proposal phases around the CPU matcher's token dependency.

        The phase reuses the ordinary device metadata, provider, sampler and
        batched owner-copy rule. Host grammar work remains between phases; no
        request-time model launch or per-owner CUDA submission is introduced.
        """
        self.grammar_host_tokens = torch.empty_like(self.next_tokens, device="cpu", pin_memory=True)
        self.grammar_proposal_completed = torch.cuda.Event()
        self.grammar_proposal_graphs = {}
        previous = (self._capture_review_only, self._capture_candidates,
                    self.head._sampling.constraint_bound)
        self._capture_review_only = True
        self.head._sampling.constraint_bound = True
        try:
            for candidates in (False, True):
                self._capture_candidates = candidates
                graphs = []
                for depth in range(self.depth):
                    with torch.cuda.stream(stream):
                        self.grammar_proposal_body(depth)
                        self.grammar_proposal_body(depth)
                    stream.synchronize()
                    barrier()
                    graph = torch.cuda.CUDAGraph(keep_graph=self._retain_audit_graph)
                    with torch.cuda.graph(graph, stream=stream, pool=self._graph_pool):
                        self.grammar_proposal_body(depth)
                    if self._retain_audit_graph:
                        graph.instantiate()
                    stream.synchronize()
                    barrier()
                    graphs.append(graph)
                self.grammar_proposal_graphs[candidates] = tuple(graphs)
        finally:
            (self._capture_review_only, self._capture_candidates,
             self.head._sampling.constraint_bound) = previous
        self.head_masked.invalidate_metadata()

    def grammar_proposal_body(self, depth):
        """Capture-only body, with the same rounded feedback and RNG as a round."""
        if depth:
            self.prepare_proposal(depth)
        self.head_masked.metadata[0].copy_(self.ctx[0])
        self.sample_head(depth)
        self.head._verification_validity[depth].copy_(self.head_masked.final_valid)
        if depth:
            self.valid.logical_and_(self.head_masked.final_valid)
        else:
            self.valid.copy_(self.head_masked.final_valid)
        self.head.stash_proposal(depth, self.C)
        self._retain_proposal(depth)
        self.grammar_host_tokens[depth].copy_(self.next_tokens[depth], non_blocking=True)

    def run_grammar_proposal(self, depth, owners, *, stochastic):
        """Replay a sealed phase under the caller's existing topology lease."""
        if (type(depth) is not int or not 0 <= depth < self.depth
                or type(owners) is not int or not 0 < owners <= self.C
                or type(stochastic) is not bool):
            raise RuntimeError("grammar proposal differs from its sealed capacity")
        stream = self.target.slot.stream
        current = torch.cuda.current_stream(self.target.body.hidden.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self.grammar_proposal_graphs[stochastic][depth].replay()
            self.grammar_proposal_completed.record()
        self.grammar_proposal_completed.synchronize()
        current.wait_event(self.grammar_proposal_completed)
        self.head_masked.invalidate_metadata()
        return tuple(int(token) for token in self.grammar_host_tokens[depth, :owners].tolist())

    def copy_fields(self,fields,dests,mode,ctx):
        for name,field in fields.items():
            width=math.prod(field.shape[1:]);tiles=min(triton.cdiv(width,2048),256);grid=(self.C,tiles)
            owner_copy[grid](ctx,self.decisions,field,dests[name],self.commit_CS,width,mode,2048,tiles,self.depth,
                             field.stride(0),dests[name].stride(0))

    def padding_page(self, physical):
        return self.scratch_page + physical % self.padding_pages

    def _checkpoint_arguments(self):
        target = getattr(self.target.body.arena, "shared_storage", None)
        provider = getattr(self.align.body.arena, "shared_storage", None)
        if target is None and provider is None:
            return dict(checkpoint_ctx=None, CHECKPOINT_CS=0, PADDING_ROWS=0)
        if target is not provider:
            raise RuntimeError("draft stages do not share their cache ownership domain")
        return dict(checkpoint_ctx=self.commit_ctx, CHECKPOINT_CS=self.commit_CS,
                    PADDING_ROWS=self.align.body.arena.spec.state_scratch_capacity_per_arena,
                    PADDING_PAGES=self.padding_pages)

    def _review_key(self):
        return (self._capture_candidates, self._capture_budget,
                self._capture_review_only, self._capture_target_constraints,
                self.target._sampling.constraint_bound, self.head._sampling.constraint_bound)

    def capture_forward_graph(self, stream):
        """Seal one target forward per ingress policy, before any replay.

        Candidate sampling and grammar masks are consumed only by completion.
        The target-constrained path uses this same forward and overlaps its
        execution with host mask preparation. Parent/review/model pools remain
        separate because a parent's temporary may span a nested child.
        """
        if not self._review_capture_open:
            raise RuntimeError("forward capture is closed after construction")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("forward graph must be sealed before parent capture")
        budget = self._capture_budget
        if budget in self._forward_graphs:
            raise RuntimeError("target forward was already captured")
        constrained = self._capture_target_constraints
        self._capture_target_constraints = True
        try:
            with torch.cuda.stream(stream):
                self.forward_body()
                self.forward_body()
            stream.synchronize()
            barrier()
            graph = torch.cuda.CUDAGraph(keep_graph=self._retain_audit_graph)
            with torch.cuda.graph(graph, stream=stream, pool=self._graph_pool):
                self.forward_body()
            if self._retain_audit_graph:
                graph.instantiate()
            stream.synchronize()
            barrier()
            self._forward_graphs[budget] = graph
        finally:
            self._capture_target_constraints = constrained

    def ordinary_phases(self, completion, *, budget=False):
        """The fixed same-stream replay sequence under the existing lease."""
        forward = self._forward_graphs.get(budget)
        if forward is None or completion is None:
            raise RuntimeError("draft transaction has no sealed forward/completion")
        return forward, completion

    def replay_ordinary(self, completion, *, budget=False):
        for graph in self.ordinary_phases(completion, budget=budget):
            graph.replay()

    def capture_review_graph(self, stream):
        """Seal one math/state transaction for ordinary and temporal owners.

        Parent graphs retain their own ingress, timing events and publication.
        The shared child contains no host copies or external timing events.
        Capture happens only in the constructor, after its ordinary warmups.
        """
        if not self._review_capture_open:
            raise RuntimeError("review capture is closed after construction")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("review graph must be sealed before parent capture")
        key = self._review_key()
        if key in self._review_graphs:
            raise RuntimeError("review transaction was already captured")
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream, pool=self._review_graph_pool):
            self.finish_review()
        graph.instantiate()
        stream.synchronize()
        barrier()
        self._review_graphs[key] = graph

    def captured_review_graph(self):
        graph = self._review_graphs.get(self._review_key())
        if graph is None:
            raise RuntimeError("review transaction has no sealed graph for this policy")
        return graph

    def append_review_graph(self):
        if not torch.cuda.is_current_stream_capturing():
            raise RuntimeError("review composition belongs to startup parent capture")
        from cubic_serve.executor.resident_window import append_captured_graph
        append_captured_graph(self.captured_review_graph())

    def stage_context(self):
        """Fill lease-private device context from immutable host inputs."""
        target = self.target
        self.ctx.copy_(self.host,non_blocking=True)
        self.commit_ctx.copy_(self.commit_host,non_blocking=True)
        target._metadata_device.copy_(self.metadata_host,non_blocking=True)
        self.index_device.copy_(self.index_host,non_blocking=True)
        if self._capture_budget:
            self.budget.stage()

    def bind_target_inputs(self):
        """Bind the current device metadata, including explicit logical activity."""
        target = self.target
        tb = target.body.buffers
        target.body.arena.reset_page_rows(self.index_device)
        target._bind_staged_metadata(target._metadata_device,self.M,self.target_T)
        activity=torch.arange(self.M,device=self.ctx.device)<self.ctx[0]*self.stride
        tb.row_activity.copy_(activity)
        if getattr(target.body,'feature_rows',None) is not None:target.body.feature_rows.zero_()

    def stage_sampling(self):
        """Fill provider sampling inputs once for a newly bound transaction."""
        head = self.head
        for host,dst in zip(self.param_hosts,
              (head._sampling.temperatures,head._sampling.top_ks,head._sampling.top_ps,
               head._sampling.seeds,head._sampling.ordinals),strict=True):
            dst.copy_(host,non_blocking=True)

    def _pending_strides(self, depth):
        return dict(SCORE_OWNER=self.pool_scores[depth].stride(1),
                    SCORE_TP=self.pool_scores[depth].stride(0),
                    ID_OWNER=self.pool_ids[depth].stride(1),
                    ID_TP=self.pool_ids[depth].stride(0),
                    TOKEN_OWNER=self.pool_tokens[depth].stride(0))

    def prepare_pending(self):
        """Load this owner's carried proposals without a host dependency."""
        head = self.head
        w=head._sampling.workspace
        for d in range(self.depth):
            for dst,hist in ((head._sampling.temperatures,head._verification_temperatures[d]),
                 (head._sampling.top_ks,head._verification_top_ks[d]),
                 (head._sampling.top_ps,head._verification_top_ps[d]),
                 (head._sampling.seeds,head._verification_seeds[d])):hist.copy_(dst)
            head._verification_ordinals[d].copy_(head._sampling.ordinals+d)
            pending_copy[(self.C,)](self.ctx,head._verification_q_scores[d],head._verification_q_token_ids[d],head._verification_draft_tokens[d],self.pool_scores[d],self.pool_ids[d],self.pool_tokens[d],self.CS,self.C,self.pool.owners,w.gathered_scores.shape[0],w.gathered_scores.shape[2],32,True,**self._pending_strides(d))

    def bind_provider_alignment_metadata(self):
        target,align=self.target,self.align
        ab=align.body.buffers
        span=getattr(align.body,'embedding_input_span',align.body.embedding_span)
        align_meta[(self.M,)](self.ctx,self.decisions,target._metadata_device,ab.token_ids,align.body.embedding_mask,ab.position_ids,ab.page_source,ab.page_destination,ab.token_offsets,ab.sequence_lengths,ab.segment_lengths,ab.row_activity,tuple(ab.state_slot_maps),ab.page_table,self.M,self.T,self.CS,target._metadata_columns,target._position_axes_slice.start,ab.state_slot_maps[0].shape[1],span.start,span.stop,align.body.arena.layout.block_size_tokens,self.scratch_page,self.scratch_state,target.vocab_size,256,self.depth,self.C,**self._checkpoint_arguments(),num_warps=4)

    def bind_provider_proposal_metadata(self,depth):
        target,head=self.target,self.head
        hb=head.body.buffers
        span=getattr(head.body,'embedding_input_span',head.body.embedding_span)
        autoregressive_meta[(self.C,)](self.ctx,self.decisions,target._metadata_device,head._token_ids,hb.token_ids,head.body.embedding_mask,hb.position_ids,hb.page_source,hb.page_destination,hb.token_offsets,hb.sequence_lengths,hb.segment_lengths,hb.row_activity,tuple(hb.state_slot_maps),hb.page_table,self.C,self.T,self.CS,target._metadata_columns,target._position_axes_slice.start,hb.state_slot_maps[0].shape[1],span.start,span.stop,head.body.arena.layout.block_size_tokens,self.scratch_page,self.scratch_state,target.vocab_size,self.depth,depth,256,**self._checkpoint_arguments(),num_warps=4)

    def finish_review(self):
        """Commit accepted state and prepare the next proposals on device.

        Target/provider arithmetic and rounded-state boundaries are shared by
        ordinary rounds and any pre-readiness-qualified temporal composition.
        This phase contains neither host transfers nor external timing events.
        """
        target, align, head = self.target, self.align, self.head
        tb, ab = target.body.buffers, align.body.buffers
        constrained = (self._capture_budget or self._capture_review_only
                       or self._capture_target_constraints)
        target_sampler = self.target_masked if constrained else self.target_greedy
        target_sampler.metadata[0].copy_(self.ctx[0]*self.stride)
        if self._capture_budget:
            self.budget_review.launch()
        if self._capture_candidates:
            review_temperatures = head._sampling.temperatures.repeat_interleave(self.stride)
            if self.M > self.C * self.stride:
                review_temperatures = torch.nn.functional.pad(
                    review_temperatures, (0, self.M - self.C * self.stride))
            target_sampler.prepare_candidates(temperatures=review_temperatures)
        else:
            target_sampler._prepare()
        if target._rank==target._sampler_rank:
            w=target._sampling.workspace
            for d in range(self.depth):
                constraint_rows = (
                    (self.budget.state[:self.C, 0] != -1) if self._capture_budget else
                    self.target_masked.constrained_rows[:self.C*self.stride]
                    .view(self.C, self.stride)[:, 0].contiguous()
                    if self._capture_review_only or self._capture_target_constraints else None
                )
                verify_one_draft_candidates(q_scores=head._verification_q_scores[d],q_token_ids=head._verification_q_token_ids[d],p_scores=w.gathered_scores,p_token_ids=w.gathered_token_ids,target_rows=self.rows[d],draft_tokens=head._verification_draft_tokens[d],temperatures=head._verification_temperatures[d],top_ps=head._verification_top_ps[d],top_ks=head._verification_top_ks[d],seeds=head._verification_seeds[d],ordinals=head._verification_ordinals[d],decisions=head.verification_decisions[d],active_rows=self.C, allow_underfilled=constrained, constrained_rows=constraint_rows)
            merge_prefix[(1,)](self.ctx,head.verification_decisions,self.decisions,self.C,self.depth,triton.next_power_of_2(self.C))
        broadcast(self.decisions,src=target._sampler_rank,group=target._world_group)
        if self._capture_budget:
            self.budget_commit.launch()
        # Target rejection commit; pool rollback snapshot precedes bonus consumption.
        self.copy_fields(target.body.arena.states,target.body.arena.states,2,self.commit_ctx)
        # Commit coherent target feedback first into all admitted target slots.
        target._state_destinations.copy_(target._metadata_device[:,target._state_destination_column])
        hidden=target._target_hidden_slots
        publish_feedback[(self.M,triton.cdiv(hidden.shape[1],256))](
            tb.row_activity,target._state_destinations,getattr(target.body,'draft_feedback',tb.norm),
            hidden,hidden.shape[1],1,hidden.stride(0),256)
        # Rejected owners select root checkpoint after the lossless row publication.
        owner_copy[(self.C,min(triton.cdiv(target._target_hidden_slots.shape[1],2048),256))](self.commit_ctx,self.decisions,target._target_hidden_slots,target._target_hidden_slots,self.commit_CS,target._target_hidden_slots.shape[1],2,2048,min(triton.cdiv(target._target_hidden_slots.shape[1],2048),256),self.depth,SOURCE_STRIDE=target._target_hidden_slots.stride(0),DESTINATION_STRIDE=target._target_hidden_slots.stride(0))
        self.copy_fields(align.state_fields,self.pool.backup,0,self.commit_ctx)
        self.bind_provider_alignment_metadata()
        feedback=getattr(target.body,'draft_feedback',tb.norm)
        H=ab.target_hidden.shape[1]
        align_feedback[(self.M,triton.cdiv(H,256))](self.ctx,self.decisions,feedback,ab.target_hidden,H,self.CS,256,self.depth)
        arena=align.body.arena
        for field in (arena.key,arena.value,arena.written,*arena.page_fields.values()):
            width=math.prod(field.shape[1:]);clear_new_pages[(self.M,triton.cdiv(width,256))](ab.row_activity,ab.token_offsets,ab.page_destination,field,width,256,STRIDE=field.stride(0))
        align.body.set_active_rows(self.M)
        _launch_body(align.body, produce_logits=False)
        self.copy_fields(align.state_fields,self.pool.backup,
                         4 if self._checkpoint_arguments()["CHECKPOINT_CS"] else 3,self.commit_ctx)
        hb=head.body.buffers
        select_head[(self.C,triton.cdiv(max(align.body.hidden.shape[1],ab.norm.shape[1]),256))](self.ctx,self.decisions,align.body.hidden,ab.norm,head.body.hidden,hb.norm,head._verification_ordinals[0],head._sampling.ordinals,self.active,align.body.hidden.shape[1],ab.norm.shape[1],self.CS,256,self.depth)
        self.head_launch()
        if not self._capture_review_only:
            self.finish_proposals()

    def finish_proposals(self):
        """The shared device proposal tail; called eagerly only by startup audits."""
        target, head = self.target, self.head
        hb = head.body.buffers
        arena = self.align.body.arena
        head_sampler = self.head_masked if self._capture_budget else self.head_greedy
        head_sampler.metadata[0].copy_(self.ctx[0])
        self.sample_head(0)
        self.valid.copy_(head_sampler.final_valid)
        self._retain_proposal(0)
        for d in range(1,self.depth):
            self.prepare_proposal(d)
            self.sample_head(d)
            self.valid.logical_and_(head_sampler.final_valid)
            self._retain_proposal(d)
        joint_result[(1,)](self.ctx,self.decisions,self.next_tokens,self.valid,self.result,self.C,self.CS,target.vocab_size,triton.next_power_of_2(self.C),self.depth)

    def prepare_proposal(self, depth):
        """Provider-owned rounded feedback and append metadata for the next q."""
        target, head = self.target, self.head
        hb, arena = head.body.buffers, self.align.body.arena
        hb.target_hidden.copy_(head._draft_binding.feedback)
        self.bind_provider_proposal_metadata(depth)
        for field in (arena.key,arena.value,arena.written,*arena.page_fields.values()):
            width=math.prod(field.shape[1:]);clear_new_pages[(self.C,triton.cdiv(width,256))](hb.row_activity,hb.token_offsets,hb.page_destination,field,width,256,STRIDE=field.stride(0))
        head.body.set_active_rows(self.C)
        _launch_body(head.body)
        head._sampling.ordinals.add_(1)

    def sample_head(self, depth=0):
        constrained = self._capture_budget or self._capture_review_only
        sampler = self.head_masked if constrained else self.head_greedy
        if self._capture_budget:
            self.budget_heads[depth].launch()
        if self._capture_candidates:
            sampler.prepare_candidates()
            sampler.sample_candidates(
                sampler._grammar_root if constrained else self.head._sampling.root_sampler,
                constrained_rows=sampler.constrained_rows if constrained else None,
            )
        else:
            sampler._prepare()
        sampler._broadcast()

    @property
    def active_draft_window(self):
        if self._last_review_only or self._last_target_constraints:
            return None
        if self._last_budget:
            return getattr(self, "_budget_resident_window" if self._last_sampling_is_greedy
                           else "_budget_candidate_window", None)
        return getattr(self, "_resident_window" if self._last_sampling_is_greedy
                       else "_candidate_window", None)

    def device_body(self):
        """The shared round math for a startup-captured resident composition.

        The caller stages a new transaction once and owns every input and
        state until completion. Inter-round ingress consumes device results;
        no host transfer or external timing event belongs to this body.
        """
        self.bind_target_inputs()
        self.prepare_pending()
        _launch_body(self.target.body)
        self.finish_review()

    def body(self):
        """The ordinary captured round retains its original operation order."""
        self.forward_body()
        self.completion_body()

    def forward_body(self):
        """Common target phase; no constraint mask is consumed here."""
        self.round_start.record()
        self.stage_context()
        self.bind_target_inputs()
        self.stage_sampling()
        if self._capture_target_constraints:
            self.constraint_forward_valid.fill_(True)
        self.prepare_pending()
        self.target._replay_started.record()
        _launch_body(self.target.body)
        self.target._replay_finished.record()

    def completion_body(self):
        """Verification consumes the prepared mask before any result is published."""
        if torch.cuda.is_current_stream_capturing():
            self.append_review_graph()
        else:
            # Constructor warmups keep their existing eager body. Runtime run()
            # only replays the completed, qualified parent graph portfolio.
            self.finish_review()
        if self._capture_target_constraints:
            self.result[:, -1].mul_(self.constraint_forward_valid)
        if self._capture_review_only:
            self.host_decisions.copy_(self.decisions, non_blocking=True)
        else:
            self.host_result.copy_(self.result, non_blocking=True)
        self.round_end.record()

    def _retain_proposal(self,depth):
        w=self.head._sampling.workspace
        self.next_tokens[depth].copy_(self.head._token_ids)
        pending_copy[(self.C,)](self.ctx,w.gathered_scores,w.gathered_token_ids,self.head._token_ids,self.pool_scores[depth],self.pool_ids[depth],self.pool_tokens[depth],self.CS,self.C,self.pool.owners,w.gathered_scores.shape[0],w.gathered_scores.shape[2],32,False,**self._pending_strides(depth))

    def capture_constraint_continuation(self):
        """Seal one device-fed forward; the ordinary completion phase is shared."""
        stream = self.target.slot.stream
        stream.wait_stream(torch.cuda.current_stream(self.ctx.device))
        with torch.cuda.stream(stream):
            self.constraint_continuation_forward_body()
        stream.synchronize()
        barrier()
        self.constraint_continuation_forward_graph = torch.cuda.CUDAGraph(keep_graph=self._retain_audit_graph)
        with torch.cuda.graph(self.constraint_continuation_forward_graph, stream=stream, pool=self._graph_pool):
            self.constraint_continuation_forward_body()
        if self._retain_audit_graph:
            self.constraint_continuation_forward_graph.instantiate()
        stream.synchronize()
        barrier()

    def constraint_continuation_forward_body(self):
        """Advance only the committed owner map and its already-granted pages."""
        self.round_start.record()
        window = self._resident_window
        window.advance()
        self.constraint_forward_valid.copy_(
            (window.ready | (torch.arange(self.C, device=self.ctx.device) >= self.ctx[0])).all())
        self.bind_target_inputs()
        self.prepare_pending()
        self.target._replay_started.record()
        _launch_body(self.target.body)
        self.target._replay_finished.record()

    def bind_constraint_continuation(self, binding):
        """Borrow the existing window's limit storage under the same joint lease."""
        if (self._constraint_forward_inflight or self._constraint_continuation_binding is not None
                or not self._last_target_constraints or binding.depth != self.depth
                or not getattr(self, "constraint_continuation_forward_graph", None)):
            raise RuntimeError("constraint continuation has no available sealed forward")
        n = len(binding.owners)
        contexts = self.host_array[1:].reshape(self.C, self.CS)
        metadata = self.metadata_host.numpy()
        target = self.target
        if (not 0 < n <= self.C or int(self.host_array[0]) != n
                or tuple(int(value) for value in contexts[:n, 1]) != binding.owners):
            raise RuntimeError("constraint continuation changed its bound owners")
        for physical, owner in enumerate(binding.owners):
            pending = self.pool.metadata[owner]
            first = physical * self.stride
            source = metadata[first:first + self.stride, target._source_slots_slice]
            layers = source.shape[1]
            if (pending is None or len(pending[6]) != self.depth
                    or not np.all(source == owner * layers + np.arange(layers))
                    or int(metadata[first + self.depth, target._state_destination_column]) != owner
                    or tuple(int(v) for v in metadata[first, target._page_table_slice]
                             [:len(binding.target_tables[physical])]) != binding.target_tables[physical]
                    or tuple(int(v) for v in contexts[physical, 2 + self.depth:]
                             [:len(binding.provider_tables[physical])]) != binding.provider_tables[physical]
                    or binding.target_limits[physical] > len(binding.target_tables[physical]) * target._block_size
                    or binding.provider_limits[physical] > len(binding.provider_tables[physical]) * target._block_size
                    or binding.target_positions[physical] + self.depth >= binding.target_limits[physical]
                    or pending[1] + 2 * self.depth >= binding.provider_limits[physical]):
                raise RuntimeError("constraint continuation changed its committed extent")
        window = self._resident_window
        for name, values in (("target_limits", binding.target_limits),
                             ("provider_limits", binding.provider_limits)):
            host = window.staging[name]
            host.numpy().fill(0)
            host.numpy()[:n] = values
            getattr(window, name).copy_(host, non_blocking=True)
        # The preceding full ingress reset all newly granted pages. Replaying
        # its reset list would erase durable data on every following round.
        self.index_host.numpy()[:] = target._scratch_pages
        self.index_device.copy_(self.index_host, non_blocking=True)
        self._constraint_continuation_binding = binding

    def start_constraint_forward(self):
        if self._constraint_forward_inflight or self._constraint_continuation_binding is None:
            raise RuntimeError("constraint forward requires one exclusive bound continuation")
        target = self.target
        if target._batch_bound or target._batch_phase_failed:
            raise RuntimeError("constraint forward has no available transaction phase")
        target.slot.stream.wait_stream(torch.cuda.current_stream(self.ctx.device))
        self._constraint_forward_inflight = True
        self._constraint_forward_queued_ns = time.monotonic_ns()
        with torch.cuda.stream(target.slot.stream):
            self.constraint_continuation_forward_graph.replay()

    def close_constraint_continuation(self):
        """An incomplete forward poisons its bind and drains before lease exit."""
        try:
            if self._constraint_forward_inflight:
                self.target._batch_phase_failed = True
                self.target.slot.stream.synchronize()
        finally:
            self._constraint_forward_inflight = False
            self._constraint_continuation_binding = None

    def _replay_target_constraints(self, graph, prepare_constraints, current):
        """Keep both sealed phases and the host preparation in the caller's lease."""
        stream = self.target.slot.stream
        if not self._constraint_forward_inflight:
            self._constraint_forward_queued_ns = time.monotonic_ns()
            with torch.cuda.stream(stream):
                self.target_constraint_forward_graph.replay()
        try:
            if prepare_constraints is not None:
                prepare_constraints()
        except BaseException:
            # Forward may already have written speculative state. A failed
            # preparation cannot publish a result or leave a reusable bind.
            self.target._batch_phase_failed = True
            self.completed.record(stream)
            self.completed.synchronize()
            self._constraint_forward_inflight = False
            self._last_transfer = True
            current.wait_stream(stream)
            raise
        # Arena.bind orders its H2D copy onto current. This dependency must be
        # inserted after preparation, not before the independent target phase.
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            graph.replay()
            self.completed.record()

    def run(self,rows,segments,*,budget=False,review_only=False,target_constraints=False,
            prepare_constraints=None):
        if prepare_constraints is not None and (
                not target_constraints or not callable(prepare_constraints)):
            raise RuntimeError("constraint preparation requires a target-constrained round")
        if target_constraints and getattr(self, "target_constraint_forward_graph", None) is None:
            raise RuntimeError("target constraints have no sealed forward phase")
        forward_inflight = self._constraint_forward_inflight
        if forward_inflight and (not target_constraints or budget or review_only):
            raise RuntimeError("in-flight target differs from its constrained transaction")
        if self._last_transfer is not None:self.completed.synchronize()
        n=len(segments)
        target=self.target
        if target._batch_bound or target._batch_phase_failed:raise RuntimeError('whole-round ingress has no available bind phase')
        if forward_inflight:
            binding = self._constraint_continuation_binding
            if (n != len(binding.owners)
                    or tuple(rows[s.start].state_source_slot for s in segments) != binding.owners
                    or (all(row.sampling.temperature == 0. for row in rows)
                        != self._last_sampling_is_greedy)):
                raise RuntimeError("in-flight target changed its owner map")
        else:
            arr=self.host_array[1:].reshape(self.C,self.CS);cc=self.commit_host_array[1:].reshape(self.C,self.commit_CS)
            self.host_array[0]=self.commit_host_array[0]=n
            for r,segment in enumerate(segments):
                root,last=rows[segment.start],rows[segment.stop-1];owner=last.state_destination_slot
                arr[r,:2]=(root.mtp_position,owner)
                arr[r,2:2+self.depth]=tuple(rows[segment.start+d+1].token_id for d in range(self.depth))
                arr[r,2+self.depth:].fill(self.padding_page(self.stride*r))
                arr[r,2+self.depth:2+self.depth+len(root.mtp_page_table)]=root.mtp_page_table
                cc[r,:3]=(0,owner,self.scratch_state+self.stride*r)
                cc[r,3:]=tuple(rows[segment.start+d].state_destination_slot for d in range(self.depth))
                for array,value in zip(self.param_arrays,(root.sampling.temperature,root.sampling.top_k,root.sampling.top_p,root.sampling.seed,root.sample_ordinal+1),strict=True):array[r]=value
            target._bound_active_rows=self.M
            transferred,resets,new_pages=target._stage_batch_rows(rows,host_slot=0)
            if transferred!=self.M or resets:raise RuntimeError('whole-round ingress exceeds its sealed ownership capacity')
            # Input initialization owns these fresh pages under the same lease.
            # Reuse existing staging for overflow; the sealed graph still clears
            # its original M-entry list. No new graph or device allocation.
            if new_pages > self.M:
                page_indices = slice(target._index_page_offset + self.M,
                                     target._index_page_offset + new_pages)
                if page_indices.stop > target._index_capacity:
                    raise RuntimeError("fresh pages exceed existing initialization staging")
                target._index_device[page_indices].copy_(
                    target._index_hosts[0][page_indices], non_blocking=True)
                target.body.arena.reset_page_rows(target._index_device[page_indices])
            np.copyto(self.metadata_host.numpy(),target._metadata_host_arrays[0])
            self.index_host.numpy()[:]=target._scratch_pages
            graph_new_pages = min(new_pages, self.M)
            self.index_host.numpy()[:graph_new_pages]=target._index_host_arrays[0][target._index_page_offset:target._index_page_offset+graph_new_pages]
            target._bound_active_rows=len(rows);target._features_bound=False
            target.body.set_active_rows(len(rows))
            target._batch_bound=True;target.complete_bound(replay=False)
        current=torch.cuda.current_stream(target.body.hidden.device);target.slot.stream.wait_stream(current)
        self._last_sampling_is_greedy = all(row.sampling.temperature == 0. for row in rows)
        self._last_budget = budget
        self._last_review_only = review_only
        self._last_target_constraints = target_constraints
        if sum((budget, review_only, target_constraints)) > 1:
            raise RuntimeError("draft transaction has conflicting sampling control")
        if target_constraints:
            graph = self.target_constraint_graphs[not self._last_sampling_is_greedy]
        elif review_only:
            graph = self.grammar_review_graph
        elif budget:
            graph = self.budget_graph if self._last_sampling_is_greedy else self.budget_candidate_graph
        else:
            graph = self.graph if self._last_sampling_is_greedy else self.candidate_graph
        if graph is None:
            raise RuntimeError("reasoning budget has no sealed draft composition")
        target._replay_started, target._replay_finished = self.review_started, self.review_finished
        if target_constraints:
            self._replay_target_constraints(graph, prepare_constraints, current)
        else:
            with torch.cuda.stream(target.slot.stream):
                self.replay_ordinary(graph, budget=budget)
                self.completed.record()
        self.completed.synchronize();self._last_transfer=True
        self._constraint_forward_inflight = False
        current.wait_stream(target.slot.stream);target._replay_timing_pending=True
        self.head_greedy.invalidate_metadata();self.target_greedy.invalidate_metadata()
        if budget or review_only or target_constraints:
            self.head_masked.invalidate_metadata(); self.target_masked.invalidate_metadata()
        if review_only:
            result = self.host_decisions[:n].tolist()
            if any(not 0 <= row[0] <= self.depth or not 0 <= row[1] < target.vocab_size
                   for row in result):
                raise RuntimeError("masked draft review guard rejected")
            return result
        result=self.host_result[:n].tolist()
        if not all(row[-1] for row in result): raise RuntimeError('whole-round guard rejected')
        return result


class CapturedDraftWindow:
    """Startup-only temporal composition of the common carried draft round.

    This primitive grants no readiness. Its caller qualifies the composition,
    supplies already-owned cache limits and retains the topology lease through
    durable consumption on every rank. It never allocates request pages.
    """

    def __init__(self, round_, *, journal=None, observer=None):
        from pathlib import Path
        from cuda.bindings import driver as cuda_driver, runtime as rt
        from cubic_serve.executor.resident_window import check, conditional_handle, add_conditional_body
        from cubic_serve.executor.conditional_graph import add_conditional_child
        from cubic_serve.ops.cuda.native import CudaScalar
        from cubic_serve.ops.cuda.native_compiler import compile_native
        from cubic_serve.runtime.continuous_batch import RESIDENT_STEP_CAPACITY

        self.round = round_
        self.requires_budget = getattr(round_, "_capture_budget", False)
        self.requires_greedy = not round_._capture_candidates
        if getattr(round_, "_last_transfer", None) is not None:
            raise RuntimeError("draft windows must be captured before request execution")
        self.capacity = round_.C
        self.depth = round_.depth
        self.steps_capacity = RESIDENT_STEP_CAPACITY
        self.row_stride = 2 * self.depth + 8
        self.record_stride = self.capacity * self.row_stride + 6
        self.qualified = False
        self.inflight = False
        self.bound = False
        self._shared_journal = journal
        device = round_.target.body.hidden.device
        self.meta = torch.zeros(5, dtype=torch.int32, device=device)
        self.meta_host = torch.zeros(5, dtype=torch.int32, pin_memory=True)
        self.policy = torch.zeros(2, dtype=torch.uint64, device=device)
        self.policy_host = torch.zeros(2, dtype=torch.uint64, pin_memory=True)
        self.clocks = torch.empty((RESIDENT_STEP_CAPACITY, 4), dtype=torch.int64, device=device)
        self.ready = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        self.target_limits = torch.zeros(self.capacity, dtype=torch.int64, device=device)
        self.provider_limits = torch.zeros_like(self.target_limits)
        # Output allowance and the optional policy token/round budgets share
        # the existing stream-ordered control staging, never a separate pool.
        self.remaining = torch.zeros((self.capacity, 3), dtype=torch.int64, device=device)
        self.eos_offsets = torch.zeros(self.capacity + 1, dtype=torch.int32, device=device)
        self.eos_values = torch.zeros(round_.target.vocab_size, dtype=torch.int64, device=device)
        self.staging = {
            name: torch.zeros_like(getattr(self, name), device="cpu", pin_memory=True)
            for name in ("target_limits", "provider_limits", "remaining", "eos_offsets", "eos_values")
        }
        elements = 64 + RESIDENT_STEP_CAPACITY * self.record_stride
        if journal is not None and torch.device(journal.device) != torch.device(device):
            raise ValueError("draft journal belongs to a different device lease")
        self.host = (torch.zeros(elements, dtype=torch.int64, pin_memory=True)
                     if journal is None else journal.tensor)
        if self.host is None or self.host.numel() != elements or not self.host.is_pinned():
            raise ValueError("draft journal differs from its captured resource")
        self.host_array = self.host.numpy()
        self.host_flags = self.host_array.view("int32")
        mapped = int(check(rt.cudaHostGetDevicePointer(self.host.data_ptr(), 0))[0])
        self.host_address = torch.tensor([mapped], dtype=torch.uint64, device=device)
        self.graph = check(rt.cudaGraphCreate(0))[0]
        self.exec = None
        self.children = ()
        try:
            handle = conditional_handle(self.graph, 1)
            source = (Path(__file__).parents[1] / "ops/cuda/sampling/resident_draft_window.cu").read_bytes()
            module, entry = compile_native((("entry.cu", source),), device, "resident_draft_window")
            args = (self.meta, round_.ctx, round_.result, round_.head._verification_ordinals[0],
                    round_.target._metadata_device, self.ready, self.remaining,
                    self.eos_offsets, self.eos_values, self.clocks, self.policy, self.host_address,
                    CudaScalar("i32", self.capacity), CudaScalar("i32", self.depth),
                    CudaScalar("i32", round_.CS), CudaScalar("i32", round_.target._metadata_columns),
                    CudaScalar("i32", round_.target._position_axes_slice.start),
                    CudaScalar("i32", round_.target.vocab_size),
                    CudaScalar("i32", int(round_.target._rank == round_.target._sampler_rank)),
                    CudaScalar("i64", int(handle)))
            self.calls = tuple(module.bind(entry, (*args, CudaScalar("i32", phase)),
                                          operator_id="runtime.resident_draft_control",
                                          grid=(1, 1, 1), block=(128, 1, 1)) for phase in range(7))
            self.control = self.meta[4:5]
            all_reduce(self.control, op=torch.distributed.ReduceOp.MAX,
                                         group=round_.target._world_group)
            stream = round_.target.slot.stream
            stream.wait_stream(torch.cuda.current_stream(device))
            # Compile the integer ingress before capture, using the round's private
            # startup fixture (zero logical owners), never a live request.
            with torch.cuda.stream(stream):
                self.advance()
            stream.synchronize()
            ingress = torch.cuda.CUDAGraph(keep_graph=True)
            body = torch.cuda.CUDAGraph(keep_graph=True)
            self.children = (ingress, body)
            with torch.cuda.graph(ingress, stream=stream,
                                  pool=round_._graph_pool if observer is None else None):
                self.meta.copy_(self.meta_host, non_blocking=True)
                self.policy.copy_(self.policy_host, non_blocking=True)
                for name, host in self.staging.items():
                    getattr(self, name).copy_(host, non_blocking=True)
                round_.stage_context()
                round_.stage_sampling()
                self.ready.fill_(True)
                self.calls[6].launch()
            with torch.cuda.graph(body, stream=stream,
                                  pool=round_._graph_pool if observer is None else None):
                self.calls[0].launch()
                round_.bind_target_inputs()
                # A future exclusively granted page is reset exactly at its
                # first append offset. Rejected suffixes can revisit offset 0;
                # no committed prefix lives on that page yet in that case.
                buffers = round_.target.body.buffers
                arena = round_.target.body.arena
                for field in (arena.key, arena.value, arena.written, *arena.page_fields.values()):
                    width = math.prod(field.shape[1:])
                    clear_new_pages[(round_.M, triton.cdiv(width, 256))](
                        buffers.row_activity, buffers.token_offsets, buffers.page_destination,
                        field, width, 256, STRIDE=field.stride(0),
                    )
                round_.prepare_pending()
                self.calls[1].launch()
                _launch_body(round_.target.body)
                self.calls[2].launch()
                round_.append_review_graph()
                if observer is not None:
                    # Capture-time audit instrumentation may retain only immutable,
                    # stream-ordered device snapshots; reference work is post-forward.
                    observer(self)
                self.calls[3].launch()
                self.advance()
                self.calls[4].launch()
                all_reduce(self.control, op=torch.distributed.ReduceOp.MAX,
                                             group=round_.target._world_group)
                self.calls[5].launch()
            stream.synchronize()
            # Only the parent is replayed; retain raw children and tensor backing.
            self.children = (ingress, body)
            begin = check(rt.cudaGraphAddChildGraphNode(self.graph, [], 0, ingress.raw_cuda_graph()))[0]
            _, loop_body = add_conditional_body(self.graph, [begin], handle, while_loop=True)
            add_conditional_child(loop_body, [], body.raw_cuda_graph())
            self.exec = check(cuda_driver.cuGraphInstantiate(self.graph, 0))[0]
            self.completed = torch.cuda.Event()
            round_.head_greedy.invalidate_metadata()
            round_.target_greedy.invalidate_metadata()
        except BaseException:
            self.close()
            raise

    def advance(self):
        round_ = self.round
        span = getattr(round_.target.body, "embedding_input_span", round_.target.body.embedding_span)
        advance_round_context[(self.capacity,)](
            round_.ctx, round_.result, round_.target._metadata_device,
            round_.head._sampling.ordinals, round_.head._verification_ordinals[0],
            self.target_limits, self.provider_limits, self.ready, self.capacity, round_.M,
            round_.CS, round_.target._metadata_columns, round_.target._position_axes_slice.start,
            round_.target._page_table_slice.start, round_.target._block_size,
            span.start if span is not None else 0, span.stop if span is not None else 0,
            round_.target.vocab_size, self.depth,
            triton.next_power_of_2(self.depth + 1), round_.target.body.stage == 0,
        )

    def bind(self, *, owners, target_positions, target_limits, provider_limits, remaining, eos,
             max_steps, max_duration_seconds, target_tables=None, provider_tables=None,
             decision_tokens=(), decision_rounds=()):
        import time
        if self.inflight:
            raise RuntimeError("draft window input is still leased by an active replay")
        self.bound = False
        if (type(owners) is not tuple or not owners or len(owners) > self.capacity
                or len(set(owners)) != len(owners)
                or any(type(owner) is not int or not 0 <= owner < self.round.pool.owners
                       for owner in owners)
                or type(max_steps) is not int or not 2 <= max_steps <= self.steps_capacity
                or type(max_duration_seconds) not in (int, float)
                or not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0):
            raise ValueError("draft window differs from its owned resource")
        n = len(owners)
        if (any(type(values) is not tuple for values in (decision_tokens, decision_rounds))
                or bool(decision_tokens) != bool(decision_rounds)
                or any(type(values) is not tuple or len(values) != n
                       or any(type(v) is not int or not 0 <= v <= 0x7fffffffffffffff
                              for v in values)
                       for values in (decision_tokens, decision_rounds) if values)
                or (decision_tokens and not any((*decision_tokens, *decision_rounds)))):
            raise ValueError("draft window decision budget differs from its owners")
        for values in (target_limits, provider_limits, remaining):
            if (type(values) is not tuple or len(values) != n
                    or any(type(value) is not int or value <= 0 for value in values)):
                raise ValueError("draft window requires explicit owned limits")
        if (type(target_positions) is not tuple or len(target_positions) != n
                or any(type(value) is not int or value < 0 for value in target_positions)):
            raise ValueError("draft window requires explicit target positions")
        if int(self.round.host_array[0]) != n:
            raise ValueError("draft window differs from its bound owner count")
        contexts = self.round.host_array[1:].reshape(self.capacity, self.round.CS)
        if tuple(int(value) for value in contexts[:n, 1]) != owners:
            raise ValueError("draft window changed its bound owner map")
        if any(value < max_steps * self.round.stride for value in remaining):
            raise ValueError("draft window output budget cannot cover its granted rounds")
        metadata = self.round.metadata_host.numpy()
        tables_provided = target_tables is not None or provider_tables is not None
        if tables_provided:
            for tables, arena, width, limits in (
                (target_tables, self.round.target.body.arena, self.round.target_T, target_limits),
                (provider_tables, self.round.align.body.arena, self.round.T, provider_limits),
            ):
                if (type(tables) is not tuple or len(tables) != n
                        or any(type(table) is not tuple or not table or len(table) > width
                               or len(set(table)) != len(table)
                               or any(type(page) is not int or not 0 <= page < arena.spec.page_capacity
                                      for page in table)
                               or limit > len(table) * self.round.target._block_size
                               for table, limit in zip(tables, limits))):
                    raise ValueError("draft window page grants differ from its cache resource")
        source_slots = self.round.target._source_slots_slice
        local_layers = source_slots.stop - source_slots.start
        for row, (owner, position) in enumerate(zip(owners, target_positions, strict=True)):
            pending = self.round.pool.metadata[owner]
            if pending is None or len(pending) != 7:
                raise ValueError("draft window has no durable pending input")
            _request, provider_position, root, sampling, ordinal, axes, proposals = pending
            if (len(proposals) != self.depth
                    or (self.requires_greedy and sampling.temperature != 0.)
                    or position + self.depth >= target_limits[row]
                    or provider_position + 2 * self.depth >= provider_limits[row]):
                raise ValueError("draft window pending input exceeds its owned limits")
            first = row * self.round.stride
            old = metadata[first:first + self.round.stride]
            if tables_provided:
                target_prefix = (position + self.round.target._block_size - 1) // self.round.target._block_size
                provider_prefix = (provider_position + self.depth + self.round.target._block_size - 1) // self.round.target._block_size
                target_slice = self.round.target._page_table_slice
                if (tuple(int(v) for v in old[0, target_slice][:target_prefix])
                        != target_tables[row][:target_prefix]
                        or tuple(int(v) for v in contexts[row, 2 + self.depth:][:provider_prefix])
                        != provider_tables[row][:provider_prefix]):
                    raise ValueError("draft window remapped its committed cache prefix")
                old[:, target_slice] = self.round.padding_page(first)
                old[:, target_slice.start:target_slice.start + len(target_tables[row])] = target_tables[row]
                contexts[row, 2 + self.depth:] = self.round.padding_page(first)
                contexts[row, 2 + self.depth:2 + self.depth + len(provider_tables[row])] = provider_tables[row]
            axes_slice = self.round.target._position_axes_slice
            step = old[1, axes_slice].astype("int64") - old[0, axes_slice].astype("int64")
            committed_source = owner * local_layers + np.arange(local_layers)
            if (not np.all(old[:, source_slots] == old[0, source_slots])
                    or int(old[-1, self.round.target._state_destination_column]) != owner
                    or not np.array_equal(old[:, 0], int(old[0, 0]) + np.arange(self.round.stride))
                    or not np.array_equal(old[:, axes_slice].astype("int64"),
                                          old[0, axes_slice] + np.arange(self.round.stride)[:, None] * step)):
                raise ValueError("draft window requires private contiguous append ownership")
            if (target_limits[row] > self.round.target_T * self.round.target._block_size
                    or provider_limits[row] > self.round.T * self.round.target._block_size):
                raise ValueError("draft window owned limits exceed the bound tables")
            # The preceding COW transaction has committed its destination.
            # Old source slots/pages can already belong to another lease; read
            # only the declared committed owner and its retained replay table.
            old[:, source_slots] = committed_source
            if axes is None:
                if not np.all(step == 1):
                    raise ValueError("implicit draft positions require the default unit step")
                axes = (position,) * (axes_slice.stop - axes_slice.start)
            contexts[row, 0] = provider_position
            contexts[row, 2:2 + self.depth] = proposals
            for j, token in enumerate((root, *proposals)):
                new_position = position + j
                page_index = new_position // self.round.target._block_size
                table_column = self.round.target._page_table_slice.start
                if not 0 <= page_index < self.round.target_T:
                    raise ValueError("draft window target position exceeds its bound page table")
                page = int(old[j, table_column + page_index])
                span = getattr(self.round.target.body, "embedding_input_span",
                               self.round.target.body.embedding_span)
                old[j, :5] = (new_position, page, page,
                              new_position % self.round.target._block_size, new_position + 1)
                if self.round.target.body.stage == 0:
                    if span is None:
                        raise ValueError("draft input stage has no declared embedding span")
                    local = span.start <= token < span.stop
                    old[j, 5:7] = (token - span.start if local else 0, int(local))
                next_axes = np.asarray(axes, dtype="int64") + j * step
                if np.any(next_axes < 0) or np.any(next_axes > 2147483647):
                    raise ValueError("draft window position axes exceed their metadata contract")
                old[j, axes_slice] = next_axes
            for host, value in zip(self.round.param_hosts,
                                   (sampling.temperature, sampling.top_k, sampling.top_p,
                                    sampling.seed, ordinal + 1), strict=True):
                host.numpy()[row] = value
        self.round.index_host.numpy()[:] = self.round.target._scratch_pages
        if (type(eos) is not tuple or len(eos) != n
                or any(type(ids) is not tuple or any(type(token) is not int or
                       not 0 <= token < self.round.target.vocab_size for token in ids) for ids in eos)):
            raise ValueError("draft window EOS differs from its owner map")
        offsets, values = [0], []
        for ids in eos:
            values.extend(ids)
            offsets.append(len(values))
        if len(values) > self.eos_values.numel():
            raise ValueError("draft window EOS exceeds its declared resource")
        self.owners = owners
        self.host_flags[0] = 0
        # Preserve a controller revocation that arrived after dispatch.
        if not (self._shared_journal is not None and execution_ready(self)):
            self.host_flags[2] = 0
        self.meta_host.numpy()[:] = (n, max_steps, 0, 0, 0)
        self.policy_host.numpy()[:] = (max(1, int(max_duration_seconds * 1e9)), 0)
        for name, data in (("target_limits", target_limits), ("provider_limits", provider_limits)):
            self.staging[name].numpy().fill(0)
            self.staging[name].numpy()[:n] = data
        budget = self.staging["remaining"].numpy()
        budget.fill(-1)
        budget[:n, 0] = remaining
        if decision_tokens:
            budget[:n, 1] = decision_tokens
            budget[:n, 2] = decision_rounds
        self.staging["eos_offsets"].numpy().fill(len(values))
        self.staging["eos_offsets"].numpy()[:n + 1] = offsets
        self.staging["eos_values"].numpy()[:len(values)] = values
        self.started = time.monotonic()
        self.bound = True

    def replay(self):
        from cuda.bindings import runtime as rt
        from cubic_serve.executor.resident_window import check
        if self.inflight:
            raise RuntimeError("draft window already has an active replay")
        if not self.bound:
            raise RuntimeError("draft window replay requires a complete owned input bind")
        self.bound = False
        stream = self.round.target.slot.stream
        stream.wait_stream(torch.cuda.current_stream(self.round.target.body.hidden.device))
        self.inflight = True
        check(rt.cudaGraphLaunch(self.exec, stream.cuda_stream))
        self.completed.record(stream)

    def records(self):
        """Immutable CPU views, valid only after each stream-ordered publication."""
        count = int(self.host_flags[0])
        if not 0 <= count <= self.steps_capacity:
            raise ValueError("draft publication count exceeds its journal")
        return self.host_array[64:64 + count * self.record_stride].reshape(count, self.record_stride)

    def receipt(self, index, *, batch_id, request_ids):
        """Decode one immutable publication without releasing its topology lease."""
        from cubic_serve.runtime.continuous_batch import BatchProtocolError, BatchResult, SegmentResult
        records = self.records()
        expected_ids = tuple(self.round.pool.metadata[owner][0] for owner in self.owners)
        if (type(index) is not int or not 0 <= index < len(records)
                or type(batch_id) is not int or batch_id < 0
                or type(request_ids) is not tuple or request_ids != expected_ids):
            raise BatchProtocolError("draft publication differs from its bound owners")
        record = records[index]
        if record[-1] or record[-2] not in (0, 1):
            raise BatchProtocolError("draft publication failed its execution guard")
        rows = record[:self.capacity * self.row_stride].reshape(self.capacity, self.row_stride)
        segments = []
        for request, row in zip(request_ids, rows):
            count = int(row[0])
            if (not 1 <= count <= self.depth + 1 or row[-1] != 1
                    or any(not 0 <= int(token) < self.round.target.vocab_size
                           for token in row[1:count + 1])):
                raise BatchProtocolError("draft publication contains an invalid token prefix")
            segments.append(SegmentResult(request, count, tuple(int(value) for value in row[1:count + 1])))
        ticks = tuple(int(value) for value in record[-6:-2])
        if not 0 <= ticks[0] < ticks[1] < ticks[2] < ticks[3]:
            raise BatchProtocolError("draft publication has unordered device clocks")
        complete = bool(record[-2])
        intervals = tuple(float(value) / 1e6 for value in np.diff(records[:index + 1, -3])) if complete else ()
        if any(not math.isfinite(value) or value <= 0 for value in intervals):
            raise BatchProtocolError("draft publication periods are invalid")
        return BatchResult(
            self.round.target._rank, batch_id, request_ids, (), tuple(segments),
            target_ms=(ticks[2] - ticks[1]) / 1e6,
            step_ms=(ticks[3] - ticks[0]) / 1e6,
            continuation_index=index, continuation_complete=complete,
            gpu_step_intervals_ms=intervals,
        )

    def consume(self):
        """Finish GPU work before installing durable pending-owner metadata."""
        if not self.inflight:
            raise RuntimeError("draft window has no replay to consume")
        self.completed.synchronize()
        records = self.records().copy()
        if (len(records) < 1 or records[-1, -2] != 1
                or any(record[-1] for record in records)):
            raise ValueError("draft window did not publish a valid final transaction")
        rows = records[-1, :self.capacity * self.row_stride].reshape(self.capacity, self.row_stride)
        updates = []
        for row, owner in zip(rows, self.owners):
            count = int(row[0])
            if (not 1 <= count <= self.depth + 1 or row[-1] != 1
                    or any(not 0 <= int(token) < self.round.target.vocab_size
                           for token in row[1:count + 1])):
                raise ValueError("draft window published an invalid owner result")
            prior = self.round.pool.metadata[owner]
            updates.append((owner, (
                prior[0], int(row[2 * self.depth + 2]), int(row[count]), prior[3],
                int(row[2 * self.depth + 3]),
                (None if prior[5] is None else tuple(
                    int(value) for value in row[2 * self.depth + 4:2 * self.depth + 7])),
                tuple(int(value) for value in row[self.depth + 2:2 * self.depth + 2]),
            )))
        for owner, pending in updates:
            self.round.pool.metadata[owner] = pending
        torch.cuda.current_stream(self.round.target.body.hidden.device).wait_stream(
            self.round.target.slot.stream)
        self.round.head_greedy.invalidate_metadata()
        self.round.target_greedy.invalidate_metadata()
        if getattr(self, "requires_budget", False):
            self.round.head_masked.invalidate_metadata()
            self.round.target_masked.invalidate_metadata()
        self.round._last_transfer = True
        self.inflight = False
        return records

    def close(self):
        from cuda.bindings import runtime as rt
        from cubic_serve.executor.resident_window import check
        if self.inflight:
            self.completed.synchronize()
            self.inflight = False
        if self.exec is not None:
            check(rt.cudaGraphExecDestroy(self.exec))
            self.exec = None
        if self.graph is not None:
            check(rt.cudaGraphDestroy(self.graph))
            self.graph = None
        for graph in self.children:
            graph.reset()
        self.children = ()


@triton.jit
def prompt_provider_metadata(activity, lengths, positions, tokens, step,
                             owners, private_states, private_pages,
                             ids, mask, out_lengths, out_positions, offsets,
                             sources, destinations, segments, out_segments,
                             out_activity, maps, tables, updates, reset_activity,
                             M: tl.constexpr, T: tl.constexpr, OWNER_STRIDE: tl.constexpr,
                             LAYERS: tl.constexpr, PAGE: tl.constexpr,
                             START: tl.constexpr, STOP: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    live = tl.load(activity + row) != 0
    index = tl.load(step)
    token = tl.load(tokens + index * M + row, live, 0)
    owned = live & (token >= START) & (token < STOP)
    tl.store(ids + row, tl.where(owned, token - START, 0))
    tl.store(mask + row, owned)
    length = tl.load(lengths + row, live, 1)
    position = length - 1
    tl.store(out_lengths + row, length)
    tl.store(offsets + row, tl.where(live, position % PAGE, 0))
    tl.store(out_activity + row, live)
    tl.store(out_segments + row, tl.load(segments + row, live, 0))
    for axis in tl.static_range(3):
        tl.store(out_positions + axis * M + row,
                 tl.load(positions + axis * M + row, live, 0))
    padding_page = tl.load(private_pages + row)
    page = tl.load(tables + row * T + position // PAGE, live, padding_page)
    tl.store(sources + row, page)
    tl.store(destinations + row, page)
    reset = live & (position % PAGE == 0)
    tl.store(reset_activity + row, reset)
    tl.store(updates + row * 3, tl.where(reset, row, -1))
    tl.store(updates + row * 3 + 1, tl.where(reset, position // PAGE, -1))
    tl.store(updates + row * 3 + 2, tl.where(reset, page, -1))
    owner = tl.load(owners + row * OWNER_STRIDE, live, 0).to(tl.int64)
    padding_state = tl.load(private_states + row)
    for field in tl.static_range(len(maps)):
        layer = tl.arange(0, triton.next_power_of_2(LAYERS[field]))
        source_index = tl.where(live, owner, padding_state) * LAYERS[field] + layer
        destination_index = source_index
        tl.store(maps[field] + layer * M + row, source_index, layer < LAYERS[field])
        tl.store(maps[field] + (LAYERS[field] + layer) * M + row,
                 destination_index, layer < LAYERS[field])


@triton.jit
def prompt_provider_feedback(activity, owner_starts, segments, owners,
                             source, durable, output, M: tl.constexpr,
                             OWNER_STRIDE: tl.constexpr, H: tl.constexpr, B: tl.constexpr,
                             PUBLISH: tl.constexpr, DURABLE_STRIDE: tl.constexpr=0):
    row = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    live = tl.load(activity + row) != 0
    first = tl.load(owner_starts + row, live, 0)
    width = tl.load(segments + first, live, 1)
    if PUBLISH:
        last = live & (row + 1 == first + width)
        owner = tl.load(owners + row * OWNER_STRIDE, last, 0).to(tl.int64)
        value = tl.load(source + row * H + h, last & (h < H), 0)
        tl.store(durable + owner * (DURABLE_STRIDE if DURABLE_STRIDE else H) + h, value, last & (h < H))
    else:
        previous = live & (row > first)
        owner = tl.load(owners + row * OWNER_STRIDE, live, 0).to(tl.int64)
        value = tl.load(source + (row - 1) * H + h, previous & (h < H), 0)
        carried = tl.load(durable + owner * (DURABLE_STRIDE if DURABLE_STRIDE else H) + h, live & ~previous & (h < H), 0)
        tl.store(output + row * H + h, tl.where(previous, value, carried), h < H)


class ResidentDraftMaintenance:
    """Device-bound prompt maintenance inside the existing native chunk loop.

    No model computation is introduced: original provider operators consume the
    same token, position, feedback and separately granted provider page tables.
    Coordination kernels preserve stored values and explicit row ownership.
    """

    def __init__(self, target, provider, window):
        binding = getattr(provider, '_draft_binding', None)
        if (binding is None or target.captured_m != provider.captured_m
                or target._target_hidden_slots is None):
            raise ValueError('native prompt maintenance lacks a bound provider contract')
        self.target, self.provider, self.window = target, provider, window
        self.feedback = getattr(target.body, 'draft_feedback', target.body.buffers.norm)
        self.durable = target._target_hidden_slots
        self.output = provider.body.buffers.target_hidden
        if (self.output is None or self.output.shape != self.feedback.shape
                or self.output.dtype != self.feedback.dtype
                or self.durable.shape[1:] != self.feedback.shape[1:]
                or self.durable.dtype != self.feedback.dtype
                or not all(t.is_contiguous() for t in (self.output,self.feedback))
                or not slot_storage_contiguous(self.durable)):
            raise ValueError('native prompt feedback storage differs from declared contract')
        self.M, self.H = self.feedback.shape
        # Fixed-owner prompt windows already prove source == destination.
        # Their owner identity belongs to common BatchRow ingress, regardless
        # of whether a model has any recurrent-state maps.
        self.owner_slots = target._metadata_device[:, target._state_destination_column]
        self.private_states = torch.tensor(binding.executor._scratch_states,
                                          device=self.feedback.device,dtype=torch.int32)
        self.private_pages = torch.tensor(binding.executor._scratch_pages,
                                         device=self.feedback.device,dtype=torch.int32)
        self.updates = torch.full((self.M,3),-1,device=self.feedback.device,dtype=torch.int32)
        self.reset_activity = torch.zeros(self.M,device=self.feedback.device,dtype=torch.bool)
        self.table_host = torch.empty_like(provider.body.buffers.page_table,
                                          device='cpu',pin_memory=True)
        self.table_array = self.table_host.numpy()

    def bind(self, widths, tables):
        b = self.provider.body.buffers
        if (len(widths) != len(tables) or sum(widths)>self.M or not widths
                or any(type(w) is not int or w<=0 for w in widths)):
            raise ValueError('native prompt provider row geometry differs')
        self.table_array[:] = np.asarray(self.provider._draft_binding.executor._scratch_pages)[:,None]
        first=0
        for width,table in zip(widths,tables,strict=True):
            if (type(table) is not tuple or not table or len(table)>b.page_table.shape[1]
                    or len(set(table))!=len(table)
                    or any(type(page) is not int or not 0<=page<self.provider.body.arena.spec.page_capacity for page in table)):
                raise ValueError('native prompt provider table exceeds its grant')
            self.table_array[first:first+width,:len(table)] = table
            first+=width
        b.page_table.copy_(self.table_host,non_blocking=True)

    def prepare(self):
        tb=self.target.body.buffers;pb=self.provider.body.buffers
        span=getattr(self.provider.body,'embedding_input_span',self.provider.body.embedding_span)
        prompt_provider_metadata[(self.M,)](
            tb.row_activity,tb.sequence_lengths,tb.position_ids,self.window.input_tokens,
            self.window.meta[2:3],self.owner_slots,
            self.private_states,self.private_pages,pb.token_ids,self.provider.body.embedding_mask,
            pb.sequence_lengths,pb.position_ids,pb.token_offsets,pb.page_source,pb.page_destination,
            tb.segment_lengths,pb.segment_lengths,pb.row_activity,tuple(pb.state_slot_maps),
            pb.page_table,self.updates,self.reset_activity,self.M,pb.page_table.shape[1],self.owner_slots.stride(0),
            tuple(m.shape[1] for m in pb.state_slot_maps),self.provider.body.arena.layout.block_size_tokens,
            span.start,span.stop,256,num_warps=4)
        from cubic_serve.ops.cuda.pointwise.qwen_decode import update_resident_pages
        arena=self.provider.body.arena
        update_resident_pages(self.updates,arena.key,arena.value,arena.written,
            pb.token_offsets,pb.page_source,pb.page_destination,pb.page_table,
            page_fields=tuple(arena.page_fields.values()),update_activity=self.reset_activity)

    def feedback_copy(self, publish=False):
        b=self.target.body.buffers
        prompt_provider_feedback[(self.M,triton.cdiv(self.H,256))](
            b.row_activity,self.window.owner_starts,b.segment_lengths,self.owner_slots,
            self.feedback,self.durable,self.output,self.M,self.owner_slots.stride(0),
            self.H,256,publish,DURABLE_STRIDE=self.durable.stride(0),num_warps=4)

    def launch(self):
        self.prepare()
        self.feedback_copy()
        self.provider.body.launch_state()
        self.feedback_copy(publish=True)


def capture_draft_prompt_windows(providers, targets, lifetime, *, qualify=False, reuse=False, qualification=None, progress=None):
    """Compose common inline provider bodies; qualify before advertising."""
    from cubic_serve.executor.resident_window import (
        ResidentDecodeWindow, SharedResidentJournal, resident_audit_rows)
    from cubic_serve.runtime.continuous_batch import RESIDENT_STEP_CAPACITY
    from cubic_serve.runtime.performance import _CapturedGraphRank
    pending = []
    for capacity, target in sorted(targets.items()):
        provider = providers.get(capacity)
        # An inline body has no unmatched pipeline send/receive. Other graph
        # contracts keep their existing sealed worker path until composed.
        bodies = (target.body, None if provider is None else provider.body)
        if (capacity < 2 or provider is None or not isinstance(target,_CapturedGraphRank)
                or getattr(provider,'_draft_binding',None) is None
                or not callable(getattr(provider.body,'launch_state',None))
                or target._target_hidden_slots is None
                or any(getattr(body,'previous',None) is not None
                       or getattr(body,'next',None) is not None
                       or getattr(body,'_pp_peer',None) is not None for body in bodies)):
            continue
        sampler = target._sampling.greedy_plan(target.slot.logits)
        if sampler is None:
            continue
        pending.append((capacity, target, provider, sampler))
    if progress is not None:
        progress(0, len(pending), phase="prompt graphs")
    for completed, (capacity, target, provider, sampler) in enumerate(pending, 1):
        if reuse:
            window = getattr(target, "_resident_window", None)
            maintenance = getattr(window, "draft_maintenance", None)
            if (not isinstance(window, ResidentDecodeWindow) or window.e is not target
                    or window.exec is None or maintenance is None
                    or maintenance.provider is not provider):
                raise ValueError("retained prompt graph differs from bound target/provider storage")
        else:
            journal=None
            if target._rank==target._sampler_rank:
                journal=SharedResidentJournal(64+RESIDENT_STEP_CAPACITY*(capacity+3),target.body.hidden.device)
                lifetime.callback(journal.close)
            window=ResidentDecodeWindow(target,sampler,journal=journal,draft_provider=provider)
            lifetime.callback(window.close)
            target._resident_window=window
        if qualify:
            domain=getattr(target.body,'bucket_logical_ms',None)
            if domain is None:
                from cubic_serve.runtime.performance import _logical_m_values
                domain=_logical_m_values(capacity)
            witnesses = resident_audit_rows(capacity,domain)
            if qualification is None:
                window.qualify_segments(witnesses)
            else:
                qualification.window(window, segments=witnesses)
        if progress is not None:
            progress(completed, len(pending), phase="prompt graphs")
