"""Captured shared round vs independent pipelined CPU transaction.
Synthetic independently rounded target/provider math, real shared graph body,
real native one-draft verifier. No model floating-point/frontier qualification.
"""
import hashlib
import pytest
from types import SimpleNamespace as S
import torch, torch.distributed as dist, triton
from cubic_serve.executor.draft_graph import CapturedDraftRound
from cubic_serve.executor.draft import (DraftSpec, DraftStep, PendingDraft,
 reference_pending_draft, reference_pipelined_draft_round)
from cubic_serve.generation.tp_sampling import SpeculativeSamplingDecision
D = torch.device('cuda', 0)
DEPTH=1;STRIDE=2;CAPACITY=2
H=5;V=32;PAGE=4;T=12;K=20
STOCHASTIC=False

def eq(x,y,label):
 a=x.detach().cpu().contiguous();b=y.detach().cpu().contiguous()
 if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
  raise AssertionError(label+': '+str((a.shape,b.shape,a.dtype,b.dtype)))
def onehot(token,device):
 if STOCHASTIC:
  return (-((torch.arange(V,device=device)-token)%V).float()/4).bfloat16().reshape(1,1,V)
 z=torch.full((1,1,V),-17.,dtype=torch.bfloat16,device=device);z[0,0,token]=17.;return z

def cpu_step(token,feedback,state,axes,provider):
 # This independent CPU step owns its immutable state and rounds before both
 # output production and the following step, rather than sharing GPU traversal.
 previous=state['rnn'].double()
 add=torch.tensor([(token%7)/32,axes[0]/64,axes[1]/128],dtype=torch.float64)
 if provider:add=add+feedback[0,0,:3].double()/8
 current=(previous/2+add).to(torch.bfloat16)
 raw=torch.cat((current,torch.tensor([axes[2],token],dtype=torch.bfloat16)))
 norm=raw.clone();norm[-1]=((token+1+axes[0]%2) if provider else token+1)%V
 new={'rnn':current,'clock':state['clock']+1,'tokens':(*state['tokens'],token)}
 return DraftStep(onehot(int(norm[-1]),torch.device('cpu')),norm[None,None],new)
class Provider:
 @property
 def spec(self):return DraftSpec('audit.hidden',V,H,DEPTH)
 def forward(self,tokens,feedback,state=None,*,position_ids=None):
  return cpu_step(int(tokens[0,0]),feedback,state,position_ids[:,0,0].tolist(),True)

def arena(C):
 owners=2*C+7;pages=(owners+CAPACITY)*T
 return S(spec=S(state_slot_capacity=owners,page_capacity=owners*T),
  layout=S(block_size_tokens=PAGE),states={'rnn':torch.zeros((owners+CAPACITY,3),dtype=torch.bfloat16,device=D),
  'clock':torch.full((owners+CAPACITY,1),2**54+3,dtype=torch.int64,device=D)},
  key=torch.zeros((pages,PAGE,H),dtype=torch.bfloat16,device=D),
  value=torch.zeros((pages,PAGE,H),dtype=torch.bfloat16,device=D),
  written=torch.zeros((pages,PAGE),dtype=torch.bool,device=D),page_fields={})

def sampler(m,body,mask=None,activity=None):
 from cubic_serve.generation.tp_sampling import ShardedSamplingParameters, ShardedSamplingWorkspace
 from cubic_serve.models.parallel_weights import AxisSpan
 from cubic_serve.ops.cuda.sampling import RootTopKTopPSamplingPrimitive
 from cubic_serve.ops.cuda.sampling.greedy import CapturedGreedySampler
 workspace=ShardedSamplingWorkspace(
  local_scores=torch.empty((m,K),dtype=torch.bfloat16,device=D),
  local_token_ids=torch.empty((m,K),dtype=torch.int64,device=D),
  gathered_scores=torch.empty((1,m,K),dtype=torch.bfloat16,device=D),
  gathered_token_ids=torch.empty((1,m,K),dtype=torch.int64,device=D))
 tok=torch.empty(m,dtype=torch.int64,device=D)
 parameters=[torch.zeros(m,device=D,dtype=d) for d in (torch.float32,torch.int64,torch.float32,torch.int64,torch.int64)]
 state=S(workspace=workspace,temperatures=parameters[0],top_ks=parameters[1],top_ps=parameters[2],seeds=parameters[3],ordinals=parameters[4])
 state.constraint_bound=False
 state.root_valid=torch.empty(m,dtype=torch.bool,device=D)
 params=ShardedSamplingParameters(parameters[0],parameters[1],parameters[2],K,parameters[3],parameters[4])
 plan=CapturedGreedySampler(
  local_logits=body.logits,local_vocab_span=AxisSpan(0,V),final_ranks=(0,),
  final_vocab_spans=(AxisSpan(0,V),),true_vocab_size=V,token_ids=tok,
  parameters=params,workspace=workspace,final_tp_group=dist.group.WORLD,
  world_group=dist.group.WORLD,root_valid=state.root_valid,
  capture_stream=torch.cuda.Stream(device=D),token_mask=mask,constrained_rows=activity)
 state.root_sampler=RootTopKTopPSamplingPrimitive(workspace.gathered_scores,workspace.gathered_token_ids,
      parameters[0],parameters[2],parameters[1],parameters[3],parameters[4],tok,state.root_valid)
 return state,tok,plan

class Body:
 def __init__(self,m,cache,provider):
  self.arena=cache;self.provider=provider;self.hidden=torch.zeros((m,H),dtype=torch.bfloat16,device=D)
  self.embedding_mask=torch.ones((m,1),dtype=torch.bool,device=D);self.embedding_span=S(start=0,stop=V)
  self.logits=torch.zeros((m,V),dtype=torch.bfloat16,device=D)
  def meta(*shape,dtype=torch.int32):return torch.zeros(shape,dtype=dtype,device=D)
  self.buffers=S(token_ids=meta(m,dtype=torch.int64),position_ids=meta(3,m),page_source=meta(m),page_destination=meta(m),token_offsets=meta(m),sequence_lengths=meta(m),segment_lengths=meta(m),row_activity=meta(m),state_slot_maps=(meta(2,1,m),meta(2,1,m)),page_table=meta(m,T if provider else T+2),norm=torch.zeros_like(self.hidden),target_hidden=torch.zeros_like(self.hidden))
  self.draft_feedback=self.buffers.norm
  def reset(indices):
   for field in (cache.key,cache.value,cache.written):field.index_fill_(0,indices,0)
  cache.reset_page_rows=reset
 def launch(self,*,produce_logits=True):
  b=self.buffers;m=len(b.token_ids);src=b.state_slot_maps[0][0,0].long();dst=b.state_slot_maps[0][1,0].long()
  before=self.arena.states['rnn'].index_select(0,src)
  clocks=self.arena.states['clock'].index_select(0,src)
  for r in range(0,m,STRIDE if m==CAPACITY else 1):
   prev=None;clock=None
   for j in range(min(STRIDE if m==CAPACITY else 1,m-r)):
    row=r+j
    base=before[row].float();oldclock=clocks[row]
    if j:
     cont=b.segment_lengths[r]>j
     base=torch.where(cont,prev.float(),base);oldclock=torch.where(cont,clock,oldclock)
    token=b.token_ids[row].float();axes=b.position_ids[:,row].float()
    add=torch.stack((token.remainder(7)/32,axes[0]/64,axes[1]/128))
    if self.provider:add=add+b.target_hidden[row,:3].float()/8
    current=(base/2+add).bfloat16();newclock=oldclock+1
    self.arena.states['rnn'].index_copy_(0,dst[row:row+1],current[None])
    self.arena.states['clock'].index_copy_(0,dst[row:row+1],newclock[None])
    raw=torch.cat((current,axes[2:3].bfloat16(),token.reshape(1).bfloat16()))
    self.hidden[row].copy_(raw);b.norm[row].copy_(raw)
    prediction=(token+1+(axes[0].remainder(2) if self.provider else 0)).remainder(V)
    b.norm[row,-1].copy_(prediction)
    page=b.page_destination[row].long();offset=b.token_offsets[row].long()
    index=(page*PAGE+offset).reshape(1)
    self.arena.key.reshape(-1,H).index_copy_(0,index,raw[None]);self.arena.value.reshape(-1,H).index_copy_(0,index,b.norm[row:row+1]);self.arena.written.reshape(-1).index_fill_(0,index,True)
    prev=current;clock=newclock
  if produce_logits:self.launch_projection()
 def set_active_rows(self,n):pass
 def launch_projection(self):
  prediction=self.buffers.norm[:,-1].long().remainder(V)
  if STOCHASTIC:
   self.logits.copy_((-((torch.arange(V,device=D)[None,:]-prediction[:,None])%V).float()/4).bfloat16())
  else:self.logits.fill_(-17);self.logits.scatter_(1,prediction[:,None],17.)


def setup(C,*,budget=False,split=False,target_only=False):
 M=CAPACITY;ta,pa=arena(C),arena(C);targetbody=Body(M,ta,False);alignbody=Body(M,pa,True);headbody=Body(C,pa,True)
 mask=torch.full((M,1),-1,dtype=torch.int32,device=D) if budget else None
 activity=torch.zeros(M,dtype=torch.bool,device=D) if budget else None
 target=S(body=targetbody,_rank=0,_sampler_rank=0,_world_group=dist.group.WORLD,vocab_size=V,_metadata_columns=14+T+2,_position_axes_slice=slice(7,10),_state_destination_column=12)
 target.slot=S(stream=torch.cuda.Stream(device=D));target._metadata_device=torch.zeros((M,target._metadata_columns),dtype=torch.int32,device=D)
 target._target_hidden_slots=torch.zeros((ta.states['rnn'].shape[0],H),dtype=torch.bfloat16,device=D);target._state_destinations=torch.empty(M,dtype=torch.int64,device=D)
 target._replay_started=torch.cuda.Event(enable_timing=True,external=True);target._replay_finished=torch.cuda.Event(enable_timing=True,external=True)
 def bind(md,m,t):
  b=targetbody.buffers;b.token_ids.copy_(md[:,0]);b.page_source.copy_(md[:,1]);b.page_destination.copy_(md[:,2]);b.token_offsets.copy_(md[:,3]);b.sequence_lengths.copy_(md[:,4]);b.segment_lengths.copy_(md[:,5]);b.position_ids.copy_(md[:,7:10].T)
  for maps in b.state_slot_maps:maps[0,0].copy_(md[:,10]);maps[1,0].copy_(md[:,11])
  b.page_table.copy_(md[:,14:])
 target._bind_staged_metadata=bind
 target._sampling,_,tgreedy=sampler(M,targetbody,mask,activity)
 head=S(body=headbody,_sampling=None,_draft_binding=S(feedback=headbody.draft_feedback),
  _verification_q_scores=torch.zeros((DEPTH,1,C,K),dtype=torch.bfloat16,device=D),
  _verification_q_token_ids=torch.zeros((DEPTH,1,C,K),dtype=torch.int64,device=D),
  _verification_draft_tokens=torch.zeros((DEPTH,C),dtype=torch.int64,device=D),
  verification_decisions=torch.zeros((DEPTH,C,2),dtype=torch.int64,device=D))
 head._sampling,head._token_ids,hgreedy=sampler(C,headbody,
   None if mask is None else mask[:C],None if activity is None else activity[:C])
 head.captured_m=C;head._max_draft_tokens=DEPTH
 head._verification_validity=torch.empty(DEPTH,dtype=torch.bool,device=D)
 head._verification_constraint_rows=torch.empty((DEPTH,C),dtype=torch.bool,device=D)
 head._sampling.constraint_bound=False
 head._sampling._constraint_activity=None if activity is None else activity[:C]
 from cubic_serve.runtime.performance import CapturedDraftRank
 from types import MethodType
 head.stash_proposal=MethodType(CapturedDraftRank.stash_proposal,head)
 for name,param in zip(('temperatures','top_ks','top_ps','seeds','ordinals'),(head._sampling.temperatures,head._sampling.top_ks,head._sampling.top_ps,head._sampling.seeds,head._sampling.ordinals)):
  setattr(head,'_verification_'+name,torch.empty((DEPTH,C),device=D,dtype=param.dtype))
 pool=S(owners=pa.spec.state_slot_capacity,scores=torch.empty((DEPTH,1,pa.spec.state_slot_capacity,K),dtype=torch.bfloat16,device=D),ids=torch.empty((DEPTH,1,pa.spec.state_slot_capacity,K),dtype=torch.int64,device=D),tokens=torch.empty((DEPTH,pa.spec.state_slot_capacity),dtype=torch.int64,device=D),backup={k:torch.empty_like(v) for k,v in pa.states.items()})
 obj=object.__new__(CapturedDraftRound);obj.target,obj.align,obj.head,obj.pool=target,S(body=alignbody,state_fields=pa.states),head,pool
 obj._review_graphs={};obj._review_capture_open=True;obj._retain_audit_graph=False
 obj._graph_pool=torch.cuda.graph_pool_handle();obj._review_graph_pool=torch.cuda.graph_pool_handle()
 obj._capture_candidates=True
 obj._capture_budget=False;obj._last_budget=False;obj.budget=None
 obj._capture_review_only=False;obj._last_review_only=False
 obj._capture_target_constraints=target_only;obj._last_target_constraints=False
 obj.constraint_forward_valid=torch.ones((),dtype=torch.bool,device=D)
 obj.C,obj.M,obj.T,obj.CS,obj.target_T=C,M,T,2+DEPTH+T,T+2
 obj.depth=DEPTH;obj.stride=STRIDE;obj.commit_CS=3+DEPTH;obj.pool_scores=pool.scores;obj.pool_ids=pool.ids;obj.pool_tokens=pool.tokens
 obj.decisions=torch.empty((C,2),device=D,dtype=torch.int64);obj.next_tokens=torch.empty((DEPTH,C),device=D,dtype=torch.int64)
 obj.ctx=torch.zeros(1+C*obj.CS,dtype=torch.int32,device=D);obj.host=torch.zeros_like(obj.ctx,device='cpu',pin_memory=True)
 obj.commit_ctx=torch.zeros(1+C*obj.commit_CS,dtype=torch.int32,device=D);obj.commit_host=torch.zeros_like(obj.commit_ctx,device='cpu',pin_memory=True)
 obj.metadata_host=torch.zeros_like(target._metadata_device,device='cpu',pin_memory=True)
 obj.index_device=torch.arange(M,device=D,dtype=torch.int64)+ta.spec.page_capacity
 obj.index_host=obj.index_device.cpu().pin_memory()
 obj.param_hosts=tuple(torch.zeros_like(p,device='cpu',pin_memory=True) for p in (head._sampling.temperatures,head._sampling.top_ks,head._sampling.top_ps,head._sampling.seeds,head._sampling.ordinals))
 obj.param_hosts[1].fill_(1);obj.param_hosts[2].fill_(1)
 obj.rows=tuple(torch.arange(C,device=D,dtype=torch.int32)*STRIDE+d for d in range(DEPTH));obj.active=torch.empty(C,dtype=torch.bool,device=D);obj.valid=torch.empty((),dtype=torch.bool,device=D)
 obj.result=torch.empty((C,DEPTH+3),dtype=torch.int64,device=D);obj.host_result=torch.empty_like(obj.result,device='cpu',pin_memory=True)
 obj.host_decisions=torch.empty((C,2),dtype=torch.int64,device='cpu',pin_memory=True)
 obj.round_start=torch.cuda.Event(enable_timing=True,external=True);obj.round_end=torch.cuda.Event(enable_timing=True,external=True)
 obj.target_greedy,obj.head_greedy=tgreedy,hgreedy;obj.head_launch=headbody.launch_projection
 obj.scratch_state=pa.spec.state_slot_capacity;obj.scratch_page=pa.spec.page_capacity
 if budget:
  from cubic_serve.ops.cuda.sampling.reasoning_budget import ReasoningBudgetProgram
  obj._capture_budget=True;obj._capture_candidates=STOCHASTIC
  obj.budget=ReasoningBudgetProgram(((2,3),(30,31),(4,5)),mask,activity)
  obj.target_masked,obj.head_masked=tgreedy,hgreedy
  obj.budget_review=obj.budget.prefix(obj.ctx,obj.next_tokens,owners=C,depth=DEPTH,vocabulary=V)
  obj.budget_heads=tuple(obj.budget.prefix(obj.ctx,obj.next_tokens,owners=C,depth=DEPTH,vocabulary=V,head_step=d) for d in range(DEPTH))
  obj.budget_commit=obj.budget.commit(obj.ctx,obj.decisions,owners=C,depth=DEPTH,vocabulary=V)
 if split:
  obj._capture_budget=False;obj._capture_review_only=True;obj._capture_candidates=True
  obj.target_masked.qualify();obj.head_masked.qualify()
 if target_only:
  obj._capture_budget=False;obj._capture_candidates=STOCHASTIC
  obj.target_masked.qualify()
  # q is the actual unmasked proposal distribution, independent of p's mask.
  head._sampling,head._token_ids,obj.head_greedy=sampler(C,headbody)
  head._sampling.constraint_bound=False;head._sampling._constraint_activity=None
 # Padding owns distinct source/destination state and pages even in warmup.
 fill(obj,{},[],0)
 target.slot.stream.wait_stream(torch.cuda.current_stream(D))
 with torch.cuda.stream(target.slot.stream):obj.body();obj.body()
 target.slot.stream.synchronize();obj.capture_review_graph(target.slot.stream)
 obj._forward_graphs={}
 obj.capture_forward_graph(target.slot.stream)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=target.slot.stream,pool=obj._graph_pool):obj.completion_body()
 target.slot.stream.synchronize();obj.graph=graph
 if split:obj.capture_grammar_proposals(target.slot.stream)
 return obj

def fill(obj,records,order,ordinal):
 C,M=obj.C,obj.M;ta=obj.target.body.arena
 obj.host.zero_();obj.commit_host.zero_();obj.host[0]=obj.commit_host[0]=len(order)
 arr=obj.host[1:].reshape(C,obj.CS);cc=obj.commit_host[1:].reshape(C,obj.commit_CS)
 md=obj.metadata_host;md.zero_()
 if obj.budget is not None:
  obj.budget.bind([records[owner]['budget'].integer_snapshot() for owner in order] or [None])
 for row in range(M):
  dest=ta.spec.state_slot_capacity+row;page=ta.spec.page_capacity+row
  md[row,1:3]=page;md[row,4]=1;md[row,10:13]=dest;md[row,14:]=page
 for r in range(C):
  owned=r<len(order);owner=order[r] if owned else 0;record=records.get(owner)
  root=record['pending'].root_token_id if owned else 0
  qs=record['pending'].proposal_tokens if owned else (0,)*DEPTH
  p=record['pending'].position if owned else 0
  axes=record['pending'].position_axes if owned else (0,0,0)
  step=record['pending'].position_step if owned else (1,1,1)
  arr[r,:2]=torch.tensor((p,owner));arr[r,2:2+DEPTH]=torch.tensor(qs)
  arr[r,2+DEPTH:]=torch.arange(T,dtype=torch.int32)+owner*T if owned else obj.scratch_page+STRIDE*r
  cc[r,:3]=torch.tensor((0,owner,obj.scratch_state+STRIDE*r))
  cc[r,3:]=ta.spec.state_slot_capacity+STRIDE*r+torch.arange(DEPTH)
  for j in range(STRIDE):
   row=STRIDE*r+j;pos=p+j if owned else 0
   page=owner*T+pos//PAGE if owned else ta.spec.page_capacity+row
   dest=owner if owned and j==DEPTH else ta.spec.state_slot_capacity+row
   src=owner if owned else ta.spec.state_slot_capacity+row
   md[row,0]=(root,*qs)[j];md[row,1:3]=page;md[row,3]=pos%PAGE;md[row,4]=pos+1
   md[row,5]=STRIDE if owned and j==0 else 0
   md[row,7:10]=torch.tensor([a+j*d for a,d in zip(axes,step)])
   md[row,10]=src;md[row,11]=dest;md[row,12]=dest
   md[row,14:]=torch.arange(T+2,dtype=torch.int32)+owner*T if owned else page
  obj.param_hosts[-1][r]=record['rngordinal'] if STOCHASTIC and owned else ordinal
  if STOCHASTIC:
   obj.param_hosts[0][r]=1.;obj.param_hosts[1][r]=20;obj.param_hosts[2][r]=.95
   obj.param_hosts[3][r]=0x123456789+owner


def cpu_distribution(logits):
 import math
 ordered=sorted(((t,s) for t,s in enumerate(logits.flatten().tolist()) if math.isfinite(s)),key=lambda x:(-x[1],x[0]))[:K]
 weights=[math.exp(score-ordered[0][1]) for _,score in ordered];total=sum(weights)
 probabilities=[w/total for w in weights];kept={};cdf=0.
 for (token,_),probability in zip(ordered,probabilities):
  if cdf<=.95:kept[token]=probability
  cdf+=probability
 total=sum(kept.values());return {token:p/total for token,p in kept.items()}

def categorical(probabilities,u):
 cdf=0.
 for token,p in probabilities.items():
  cdf+=p
  if cdf>u:return token
 raise AssertionError('categorical outside probability mass')

def cpu_sample(logits,owner,ordinal):
 from tests.ops.test_cuda_greedy_sampling import _philox_uniform
 seed=0x123456789+owner
 return categorical(cpu_distribution(logits),_philox_uniform(seed^(seed>>32)^(ordinal>>32),ordinal))

def cpu_verify(qs,qlogits,ps,owner,ordinal):
 from tests.ops.test_cuda_greedy_sampling import _philox_uniform
 seed=0x123456789+owner;key=(seed^(seed>>32)^0x9E3779B9)&0xffffffff
 for d,q in enumerate(qs):
  qdist,p0,p1=map(cpu_distribution,(qlogits[d],ps[d],ps[d+1]))
  u=_philox_uniform(key,(ordinal+d)*2)
  v=_philox_uniform(key^0x85EBCA6B,(ordinal+d)*2+1)
  if u>=min(1.,p0.get(q,0.)/qdist[q]):
   residual={token:max(0.,p-qdist.get(token,0.)) for token,p in p0.items()}
   mass=sum(residual.values())
   return SpeculativeSamplingDecision(d,categorical({t:p/mass for t,p in residual.items()},v))
 return SpeculativeSamplingDecision(len(qs),categorical(p1,v))

def seed(obj,n):
 ta,pa=obj.target.body.arena,obj.align.body.arena;records={};provider=Provider()
 for owner in range(n):
  state={'rnn':torch.tensor([.125+owner/64,.25,.375],dtype=torch.bfloat16),'clock':2**54+13+owner,'tokens':(3,7,11)}
  axes=(3,7+owner,11+owner);root=(owner+5)%V
  feedback=torch.tensor([.5,.75,1.,3.,7.],dtype=torch.bfloat16).reshape(1,1,H)
  budget=None
  if obj.budget is not None:
   from cubic_serve.contracts.request import ReasoningControl
   from cubic_serve.generation.constraints import ConstraintSpec,create_logit_constraint
   budget=create_logit_constraint(spec=ConstraintSpec.identity(),request_id='budget',reasoning_control=ReasoningControl((2,3),(30,31),owner%3,True,implicit_ends=((4,5),)))._matcher
  branch=None if budget is None or obj._capture_target_constraints else budget.fork()
  def initial_propose(logits,d):
   masked=budget_logits(logits,branch)
   token=cpu_sample(masked,owner,(1<<33)+owner+d) if STOCHASTIC else int(masked.argmax())
   if branch is not None:assert branch.accept_token(token)
   return token
  pending=reference_pending_draft(provider=provider,root_token=torch.tensor([[root]]),feedback=feedback,draft_state=state,position=3,depth=DEPTH,propose=initial_propose,position_axes=axes,position_step=(1,2,3))
  # Fix the first proposal/decision trace explicitly to exercise both branches.
  qs=pending.proposal_tokens if STOCHASTIC or budget is not None else tuple((root+d+1+(int(d>=owner%(DEPTH+1))))%V for d in range(DEPTH))
  pending=PendingDraft(root,3,qs,pending.proposal_steps,pending.root_step,axes,(1,2,3))
  records[owner]={'pending':pending,'target':dict(state),'draft':state,'rngordinal':(1<<33)+owner}
  if budget is not None:records[owner]['budget']=budget
  ta.states['rnn'][owner].copy_(state['rnn']);ta.states['clock'][owner,0]=state['clock']
  pa.states['rnn'][owner].copy_(pending.root_step.state['rnn']);pa.states['clock'][owner,0]=pending.root_step.state['clock']
  obj.pool.scores[:,:,owner].fill_(float('nan'));obj.pool.ids[:,:,owner].fill_(-1)

  branch=None if budget is None or obj._capture_target_constraints else budget.fork()
  for d,q in enumerate(qs):
   masked=budget_logits(pending.proposal_steps[d].logits,branch)
   if STOCHASTIC:
    scores,ids=masked.flatten().topk(K);valid=torch.isfinite(scores)
    obj.pool.scores[d,0,owner].copy_(scores.masked_fill(~valid,float('nan')));obj.pool.ids[d,0,owner].copy_(ids.masked_fill(~valid,-1))
   else:obj.pool.scores[d,:,owner,0]=masked.flatten()[q];obj.pool.ids[d,:,owner,0]=q
   if branch is not None:assert branch.accept_token(q)
   obj.pool.tokens[d,owner]=q
  # Only semantic prefix is protected; physical tentative review tails may write.
  for pos in range(3):
   for cache in (ta,pa):cache.key[owner*T,pos].fill_(pos+owner);cache.value[owner*T,pos].fill_(2*pos+owner);cache.written[owner*T,pos]=True
  raw=torch.cat((pending.root_step.state['rnn'],torch.tensor([axes[2],root],dtype=torch.bfloat16)))
  pa.key[owner*T,3].copy_(raw);pa.value[owner*T,3].copy_(pending.root_step.feedback[0,0]);pa.written[owner*T,3]=True
 return records

def budget_logits(logits,matcher):
 if matcher is None:return logits
 mask=matcher.fill_token_mask();result=logits.flatten().clone()
 if mask.needs_apply:matcher.apply_token_mask(result,mask.payload)
 return result.reshape(logits.shape)

def cpu_mask_planes(obj,matchers):
 masks=torch.full_like(obj.budget.mask,-1,device='cpu')
 active=torch.zeros_like(obj.budget.activity,device='cpu')
 for row,matcher in enumerate(matchers):
  allowed=torch.isfinite(budget_logits(torch.zeros(V),matcher)).nonzero().flatten().tolist()
  word=sum(1<<token for token in allowed)
  masks[row,0]=word if word<2**31 else word-2**32;active[row]=True
 obj.budget.mask.copy_(masks);obj.budget.activity.copy_(active)

def finish_split_fixture(obj,records,order):
 from cubic_serve.executor.draft_graph import joint_result
 future=[]
 for row,owner in enumerate(order):
  accepted,bonus=obj.host_decisions[row].tolist()
  branch=records[owner]['budget'].fork()
  for token in (*records[owner]['pending'].proposal_tokens[:accepted],bonus):assert branch.accept_token(token)
  future.append(branch)
 obj.valid.fill_(True)
 for depth in range(DEPTH):
  cpu_mask_planes(obj,future)
  tokens=obj.run_grammar_proposal(depth,len(order),stochastic=STOCHASTIC)
  for branch,token in zip(future,tokens):assert branch.accept_token(token)
 joint_result[(1,)](obj.ctx,obj.decisions,obj.next_tokens,obj.valid,obj.result,obj.C,obj.CS,V,triton.next_power_of_2(obj.C),DEPTH)
 obj.host_result.copy_(obj.result)

def audit(obj,n):
 records=seed(obj,n);provider=Provider();cases=[]
 for ordinal in range(4):
  order=list(records)
  if ordinal%2:order.reverse()
  fill(obj,records,order,ordinal)
  if obj._capture_review_only or obj._capture_target_constraints:
   matchers=[]
   for owner in order:
    branch=records[owner]['budget'].fork()
    for d in range(DEPTH+1):
     matchers.append(branch.fork())
     if d<DEPTH:
      token=records[owner]['pending'].proposal_tokens[d]
      allowed=bool(torch.isfinite(budget_logits(torch.zeros(V),branch)[token]))
      if obj._capture_target_constraints and not allowed:
       matchers.extend([None]*(DEPTH-d));break
      assert branch.accept_token(token)
   cpu_mask_planes(obj,matchers)
  for cache in (obj.target.body.arena,obj.align.body.arena):
   cache.states["rnn"][cache.spec.state_slot_capacity:].fill_(float("nan"))
   cache.key[cache.spec.page_capacity:].fill_(float("nan"))
   cache.value[cache.spec.page_capacity:].fill_(float("nan"))
  prefixes={}
  for name,cache in (("target",obj.target.body.arena),("provider",obj.align.body.arena)):
   for owner,record in records.items():
    p=record["pending"].position
    for position in range(p):
     page=owner*T+position//PAGE;offset=position%PAGE
     prefixes[name,owner,position]=(cache.key[page,offset].clone(),cache.value[page,offset].clone(),cache.written[page,offset].clone())
  protected={}
  for name,cache in (('target',obj.target.body.arena),('provider',obj.align.body.arena)):
   protected[name]=(cache.states['rnn'][n:cache.spec.state_slot_capacity].clone(),cache.states['clock'][n:cache.spec.state_slot_capacity].clone(),cache.key[n*T:cache.spec.page_capacity].clone(),cache.value[n*T:cache.spec.page_capacity].clone(),cache.written[n*T:cache.spec.page_capacity].clone())
  obj.target.slot.stream.wait_stream(torch.cuda.current_stream(D))
  with torch.cuda.stream(obj.target.slot.stream):obj.replay_ordinary(obj.graph,budget=obj._capture_budget)
  obj.target.slot.stream.synchronize()
  if obj._capture_review_only:finish_split_fixture(obj,records,order)
  schedules=[None];dist.all_gather_object(schedules,('round',obj.M,n,ordinal,tuple(order)))
  assert all(x==schedules[0] for x in schedules)
  built=obj.host_result[:n].tolist()
  for r,owner in enumerate(order):
   record=records[owner];pending=record['pending']
   def target(token,state,axes):return cpu_step(int(token[0,0]),None,state,axes[:,0,0].tolist(),False)
   def verify(qs,qlogits,ps):
    budget=record.get('budget')
    if budget is not None:
     branch=budget.fork();qmasked=[];pmasked=[]
     for d in range(DEPTH+1):
      pmasked.append(budget_logits(ps[d],branch))
      if d<DEPTH:
       qmasked.append(qlogits[d] if obj._capture_target_constraints else budget_logits(qlogits[d],branch))
       allowed=bool(torch.isfinite(pmasked[-1].flatten()[qs[d]]))
       if obj._capture_target_constraints and not allowed:
        pmasked.extend(ps[d+1:]);qmasked.extend(qlogits[d+1:]);break
       assert branch.accept_token(qs[d])
     qlogits,ps=tuple(qmasked),tuple(pmasked)
    if STOCHASTIC:
     decision=cpu_verify(qs,qlogits,ps,owner,record['rngordinal']);record['accepted']=decision.accepted_draft_tokens
    else:
     a=0
     for d,q in enumerate(qs):
      if q!=int(ps[d].argmax()):break
      a+=1
     decision=SpeculativeSamplingDecision(a,int(ps[a].argmax()))
    if budget is not None:
     for token in (*qs[:decision.accepted_draft_tokens],decision.next_root_token_id):assert budget.accept_token(token)
     record['next_budget']=budget.fork()
    return decision
   def propose(logits,d):
    branch=None if obj._capture_target_constraints else record.get('next_budget');masked=budget_logits(logits,branch)
    token=cpu_sample(masked,owner,record['rngordinal']+record['accepted']+1+d) if STOCHASTIC else int(masked.argmax())
    if branch is not None:assert branch.accept_token(token)
    return token
   result=reference_pipelined_draft_round(provider=provider,target=target,pending=pending,target_state=record['target'],device=torch.device('cpu'),next_depth=DEPTH,propose=propose,verify=verify)
   a,bonus,*qvalid=built[r];nextqs=tuple(qvalid[:-1]);valid=qvalid[-1]
   assert (a,bonus,nextqs,valid)==(result.decision.accepted_draft_tokens,result.decision.next_root_token_id,result.pending.proposal_tokens,1), (built[r],result.decision,result.pending.proposal_tokens)
   ta,pa=obj.target.body.arena,obj.align.body.arena
   eq(ta.states['rnn'][owner],result.target_state['rnn'],'target commit')
   eq(ta.states['clock'][owner],torch.tensor([result.target_state['clock']],dtype=torch.int64),'target typed clock')
   eq(obj.target._target_hidden_slots[owner],result.feedback[0,0],'target feedback checkpoint')
   eq(pa.states['rnn'][owner],result.pending.root_step.state['rnn'],'provider pending root')
   eq(pa.states['clock'][owner],torch.tensor([result.pending.root_step.state['clock']],dtype=torch.int64),'provider root typed clock')
   eq(obj.pool.backup['rnn'][owner],result.draft_state['rnn'],'provider durable checkpoint')
   eq(obj.pool.backup['clock'][owner],torch.tensor([result.draft_state['clock']],dtype=torch.int64),'provider durable typed clock')
   branch=None if 'budget' not in record or obj._capture_target_constraints else record['budget'].fork()
   for d,q in enumerate(nextqs):
    assert int(obj.pool.tokens[d,owner])==q
    masked=budget_logits(result.pending.proposal_steps[d].logits,branch)
    if STOCHASTIC:
     scores,ids=masked.flatten().topk(K);valid=torch.isfinite(scores)
     scores=scores.masked_fill(~valid,float('nan'));ids=ids.masked_fill(~valid,-1)
     actual_scores=obj.pool.scores[d,0,owner]
     # Inactive candidates carry a NaN sentinel; its payload bits are not a
     # floating value. Every actual candidate score remains byte-exact.
     eq(torch.isnan(actual_scores),torch.isnan(scores),'next q sentinel ownership')
     eq(actual_scores[valid.to(D)],scores[valid],'next q actual logits')
     eq(obj.pool.ids[d,0,owner],ids,'next q actual IDs')
    else:assert int(obj.pool.ids[d,0,owner,0])==q and float(obj.pool.scores[d,0,owner,0])==float(masked.flatten()[q])
    if branch is not None:assert branch.accept_token(q)
   if obj._capture_budget:
    _control,phase,used,tail=record['budget'].integer_snapshot();state=obj.budget.state[r].cpu()
    assert tuple(state[:4].tolist())==(phase,used,owner%3,len(tail))
    assert tuple(state[5:5+len(tail)].tolist())==tail
   eq(obj.head.body.buffers.norm[r],result.pending.proposal_steps[-1].feedback[0,0],'head producer boundary')
   for j,step in enumerate(result.target_steps[:a+1]):
    pos=pending.position+j;page=owner*T+pos//PAGE
    raw=torch.cat((step.state['rnn'],torch.tensor([pending.position_axes[2]+j*3,(pending.root_token_id,*pending.proposal_tokens)[j]],dtype=torch.bfloat16)))
    eq(ta.key[page,pos%PAGE],raw,'accepted target KV write')
   for j,step in enumerate(result.alignment_steps):
    pos=pending.position+j+1;page=owner*T+pos//PAGE
    token=(*pending.proposal_tokens[:a],bonus)[j]
    raw=torch.cat((step.state['rnn'],torch.tensor([pending.position_axes[2]+(j+1)*3,token],dtype=torch.bfloat16)))
    eq(pa.key[page,pos%PAGE],raw,'aligned provider KV write')
    eq(pa.value[page,pos%PAGE],step.feedback[0,0],'aligned provider KV rounded feedback')
   record.update(pending=result.pending,target=result.target_state,draft=result.draft_state,
                 rngordinal=record['rngordinal']+a+1)
   digest=hashlib.sha256()
   for value in (ta.states["rnn"][owner],ta.states["clock"][owner],pa.states["rnn"][owner],pa.states["clock"][owner],obj.pool.backup["rnn"][owner],obj.pool.backup["clock"][owner],obj.target._target_hidden_slots[owner],obj.pool.tokens[:,owner]):digest.update(repr((str(value.dtype),tuple(value.shape),value.detach().cpu().tolist())).encode())
   cases.append(dict(digest=digest.hexdigest(),depth=DEPTH,physical_m=obj.M,logical_requests=n,ordinal=ordinal,owner=owner,accepted=a,row=r))
  for (name,owner,position),values in prefixes.items():
   cache=obj.target.body.arena if name=="target" else obj.align.body.arena
   page=owner*T+position//PAGE;offset=position%PAGE
   for actual,expected in zip((cache.key[page,offset],cache.value[page,offset],cache.written[page,offset]),values):eq(actual,expected,"immutable prefix")
  for name,cache in (('target',obj.target.body.arena),('provider',obj.align.body.arena)):
   after=(cache.states['rnn'][n:cache.spec.state_slot_capacity],cache.states['clock'][n:cache.spec.state_slot_capacity],cache.key[n*T:cache.spec.page_capacity],cache.value[n*T:cache.spec.page_capacity],cache.written[n*T:cache.spec.page_capacity])
   for x,y in zip(after,protected[name]):eq(x,y,name+' foreign owner protection')
 return cases

@pytest.mark.skipif(not torch.cuda.is_available(), reason="captured draft transaction")
@pytest.mark.parametrize("depth", [1, 2, 3, 7])
@pytest.mark.parametrize("stochastic",[False,True])
def test_captured_round_matches_independent_rounded_transaction_cross_m(depth, stochastic, tmp_path):
    global DEPTH, STRIDE, CAPACITY, D, STOCHASTIC
    STOCHASTIC=stochastic
    D = torch.device("cuda", torch.cuda.current_device())
    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group("nccl", init_method="file://" + str(tmp_path / "rank"),
                                rank=0, world_size=1, device_id=D)
    elif dist.get_world_size() != 1:
        pytest.skip("single-rank transaction fixture")
    try:
        DEPTH, STRIDE = depth, depth + 1
        cases = []
        for capacity in (1, 2, 4, 8, 16, 32):
            CAPACITY = triton.next_power_of_2(capacity * STRIDE)
            round_ = setup(capacity)
            try:
                for logical in sorted({1, max(1, capacity - 1), capacity}):
                    cases.extend(audit(round_, logical))
            finally:
                round_.graph.reset()
                for graph in round_._forward_graphs.values():graph.reset()
                for graphs in getattr(round_, "grammar_proposal_graphs", {}).values():
                    for graph in graphs:graph.reset()
                round_.target_greedy.close()
                round_.head_greedy.close()
            del round_
        signatures = {}
        for case in cases:
            key = case["ordinal"], case["owner"]
            if key in signatures:
                assert signatures[key] == case["digest"], ("cross-M/permuted-row", case)
            signatures[key] = case["digest"]
        if not stochastic:assert set(range(depth + 1)) <= {case["accepted"] for case in cases}
        else:assert {0,depth} <= {case["accepted"] for case in cases}
    finally:
        if owned_group:
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captured budget transaction")
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("split", [False, True])
def test_budget_round_matches_independent_rounded_transaction(stochastic, depth, split, tmp_path):
    global DEPTH, STRIDE, CAPACITY, D, STOCHASTIC
    DEPTH, STRIDE, STOCHASTIC = depth, depth + 1, stochastic
    D = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", init_method="file://" + str(tmp_path / "rank"),
                            rank=0, world_size=1, device_id=D)
    try:
        signatures = {}
        for capacity in (1, 4):
            CAPACITY = triton.next_power_of_2(capacity * STRIDE)
            round_ = setup(capacity, budget=True, split=split)
            try:
                for logical in sorted({1, max(1, capacity - 1), capacity}):
                    for case in audit(round_, logical):
                        key = case["ordinal"], case["owner"]
                        if key in signatures:
                            assert signatures[key] == case["digest"], case
                        signatures[key] = case["digest"]
            finally:
                round_.graph.reset()
                for graph in round_._forward_graphs.values():graph.reset()
                for graphs in getattr(round_, "grammar_proposal_graphs", {}).values():
                    for graph in graphs:graph.reset()
                round_.target_greedy.close()
                round_.head_greedy.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captured target-constraint transaction")
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_target_only_constraints_match_independent_transaction(stochastic, depth, tmp_path):
    global DEPTH, STRIDE, CAPACITY, D, STOCHASTIC
    DEPTH, STRIDE, STOCHASTIC = depth, depth + 1, stochastic
    D = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", init_method="file://" + str(tmp_path / "rank"),
                            rank=0, world_size=1, device_id=D)
    try:
        signatures = {}
        for capacity in (1, 4):
            CAPACITY = triton.next_power_of_2(capacity * STRIDE)
            round_ = setup(capacity, budget=True, target_only=True)
            try:
                for logical in sorted({1, max(1, capacity - 1), capacity}):
                    for case in audit(round_, logical):
                        key = case["ordinal"], case["owner"]
                        if key in signatures:
                            assert signatures[key] == case["digest"], case
                        signatures[key] = case["digest"]
            finally:
                round_.graph.reset()
                for graph in round_._forward_graphs.values():graph.reset()
                round_.target_greedy.close()
                round_.head_greedy.close()
                round_.head_masked.close()
    finally:
        dist.destroy_process_group()
