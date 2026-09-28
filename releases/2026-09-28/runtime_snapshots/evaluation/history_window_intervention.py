"""Inference-only recent-history stress test; full state retained for bookkeeping.
Success under this intervention is a robustness diagnostic, not a retrained ablation.
"""
import os,torch
def install():
 value=os.environ.get('ACTMEM_DIAGNOSTIC_HISTORY_WINDOW')
 if value is None or int(value)<0:return
 keep=int(value);assert keep>0
 from openpi.models_pytorch.futuremamba import FutureMambaPytorch
 original=FutureMambaPytorch.sample_actions_with_memory
 def sample(self,observation,history,executed_actions,executed_action_mask,**kwargs):
  assert executed_actions.shape[0]==1
  empty=int(history.committed_chunks[0])==0 and not bool(history.pending_mask.any())
  previous=None if empty else getattr(self,'_diagnostic_recent_actions',None)
  incoming=executed_actions[:,executed_action_mask[0]].detach()
  recent=incoming if previous is None else torch.cat([previous,incoming],dim=1)
  recent=recent[:,-keep:].clone();self._diagnostic_recent_actions=recent
  total=int(history.committed_chunks[0])*self.futuremamba.chunk_size+int(history.pending_mask.sum())+int(executed_action_mask.sum())
  if total<=keep:return original(self,observation,history,executed_actions,executed_action_mask,**kwargs)
  with torch.no_grad():
   state=self.initial_history_state(1,executed_actions.device,next(self.futuremamba.parameters()).dtype)
   for begin in range(0,recent.shape[1],20):
    block=recent[:,begin:begin+20]
    token,state,_=self.futuremamba.advance_history(state,block,torch.ones(block.shape[:2],device=block.device,dtype=torch.bool))
  hook=self.futuremamba.memory_token_projection.register_forward_hook(lambda module,args,out:token.reshape_as(out).to(out.dtype))
  try:return original(self,observation,history,executed_actions,executed_action_mask,**kwargs)
  finally:hook.remove()
 FutureMambaPytorch.sample_actions_with_memory=sample
