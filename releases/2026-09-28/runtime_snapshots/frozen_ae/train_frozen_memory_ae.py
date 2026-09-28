import argparse,json,time,random,traceback,hashlib,dataclasses
from pathlib import Path
import numpy as np
import torch
import safetensors.torch
from openpi.training import config as configs
from openpi.training.memory_ae_data_loader import create_memory_ae_data
from openpi.training.futuremamba_checkpoint import _capture_rng_state,_restore_rng_state
from frozen_memory_ae import FrozenMemoryAE,hybrid_chain

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True,choices=[6,7,8]);p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,default=3000);p.add_argument('--resume',action='store_true');p.add_argument('--seed',type=int,required=True);args=p.parse_args()
 root=args.root;root.mkdir(parents=True,exist_ok=True)
 if (root/'status.json').exists() and not args.resume:raise RuntimeError('Existing run; use --resume')
 started=time.monotonic()
 def status(phase,**kw):
  import os
  value=dict(phase=phase,pid=os.getpid(),task=args.task,seed=args.seed,updated_unix=time.time(),session_seconds=time.monotonic()-started,**kw)
  tmp=root/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(root/'status.json')
 try:
  torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
  cfg=configs.get_config(f'futuremamba_nope_memory_ae_libero_mem_bowl_t{args.task}')
  cfg=dataclasses.replace(cfg,seed=args.seed,model=dataclasses.replace(cfg.model,train_seed=args.seed))
  status('loading_data');data=create_memory_ae_data(cfg)
  status('loading_model');model=FrozenMemoryAE(cfg.model).cuda()
  weights=Path(cfg.pytorch_weight_path)/'model.safetensors'
  safetensors.torch.load_model(model.base,weights,strict=True,device='cuda');model.freeze_base();model.train()
  assert not any(p.requires_grad for p in model.base.parameters())
  params=[p for p in model.parameters() if p.requires_grad]
  assert all(name.startswith('futuremamba.') for name,p in model.named_parameters() if p.requires_grad)
  versions={n:p._version for n,p in model.base.named_parameters()}
  status('verifying_gradient_and_schedule')
  calls=[]
  hybrid_chain(torch.zeros(1,20,32,device='cuda'),10,.4,
   lambda x,t:(calls.append(('memory',float(t[0]))) or torch.ones_like(x)),
   lambda x,t:(calls.append(('plain',float(t[0]))) or torch.ones_like(x)))
  assert [v[0] for v in calls]==['memory']*4+['plain']*6
  assert abs(calls[4][1]-.6)<1e-6
  rng=_capture_rng_state()
  history=torch.randn(1,61,32,device='cuda',requires_grad=True)
  token=model._memory_token_from_history(history,torch.ones(1,61,device='cuda',dtype=torch.bool))
  token.float().square().mean().backward()
  norms=[float(history.grad[:,a:b].float().norm()) for a,b in [(0,20),(20,40),(40,60),(60,61)]]
  assert all(np.isfinite(n) and n>0 for n in norms),norms
  model.zero_grad(set_to_none=True)
  # Confirm AE operations transmit the actual flow loss gradient into memory.
  smoke=create_memory_ae_data(cfg,queries_per_update=2)
  batch=next(iter(smoke)).to('cuda');loss=model.compute_query_loss(batch)['loss'];loss.backward()
  gradnorm=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True))
  assert gradnorm>0 and all(p.grad is None for p in model.base.parameters())
  model.zero_grad(set_to_none=True);_restore_rng_state(rng)
  (root/'validation.json').write_text(json.dumps(dict(history_block_gradient_norms=norms,flow_loss=float(loss.detach()),memory_gradient_norm=gradnorm,base_has_grad=False,schedule=calls),indent=2))
  del history,token,batch,loss,smoke
  lr=float(getattr(cfg.lr_schedule,'peak_lr',2.5e-5));wd=float(getattr(cfg.optimizer,'weight_decay',1e-10))
  optim=torch.optim.AdamW(params,lr=lr,weight_decay=wd)
  metadata=dict(architecture='frozen_memory_ae_early04_v1',task=args.task,task_name=cfg.model.task_name,seed=args.seed,
   handoff_ratio=.4,num_denoise_steps=10,memory_calls=4,plain_ae_calls=6,PE=False,
   trainable_parameters=sum(p.numel() for p in params),base_frozen=True,base_weights=str(weights),
   base_manifest=json.loads((weights.parent/'conversion_manifest.json').read_text()),
   source_model_config=cfg.model.checkpoint_metadata(),queries_per_update=data.queries_per_update,microbatch_size=2,
   episodes=data.episode_count,lr=lr,weight_decay=wd,flow_time='0.6+0.4*Beta(1.5,1)',terminal_loss=0,
   training_history='differentiable full Mamba sequence; no inference state updates',
   query_sampling='uniform_episode_then_query',gpu=torch.cuda.get_device_name(),
   source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['frozen_memory_ae.py','train_frozen_memory_ae.py']})
  (root/'metadata.json').write_text(json.dumps(metadata,indent=2))
  step0=0;prior_elapsed=0.
  if args.resume:
   checkpoints=sorted(int(x.name) for x in root.iterdir() if x.is_dir() and x.name.isdigit())
   step0=checkpoints[-1];cp=root/str(step0);savedmeta=json.loads((cp/'metadata.json').read_text())
   for k in ['architecture','task','seed','base_weights','queries_per_update','lr','weight_decay','source_sha256']:
    if savedmeta[k]!=metadata[k]:raise ValueError(f'Resume identity mismatch: {k}')
   safetensors.torch.load_model(model.futuremamba,str(cp/'memory.safetensors'),strict=True)
   state=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False)
   optim.load_state_dict(state['optimizer']);_restore_rng_state(state['rng']);prior_elapsed=state['elapsed_seconds']
  iterator=data.iter_from_update(step0);train_start=time.monotonic()
  status('training',step=step0,queries_per_update=data.queries_per_update)
  for step in range(step0+1,args.steps+1):
   tick=time.monotonic();batch=next(iterator);optim.zero_grad(set_to_none=True);values=[]
   for i in range(0,len(batch.actions),2):
    part=batch.slice(i,min(i+2,len(batch.actions))).to('cuda');count=len(part.actions)
    loss=model.compute_query_loss(part)['loss'];(loss*count/len(batch.actions)).backward();values.append(float(loss.detach())*count)
   norm=torch.nn.utils.clip_grad_norm_(params,float(getattr(cfg.optimizer,'clip_gradient_norm',1.0)),error_if_nonfinite=True)
   optim.step();torch.cuda.synchronize()
   assert all(p.grad is None and p._version==versions[n] for n,p in model.base.named_parameters())
   elapsed=prior_elapsed+time.monotonic()-train_start
   rec=dict(step=step,loss=sum(values)/len(batch.actions),grad_norm=float(norm),step_seconds=time.monotonic()-tick,train_seconds=elapsed,queries=step*data.queries_per_update,peak_gib=torch.cuda.max_memory_allocated()/2**30)
   with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   print(json.dumps(rec),flush=True);status('training',**rec)
   if step%500==0 or step==args.steps:
    status('saving',step=step);cp=root/f'{step}.tmp';cp.mkdir()
    safetensors.torch.save_model(model.futuremamba,str(cp/'memory.safetensors'))
    torch.save(dict(optimizer=optim.state_dict(),rng=_capture_rng_state(),step=step,data_iterator_step=step,elapsed_seconds=elapsed),cp/'training_state.pt')
    (cp/'metadata.json').write_text(json.dumps(metadata,indent=2));cp.rename(root/str(step))
  status('completed',step=args.steps)
 except Exception:
  status('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
