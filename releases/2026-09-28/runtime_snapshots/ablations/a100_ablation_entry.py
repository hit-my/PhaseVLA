from pathlib import Path
import sys,os,json,dataclasses,hashlib
R=Path(__file__).parent;B=Path('/data/libero_mem_baseline');P=Path('/home/nvidia/zyx/PhaseVLA');V=B/'actmem_gradientfix_v1_20260910'
sys.path[:0]=[str(V),str(B/'isolated_speedup'),str(P/'src'),str(P)]
job=json.loads(Path(sys.argv[1]).read_text());logs=Path(job['log_dir']);logs.mkdir(parents=True,exist_ok=True)
from mamba_differentiable_step import install_training_fix
install_training_fix()
from openpi.training import config as tc
original=tc.get_config;cfg=original(job['config']);change=job['change'];model=dataclasses.replace(cfg.model,train_seed=42)
if change['kind']=='handover':model=dataclasses.replace(model,handoff_ratio=change['value'])
elif change['kind']=='memory_width':model=dataclasses.replace(model,memory=dataclasses.replace(model.memory,d_model=change['value']))
elif change['kind']=='memory_depth':model=dataclasses.replace(model,memory=dataclasses.replace(model.memory,depth=change['value']))
elif change['kind']=='pae_depth':model=dataclasses.replace(model,progress_depth=change['value'],progress_layer_mapping=None)
else:raise ValueError(change)
cfg=dataclasses.replace(cfg,seed=42,model=model,num_train_steps=3000,wandb_run_name=job['id'])
assert cfg.model.terminal_loss_weight==cfg.model.handoff_loss_weight==cfg.model.boundary_loss_weight==0
assert cfg.save_interval==500 and cfg.batch_size==1
tc.get_config=lambda n:cfg if n==job['config'] else original(n)
conditioning=R/'conditioning_cache_pae6' if model.progress_depth==6 else B/'conditioning_cache_handoff04_all10'
identity={'job':job,'training_version':'differentiable_recurrent_step_v1','platform':'A100','model_metadata':model.checkpoint_metadata(),'conditioning_root':str(conditioning),'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),V/'mamba_differentiable_step.py']},'optimizer':dataclasses.asdict(cfg.optimizer),'lr_schedule':dataclasses.asdict(cfg.lr_schedule)}
(logs/'gradient_version_identity.json').write_text(json.dumps(identity,indent=2))
os.environ.update(WANDB_MODE='offline',WANDB_DIR=str(logs),WANDB_INIT_TIMEOUT='90')
import torch
torch.set_num_threads(4)
from openpi.models_pytorch.mamba_memory import Mamba2MemoryBackend
backend=Mamba2MemoryBackend(model.memory).cuda().to(torch.bfloat16);x=torch.randn(1,4,model.memory.d_model,device='cuda',dtype=torch.bfloat16,requires_grad=True);state=backend.initial_state(1,device=x.device,dtype=x.dtype)
for i in range(4):y,state=backend.step(x[:,i],state)
grad=torch.autograd.grad((y.float()*torch.randn_like(y.float())).sum(),x)[0];norms=grad.float().norm(dim=-1);assert torch.isfinite(grad).all() and (norms>0).all()
(logs/'gradient_preflight.json').write_text(json.dumps({'passed':True,'norms':norms.tolist()}));del backend,x,state,y,grad,norms;torch.cuda.empty_cache()
import accelerated_futuremamba_train_all10 as training
if model.progress_depth==2:
 old=training.CompleteCachedEpisodeDataset.__getitem__
 def select_layers(self,index):
  e=old(self,index)
  for key in ['action_expert_keys','action_expert_values']:
   value=e.conditioning_cache[key];assert value.shape[1]==4;e.conditioning_cache[key]=value[:,[0,3]].copy()
  return e
 training.CompleteCachedEpisodeDataset.__getitem__=select_layers
cache=job['config'];key=Path('/home/nvidia/.config/phasevla/wandb_ten_task.key')
sys.argv=[training.__file__,'--config',job['config'],'--source-step','0','--checkpoint-root',job['checkpoint_root'],'--log-file',str(logs/'train.log'),'--wandb-run-name',job['id'],'--action-cache-root',str(B/'isolated_speedup/action_cache_handoff04_all10'),'--conditioning-cache-root',str(conditioning),'--action-cache-name',cache,'--conditioning-cache-name',cache,'--wandb-entity','2023112993-harbin-institute-of-technology','--wandb-api-key-file',str(key),'--num-train-steps','3000']
raise SystemExit(training.main())
