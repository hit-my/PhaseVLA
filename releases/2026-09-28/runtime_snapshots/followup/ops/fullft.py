"""Single-task full adaptation, stopped at a preregistered measured time budget."""
import argparse,json,os,time,random,traceback
from pathlib import Path
import numpy as np
import torch
import safetensors.torch
from openpi.training import config as configs
from openpi.training.memory_ae_data_loader import create_memory_ae_data
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
a=argparse.ArgumentParser();a.add_argument('--task',type=int,required=True);a.add_argument('--root',type=Path,required=True);a.add_argument('--reference',type=Path,required=True);args=a.parse_args()
root=args.root;root.mkdir(parents=True,exist_ok=True)
if (root/'status.json').exists():raise RuntimeError('Existing full-FT run: inspect before resuming')
start=time.monotonic()
def status(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),task=args.task,seed=42,updated_unix=time.time(),**kw)
    tmp=root/'status.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(root/'status.json')
try:
    ref=json.loads(args.reference.read_text());assert ref['status']=='completed' and ref['steps']==1500
    budget=float(ref['budget_seconds']);assert budget>0
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    cfg=configs.get_config(f'futuremamba_action_history_handoff04_libero_mem_bowl_t{args.task}')
    status('loading_data');data=create_memory_ae_data(cfg,queries_per_update=8)
    status('loading_model');model=PI0Pytorch(cfg.model).cuda()
    weights=Path(cfg.pytorch_weight_path)/'model.safetensors'
    safetensors.torch.load_model(model,weights,strict=True,device='cuda')
    model.requires_grad_(True);model.train();model.gradient_checkpointing_enable()
    params=[p for p in model.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    manifest=dict(experiment='single_task_pi05_incremental_time_budget',task=args.task,seed=42,
        reference=str(args.reference),budget_seconds=budget,budget_scope=ref['budget_scope'],
        caveat=ref['comparison'],base_weights=str(weights),episodes=data.episode_count,
        sampling='uniform episode then uniform query',effective_batch=8,microbatch=1,
        trainable_parameters=sum(p.numel() for p in params),lr=2.5e-5,betas=[.9,.95],eps=1e-8,weight_decay=.01,
        stop_rule='first complete optimizer update reaching measured 1500-step ActMem reference time; record overshoot',
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    stream=iter(data);elapsed=0.;step=0;torch.cuda.reset_peak_memory_stats()
    while elapsed<budget:
        tick=time.monotonic();batch=next(stream);optim.zero_grad(set_to_none=True);values=[]
        for i in range(8):
            part=batch.slice(i,i+1).to('cuda');mask=part.action_mask[...,None]
            actions=torch.where(mask,part.actions,0);noise=torch.where(mask,torch.randn_like(actions),0)
            losses=model(part.observation,actions,noise=noise)
            loss=(losses*mask).sum()/(mask.sum()*actions.shape[-1]);(loss/8).backward();values.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optim.step();torch.cuda.synchronize();step+=1;duration=time.monotonic()-tick;elapsed+=duration
        rec=dict(step=step,loss=sum(values)/8,grad_norm=float(norm),step_seconds=duration,
                 train_seconds=elapsed,budget_seconds=budget,queries=step*8,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec),flush=True);status('training',**rec)
    status('saving',**rec);cp=root/'checkpoint.tmp';cp.mkdir()
    safetensors.torch.save_model(model,str(cp/'model.safetensors'))
    torch.save(dict(optimizer=optim.state_dict(),step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),cp/'training_state.pt')
    (cp/'assets').symlink_to(weights.parent/'assets',target_is_directory=True)
    (cp/'manifest.json').write_text(json.dumps(manifest,indent=2));cp.rename(root/'checkpoint')
    # Keep the audit identity outside the checkpoint: metadata.json there denotes a plugin checkpoint.
    identity=json.loads(Path('/data/libero_mem_baseline/a100_followup_20260912/baseline_identity.json').read_text())
    identity.update(step=step,fullft_manifest=str(root/'manifest.json'),budget_seconds=budget)
    (root/'identity.json').write_text(json.dumps(identity,indent=2))
    status('completed',**rec,overshoot_seconds=elapsed-budget,total_seconds=time.monotonic()-start)
except Exception:
    status('failed',error=traceback.format_exc());raise
