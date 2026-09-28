"""Measure the production corrected cached training path on an exclusive GPU."""
import json, os, sys, time, runpy
from pathlib import Path
B=Path('/data/libero_mem_baseline'); P=Path('/home/nvidia/zyx/PhaseVLA')
R=B/'a100_followup_20260912'
job=json.loads(Path(sys.argv[sys.argv.index('--job')+1]).read_text())
out=Path(job['log_dir']);out.mkdir(parents=True,exist_ok=True)
os.environ['WANDB_MODE']='disabled'
sys.path[:0]=[str(B/'isolated_speedup'),str(P/'src'),str(P)]
import torch
torch.set_num_threads(4)
import isolated_futuremamba_speedup_benchmark as bench
original_load=bench.load_training_module
def load():
    mod=original_load(); original_run=mod.run_training
    def run(model,*args,**kwargs):
        start=time.monotonic(); elapsed=0.; count=0
        info=dict(task=job['task'],seed=42,reference_steps=1500,
            added_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('futuremamba.')),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in model.parameters()),
            budget_scope='sum synchronized optimizer-step wall time including data; excludes pre-existing cache construction, initialization, logging and checkpoint serialization',
            comparison='incremental adaptation with pre-existing ActMem caches, not end-to-end equal compute',
            gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],gpu=torch.cuda.get_device_name(),torch=torch.__version__)
        (out/'cost_manifest.json').write_text(json.dumps(info,indent=2))
        torch.cuda.reset_peak_memory_stats()
        def logger(metrics,step):
            nonlocal elapsed,count
            elapsed+=float(metrics['train/step_time']);count=step
            record=dict(step=step,metrics=dict(metrics),cumulative_step_seconds=elapsed,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,updated_unix=time.time())
            with (out/'cost_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            tmp=out/'cost_status.tmp';tmp.write_text(json.dumps(record));tmp.replace(out/'cost_status.json')
        kwargs['metric_logger']=logger
        result=original_run(model,*args,**kwargs)
        assert count==1500,count
        info.update(status='completed',steps=count,budget_seconds=elapsed,
                    training_and_saving_seconds=time.monotonic()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        tmp=out/'cost_result.tmp';tmp.write_text(json.dumps(info,indent=2));tmp.replace(out/'cost_result.json')
        return result
    mod.run_training=run
    return mod
bench.load_training_module=load
runpy.run_path(str(B/'actmem_gradientfix_v1_20260910/train_corrected.py'),run_name='__main__')
