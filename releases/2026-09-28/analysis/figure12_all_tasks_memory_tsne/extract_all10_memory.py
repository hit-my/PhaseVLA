"""Replay fixed first-five demonstration episodes for all ten task-specific models."""
from pathlib import Path
import argparse,hashlib,json
import h5py
import numpy as np
import torch
from safetensors import safe_open
from openpi.training import config as tc
from openpi.training.cached_query_data_loader import _ActionTransform
from openpi.models_pytorch.futuremamba import FutureMambaPluginPytorch

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--selection',required=True)
    ap.add_argument('--tasks',default='1,2,3,4,5,6,7,8,9,10')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    root=Path('/data/libero_mem_baseline')
    template=tc.get_config('futuremamba_action_history_handoff04_libero_mem_bowl_t6')
    dc=template.data.create(template.assets_dirs,template.model)
    transform=_ActionTransform(dc)
    selected=json.loads(Path(args.selection).read_text())
    manifest=dict(source_policy='Best corrected main-experiment checkpoints selected from audited success rates',selection='First five numerically ordered demonstration episodes per task, fixed independently of memory visualization.',
                  query_interval=20,empty_history_excluded=True,memory_input='Normalized and padded executed actions, not joint observations',
                  normalization='Original all10 base-policy action transform; quantile normalization='+str(dc.use_quantile_norm),
                  diagnostic='Offline demonstration replay, not held-out policy rollouts',tasks=[])
    if (out/'extraction_manifest.json').exists():
        manifest['tasks']=json.loads((out/'extraction_manifest.json').read_text())['tasks']
    for task in map(int,args.tasks.split(',')):
        chosen=next(r for r in selected if r['task']==task)
        ckpt=Path(chosen.get('local_checkpoint',chosen['job']['checkpoint']))/'plugin.safetensors'
        version='corrected'
        assert ckpt.exists(),ckpt
        metadata=json.loads(ckpt.with_name('metadata.json').read_text())
        raw=list((root/'raw').glob(f'KITCHEN_SCENE1_{task}_*.hdf5'))
        assert len(raw)==1,raw
        source=raw[0]
        assert metadata['action_history_chunk_size']==20
        assert metadata['memory_config']['d_model']==template.model.memory.d_model==1024
        assert metadata['memory_config']['depth']==template.model.memory.depth==2
        plugin=FutureMambaPluginPytorch(template.model,include_progress_expert=False)
        with safe_open(ckpt,framework='pt',device='cpu') as f:
            weights={k:f.get_tensor('futuremamba.'+k) for k in plugin.state_dict()}
        plugin.load_state_dict(weights,strict=True);del weights
        plugin=plugin.cuda().eval();dtype=plugin.empty_history.dtype
        arrays={k:[] for k in ['memory','query','episode','fraction']}
        episodes=[]
        with h5py.File(source,'r') as f,torch.inference_mode():
            keys=sorted(f['data'],key=lambda k:int(k.rsplit('_',1)[1]))[:5]
            assert len(keys)==5
            for index,key in enumerate(keys):
                e=f['data'][key];a=e['actions'][:].astype(np.float32);n=len(a)
                state=np.concatenate([e['obs/ee_states'][:],e['obs/gripper_states'][:]],axis=-1).astype(np.float32)
                processed=transform(a[:,None,:],state)[:,0]
                assert processed.shape==(n,32)
                history=plugin.initial_history_state(1,torch.device('cuda'),dtype)
                queries=np.arange(20,n,20);tokens=[];previous=0
                for q in queries:
                    actions=torch.as_tensor(processed[previous:q],device='cuda',dtype=dtype)[None]
                    mask=torch.ones(actions.shape[:2],device='cuda',dtype=torch.bool)
                    token,history,diag=plugin.advance_history(history,actions,mask)
                    assert diag['history_actions']==q and diag['pending_actions']==0
                    tokens.append(token[0,0].float().cpu().numpy());previous=q
                arrays['memory'].append(np.asarray(tokens));arrays['query'].append(queries)
                arrays['episode'].append(np.full(len(queries),index));arrays['fraction'].append(queries/(n-1))
                episodes.append(dict(id=key,length=n,queries=len(queries),actions_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        result={k:np.concatenate(v) for k,v in arrays.items()}
        assert np.isfinite(result['memory']).all() and len(result['memory'])>15
        np.savez_compressed(out/f'T{task:02d}_memory.npz',**result)
        info=dict(task=task,task_name=metadata['task_name'],version=version,seed=metadata['train_seed'],step=chosen['step'],evaluation_sr=chosen['sr'],
                  checkpoint=str(ckpt),checkpoint_sha256=sha256(ckpt),source=str(source),episodes=episodes,
                  memory_shape=list(result['memory'].shape))
        info['original_checkpoint']=chosen['job']['checkpoint']
        provenance=ckpt.with_name('subset_provenance.json')
        if provenance.exists():info['memory_subset_provenance']=json.loads(provenance.read_text())
        manifest['tasks']=[r for r in manifest['tasks'] if r['task']!=task]+[info]
        manifest['tasks'].sort(key=lambda r:r['task'])
        (out/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(f'T{task}: {version}, seed {info["seed"]}, {len(result["memory"])} real memory tokens',flush=True)
        del plugin,history,actions,token;torch.cuda.empty_cache()
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
