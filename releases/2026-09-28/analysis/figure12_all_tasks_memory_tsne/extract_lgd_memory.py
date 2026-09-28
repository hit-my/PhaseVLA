"""Extract original LGD checkpoint memory from identically preprocessed replay inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from safetensors import safe_open
from openpi.training import config as tc
from openpi.models_pytorch.futuremamba import FutureMambaPluginPytorch

out=Path(__file__).resolve().parent
inputs=out/'replay_inputs.npz'
selection=json.loads((out/'selected_checkpoints.json').read_text())
cfg=tc.get_config('futuremamba_action_history_handoff04_libero_mem_bowl_t6')
manifest={'tasks':[],'execution_host':'LGD:9024','input_file_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest()}
with np.load(inputs,allow_pickle=False) as data,torch.inference_mode():
    metadata=json.loads(str(data['metadata']))
    for task in [3,4,5,9]:
        row=next(x for x in selection if x['task']==task)
        ckpt=Path(row['job']['checkpoint'])/'plugin.safetensors'
        m=json.loads(ckpt.with_name('metadata.json').read_text())
        assert m['train_seed']==row['seed'] and m['step']==row['step']
        assert m['action_history_chunk_size']==20
        assert m['memory_config']['d_model']==cfg.model.memory.d_model==1024
        assert m['memory_config']['depth']==cfg.model.memory.depth==2
        plugin=FutureMambaPluginPytorch(cfg.model,include_progress_expert=False)
        with safe_open(ckpt,framework='pt',device='cpu') as f:
            weights={k:f.get_tensor('futuremamba.'+k) for k in plugin.state_dict()}
        tensor_hashes={k:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in weights.items()}
        plugin.load_state_dict(weights,strict=True);del weights
        plugin=plugin.cuda().eval();dtype=plugin.empty_history.dtype
        arrays={k:[] for k in ['memory','query','episode','fraction']};episodes=[]
        for index,source in enumerate(metadata[str(task)]['episodes']):
            a=data[f'T{task:02d}_{index}'];n=len(a)
            assert a.shape==(n,32) and n==source['length']
            history=plugin.initial_history_state(1,torch.device('cuda'),dtype)
            queries=np.arange(20,n,20);tokens=[];previous=0
            for q in queries:
                actions=torch.as_tensor(a[previous:q],device='cuda',dtype=dtype)[None]
                mask=torch.ones(actions.shape[:2],device='cuda',dtype=torch.bool)
                token,history,diag=plugin.advance_history(history,actions,mask)
                assert diag['history_actions']==q and diag['pending_actions']==0
                tokens.append(token[0,0].float().cpu().numpy());previous=q
            arrays['memory'].append(np.asarray(tokens));arrays['query'].append(queries)
            arrays['episode'].append(np.full(len(queries),index));arrays['fraction'].append(queries/(n-1))
            episodes.append(dict(**source,queries=len(queries)))
        result={k:np.concatenate(v) for k,v in arrays.items()}
        assert result['memory'].shape[1]==1024 and np.isfinite(result['memory']).all()
        np.savez_compressed(out/f'T{task:02d}_memory.npz',**result)
        info=dict(task=task,task_name=m['task_name'],version='corrected',seed=m['train_seed'],step=row['step'],evaluation_sr=row['sr'],checkpoint=str(ckpt),original_checkpoint=row['job']['checkpoint'],memory_tensor_sha256=tensor_hashes,source=metadata[str(task)]['source'],episodes=episodes,memory_shape=list(result['memory'].shape),execution_host='LGD:9024')
        manifest['tasks'].append(info)
        (out/'lgd_extraction_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(f'T{task}: {len(result["memory"])} real memory tokens',flush=True)
        del plugin,history,actions,token;torch.cuda.empty_cache()
print('COMPLETE',flush=True)
