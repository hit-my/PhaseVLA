"""Isolated corrected single-seed capacity and handover training."""
from pathlib import Path
import sys, os, json, dataclasses, importlib.util, hashlib

R = Path(__file__).parent
D = Path('/pfs/pfs-7jnepv/lgd/libero_multiseed_migration_20260908')
F = Path('/pfs/pfs-7jnepv/lgd/libero_fullmatrix_20260909')
M = Path('/pfs/pfs-7jnepv/lgd/actmem_gradientfix_v1_20260910')
P = D / 'PhaseVLA'
sys.path[:0] = [str(M), str(D/'isolated_speedup'), str(P/'src'), str(P)]
job = json.loads(Path(sys.argv[1]).read_text())
logs = Path(job['log_dir'])
from mamba_differentiable_step import install_training_fix
install_training_fix()
spec = importlib.util.spec_from_file_location('relocation', D/'ops/train_entry_lgd.py')
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)
a.CONDITIONING = R/'conditioning_cache' if job['task'] == 6 else D/'assets/conditioning_cache_handoff04_all10'
if job['task'] == 6:
    a.ACTIONS = R/'action_cache'
os.environ.update(OPENPI_DATA_HOME=str(D/'assets/openpi'), WANDB_MODE='offline', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
from openpi.training import config as tc
original = tc.get_config
_, cfg = a.relocate_config(original(job['config']), job, None)
change = job['change']
model = cfg.model
if change['kind'] == 'handover':
    model = dataclasses.replace(model, handoff_ratio=change['value'])
elif change['kind'] == 'memory_width':
    model = dataclasses.replace(model, memory=dataclasses.replace(model.memory, d_model=change['value']))
elif change['kind'] == 'memory_depth':
    model = dataclasses.replace(model, memory=dataclasses.replace(model.memory, depth=change['value']))
elif change['kind'] == 'pae_depth':
    model = dataclasses.replace(model, progress_depth=change['value'], progress_layer_mapping=None)
else:
    raise ValueError(change)
cfg = dataclasses.replace(cfg, model=model, num_train_steps=3000)
assert cfg.seed == cfg.model.train_seed == 42
assert cfg.model.terminal_loss_weight == cfg.model.handoff_loss_weight == cfg.model.boundary_loss_weight == 0
assert cfg.model.memory.d_model*cfg.model.memory.expand % cfg.model.memory.headdim == 0
tc.get_config = lambda name: cfg if name == job['config'] else original(name)
a.configure_wandb(logs, probe=False)
identity = {'job': job, 'training_version': 'differentiable_recurrent_step_v1',
            'model_metadata': cfg.model.checkpoint_metadata(),
            'effective_base': cfg.pytorch_weight_path, 'conditioning_root': str(a.CONDITIONING),
            'optimizer': dataclasses.asdict(cfg.optimizer), 'lr_schedule': dataclasses.asdict(cfg.lr_schedule),
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [Path(__file__), M/'mamba_differentiable_step.py', P/'src/openpi/models_pytorch/mamba_memory.py', D/'isolated_speedup/accelerated_futuremamba_train_all10.py', D/'isolated_speedup/isolated_futuremamba_speedup_benchmark.py']}}
(logs/'gradient_version_identity.json').write_text(json.dumps(identity, indent=2))
import torch
from openpi.models_pytorch.mamba_memory import Mamba2MemoryBackend
torch.set_num_threads(4)
backend = Mamba2MemoryBackend(cfg.model.memory).cuda().to(torch.bfloat16)
x = torch.randn(1,4,cfg.model.memory.d_model,device='cuda',dtype=torch.bfloat16,requires_grad=True)
state = backend.initial_state(1,device=x.device,dtype=x.dtype)
for i in range(4):
    y,state = backend.step(x[:,i],state)
grad = torch.autograd.grad((y.float()*torch.randn_like(y.float())).sum(),x)[0]
norms = grad.float().norm(dim=-1)
assert torch.isfinite(grad).all() and (norms > 0).all()
(logs/'gradient_preflight.json').write_text(json.dumps({'passed': True, 'block_gradient_norms': norms.tolist()}))
del backend,x,state,y,grad,norms
torch.cuda.empty_cache()
import accelerated_futuremamba_train_all10 as training
sys.argv = [training.__file__, *a.training_arguments(job,cfg)]
print(json.dumps({'event':'ablation_configuration_verified','model_metadata':identity['model_metadata']}),flush=True)
raise SystemExit(training.main())
