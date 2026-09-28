from pathlib import Path
import subprocess,os
r=Path('/pfs/pfs-7jnepv/lgd');p=r/'libero_multiseed_migration_20260908/PhaseVLA'
env=os.environ.copy();env.update(PYTHONPATH=str(p/'src')+':'+str(p),JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2')
code="from openpi.training import config as tc;from openpi.models_pytorch.futuremamba import FutureMambaPluginPytorch;import torch; c=tc.get_config('futuremamba_action_history_handoff04_libero_mem_bowl_t6');p=FutureMambaPluginPytorch(c.model,include_progress_expert=False).cuda();print('READY',c.model.memory,torch.cuda.get_device_name())"
x=subprocess.run([str(r/'phasevla-runtime/bin/python'),'-c',code],env=env,capture_output=True,text=True)
print(x.returncode,x.stdout,x.stderr[-5000:])
