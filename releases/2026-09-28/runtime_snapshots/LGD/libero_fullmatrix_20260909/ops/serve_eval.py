from pathlib import Path
import os,sys,dataclasses,runpy,random
R=Path('/pfs/pfs-7jnepv/lgd/libero_fullmatrix_20260909');D=Path('/pfs/pfs-7jnepv/lgd/libero_multiseed_migration_20260908');P=R/'policy_snapshot'
sys.path[:0]=[str(P/'src'),str(P),str(P/'packages/openpi-client/src')]
os.environ['OPENPI_DATA_HOME']=str(D/'assets/openpi')
from openpi.shared import download
original_download=download.maybe_download
def relocated_download(uri,*args,**kwargs):
 prefix='/data/libero_mem_baseline/pytorch/pi05_libero_mem_all10_step49999_float32'
 if str(uri).removeprefix('file://').startswith(prefix):return str(D/'assets/pytorch/pi05_libero_mem_all10_step49999_float32')+str(uri).removeprefix('file://')[len(prefix):]
 return original_download(uri,*args,**kwargs)
download.maybe_download=relocated_download
from openpi.training import config as tc
get=tc.get_config
def get_config(name):
 if name=='pi05_libero_mem_baseline':
  from openpi.models.pi0_config import Pi0Config
  cfg=tc.TrainConfig(name=name,model=Pi0Config(pi05=True,action_horizon=20,discrete_state_input=False,pytorch_compile_mode=None),data=tc.LeRobotLiberoDataConfig(repo_id='libero-mem/LIBERO-Mem-Lerobot',base_config=tc.DataConfig(prompt_from_task=True),extra_delta_transform=False))
 else:cfg=get(name)
 seed=int(os.environ['FULLMATRIX_TRAIN_SEED']);model=cfg.model
 if hasattr(model,'train_seed'):model=dataclasses.replace(model,train_seed=seed)
 data=cfg.data
 if hasattr(data,'assets') and data.assets.assets_dir:
  data=dataclasses.replace(data,assets=dataclasses.replace(data.assets,assets_dir=str(D/'assets/pytorch/pi05_libero_mem_all10_step49999_float32/assets')))
 return dataclasses.replace(cfg,seed=seed,model=model,data=data)
tc.get_config=get_config
from openpi.policies import policy_config
from openpi.training import memory_ae_checkpoint
load_nope_base=memory_ae_checkpoint.load_base
def relocated_nope_base(model_config,device,*,weight_path=None):
 return load_nope_base(model_config,device,weight_path=relocated_download(str(weight_path or model_config.base_checkpoint_uri)))
memory_ae_checkpoint.load_base=relocated_nope_base
create=policy_config.create_trained_policy
def seeded_policy(*args,**kwargs):
 policy=create(*args,**kwargs)
 import numpy as np,torch
 seed=int(os.environ['FULLMATRIX_EVAL_SEED']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 if hasattr(policy,'_rng'):
  import jax
  policy._rng=jax.random.key(seed)
 return policy
policy_config.create_trained_policy=seeded_policy
runpy.run_path(str(P/'scripts/serve_policy.py'),run_name='__main__')
