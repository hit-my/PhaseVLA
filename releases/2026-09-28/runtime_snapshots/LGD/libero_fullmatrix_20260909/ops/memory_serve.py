from pathlib import Path
import sys,os,ast,types,runpy,json,random
R=Path('/pfs/pfs-7jnepv/lgd/libero_fullmatrix_20260909');P=R/'MemoryVLA'
sys.path[:0]=[str(R/'memory_site'),str(P)]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MEMORYVLA_LLAMA2_7B_PATH=str(P/'pretrained/Llama-2-7b-hf'))
import torch,numpy as np,timm
torch.set_num_threads(4)
# All evaluated checkpoints contain the complete trained vision backbone.
# Avoid downloading initialization weights that the checkpoint replaces strictly.
create=timm.create_model
def create_local(*args,**kwargs):kwargs['pretrained']=False;return create(*args,**kwargs)
timm.create_model=create_local
load=torch.load
def load_checked(path,*args,**kwargs):
 kwargs.setdefault('weights_only',False)
 result=load(path,*args,**kwargs)
 if str(path).endswith('/checkpoints/model.pt'):
  assert {'vision_backbone','llm_backbone','projector','action_model'}<=set(result['model']),'Incomplete trained model'
 return result
torch.load=load_checked
# Dataset materialization is training-only; the inference package remains original.
init=P/'vla/__init__.py';tree=ast.parse(init.read_text())
tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='materialize')]
vla=types.ModuleType('vla');vla.__file__=str(init);vla.__package__='vla';vla.__path__=[str(P/'vla')];sys.modules['vla']=vla
exec(compile(tree,str(init),'exec'),vla.__dict__)
create_vla=vla.load_vla
def seeded_vla(*args,**kwargs):
 model=create_vla(*args,**kwargs)
 seed=int(os.environ['FULLMATRIX_EVAL_SEED']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 return model
vla.load_vla=seeded_vla
# Import the original action ensembler without initializing the unrelated
# Simpler robot-policy package (LIBERO uses deploy.py's policy directly).
import evaluation
package=types.ModuleType('evaluation.simpler_env')
package.__package__='evaluation.simpler_env'
package.__path__=[str(P/'evaluation/simpler_env')]
sys.modules['evaluation.simpler_env']=package
if '--preflight' in sys.argv:
 imports=ast.parse((P/'deploy.py').read_text())
 imports.body=[n for n in imports.body if isinstance(n,(ast.Import,ast.ImportFrom))]
 exec(compile(imports,str(P/'deploy.py'),'exec'),{})
 print(json.dumps({'status':'imports_passed','torch':torch.__version__,'cuda_arches':torch.cuda.get_arch_list(),'timm':timm.__version__}));sys.exit(0)
os.chdir(os.environ['FULLMATRIX_JOB_WORKDIR'])
runpy.run_path(str(P/'deploy.py'),run_name='__main__')
