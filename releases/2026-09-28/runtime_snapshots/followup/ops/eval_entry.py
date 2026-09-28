from pathlib import Path
import sys,os,json,ast,argparse
R=Path('/data/libero_mem_baseline/a100_followup_20260912');B=Path('/data/libero_mem_baseline');P=Path('/home/nvidia/zyx/PhaseVLA');D=Path('/pfs/pfs-7jnepv/lgd/libero_multiseed_migration_20260908')
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--trials',type=int,default=20);a=p.parse_args()
j=json.loads(Path(a.job).read_text());model=j['model'];ck=Path(j['checkpoint']);task=j['task']-1
os.environ['FOLLOWUP_GPU_UUID']=json.loads((R/'manifest.json').read_text())['gpu_uuids'][a.gpu]
os.environ['CUDA_VISIBLE_DEVICES']='' # CPU-only simulator; EGL visibility is independent and explicitly mapped
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['MUJOCO_EGL_DEVICE_ID']=str(json.loads((R/'manifest.json').read_text())['cuda_to_egl'][str(a.gpu)])
if model=='MemoryVLA':
 import runpy
 runpy.run_path(str(R/'ops/memory_eval.py'),run_name='__main__');sys.exit(0)
source=R/'eval_assets/run_handoff04_bowl_video_eval.py'
tree=ast.parse(source.read_text())
names=json.loads((R/'task_names.json').read_text());out=R/'evaluations'/j['id']
replacements={'PORT':9100+a.gpu,'GPU':a.gpu,'SELECTED_TASK_ID':task,'CHECKPOINT_STEP':j['step'],'TASK_IDS':[task],'TASK_NAMES':{task:names[str(j['task'])]},'POLICY_CONFIGS':{task:j['config']},'POLICY_CONFIG':j['config'],'TRAIN_SEED':j['train_seed'],'ROLLOUT_SEED':j['eval_seed'],'TRIALS':a.trials}
paths={'BASE':B,'PHASE':P,'OFFICIAL':Path('/home/nvidia/libero-mem-official'),'RUNNER_PYTHON':Path('/data/libero_mem_baseline/SANE/.venv/bin/python'),'POLICY_PYTHON':Path('/home/nvidia/zyx/PhaseVLA/environments/futuremamba/.venv/bin/python'),'SERVER':R/'ops/serve_eval.py','OUTPUT_ROOT':out,'OUTPUT':out,'POLICY_DIR':ck,'IDENTITY_FILE':ck/'metadata.json'}
if model=='pi05':paths['IDENTITY_FILE']=Path(j['identity_file'])
seen=set()
for node in tree.body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
  key=node.targets[0].id
  if key in replacements:node.value=ast.parse(repr(replacements[key]),mode='eval').body;seen.add(key)
  if key in paths:node.value=ast.parse('Path('+repr(str(paths[key]))+')',mode='eval').body;seen.add(key)
assert {'GPU','TRAIN_SEED','ROLLOUT_SEED','OUTPUT','POLICY_DIR'}<=seen,seen
class RelocateAssets(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='Path' and node.args and isinstance(node.args[0],ast.Subscript) and isinstance(node.args[0].slice,ast.Constant) and node.args[0].slice.value=='assets_uri':node.args=[ast.Constant(str(Path('/data/libero_mem_baseline/pytorch/pi05_libero_mem_all10_step49999_float32/assets')))]
  if isinstance(node.func,ast.Attribute) and node.func.attr=='Args':
   node.keywords=[kw for kw in node.keywords if kw.arg!='stabilize_libero_mem_objects'];node.keywords.append(ast.keyword(arg='stabilize_libero_mem_objects',value=ast.Constant(False)))
  return node
tree=RelocateAssets().visit(tree)
if model in ['no-PE','pi05']:
 class NoPE(ast.NodeTransformer):
  def visit_Constant(self,node):
   if node.value=='plugin.safetensors':node.value='memory_ae.safetensors' if model=='no-PE' else 'model.safetensors'
   if node.value=='action_history_mamba_pe_ae_handoff':node.value='action_history_mamba_memory_ae' if model=='no-PE' else 'none'
   return node
  def visit_Call(self,node):
   self.generic_visit(node)
   if isinstance(node.func,ast.Attribute) and node.func.attr=='Args':
    for kw in node.keywords:
     if kw.arg=='handoff_ratio':kw.value=ast.Constant(0.)
   return node
 tree=NoPE().visit(tree)
os.environ.update(FULLMATRIX_TRAIN_SEED=str(j['train_seed']),FULLMATRIX_EVAL_SEED=str(j['eval_seed']))
# Batch inference can exceed the client's default heartbeat deadline on shared GPUs.
import websockets.sync.client as websocket_transport
original_connect=websocket_transport.connect
def batch_connect(*args,**kwargs):
 kwargs.update(ping_interval=None,ping_timeout=None)
 return original_connect(*args,**kwargs)
websocket_transport.connect=batch_connect
assert not out.exists(),str(out)
ns={'__name__':'fullmatrix_canonical_eval','__file__':str(source)};exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),ns)
original_audit=ns['audit']
def audit(*args,**kwargs):
 report=original_audit(*args,**kwargs);report['fullmatrix_job']=j;report['policy_rng_seed']=j['eval_seed']
 report['protocol']['stabilize_libero_mem_objects']=False
 if model in ['no-PE','pi05']:
  report['protocol'].update(handoff_ratio=0.,denoising_order='action_expert_only',progress_denoise_steps=0,action_expert_denoise_steps=10)
 if model in ['no-memory','pi05']:
  report['protocol'].update(memory_update_timing='none',partial_chunk_behavior='none',empty_history_behavior='none')
 report_path=out/'audited_summary.json';report_path.write_text(json.dumps(report,indent=2));return report
ns['audit']=audit
sys.exit(ns['main']())
