from pathlib import Path
import base64,hashlib
root=Path(__file__).resolve().parent
payload=Path('C:/Users/lenovo/phasevla-session-migration-20260907/memory_replay_inputs_payload.txt').read_text().strip()
data=base64.b64decode(payload,validate=True)
(root/'replay_inputs.npz').write_bytes(data)
files={name:base64.b64encode((root/name).read_bytes()).decode() for name in ['replay_inputs.npz','selected_checkpoints.json','extract_lgd_memory.py']}
code="from pathlib import Path\nimport base64,subprocess,os,hashlib\nr=Path('/pfs/pfs-7jnepv/lgd')\nout=r/'analysis/all10_best_memory_tsne';out.mkdir(parents=True,exist_ok=True)\n"
for name,encoded in files.items():code+=f"(out/{name!r}).write_bytes(base64.b64decode({encoded!r}))\n"
code+=f"assert hashlib.sha256((out/'replay_inputs.npz').read_bytes()).hexdigest()=={hashlib.sha256(data).hexdigest()!r}\n"
code+="p=r/'libero_multiseed_migration_20260908/PhaseVLA'\nenv=os.environ.copy();env.update(PYTHONPATH=str(p/'src')+':'+str(p),JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')\nlog=open(out/'extract.log','w')\nproc=subprocess.Popen([str(r/'phasevla-runtime/bin/python'),str(out/'extract_lgd_memory.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint('LAUNCHED',proc.pid)\n"
(root/'launch_lgd_extraction.py').write_text(code,encoding='utf-8')
print('Replay input bytes',len(data))
