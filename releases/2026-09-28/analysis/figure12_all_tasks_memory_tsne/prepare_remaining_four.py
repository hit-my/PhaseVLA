"""Package verified LGD memory-only subsets for the original A100 runtime."""
from pathlib import Path
import base64,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
SESSION=Path('C:/Users/lenovo/phasevla-session-migration-20260907')
archive=ROOT/'lgd_memory_subsets.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for task in [3,4,5,9]:
        folder=ROOT/'lgd_memory_subsets'/f'T{task:02d}'
        prov=json.loads((folder/'subset_provenance.json').read_text())
        assert hashlib.sha256((folder/'plugin.safetensors').read_bytes()).hexdigest()==prov['subset_sha256']
        for name in ['plugin.safetensors','metadata.json','subset_provenance.json']:
            tar.add(folder/name,arcname=f'lgd_memory_subsets/T{task:02d}/{name}')
selected=json.loads((ROOT/'selected_checkpoints.json').read_text())
for row in selected:
    if row['task'] in [3,4,5,9]:
        row['local_checkpoint']=f'/data/libero_mem_baseline/analysis/all10_best_memory_tsne/lgd_memory_subsets/T{row["task"]:02d}'
selection=base64.b64encode(json.dumps(selected).encode()).decode()
script=base64.b64encode((ROOT/'extract_all10_memory.py').read_bytes()).decode()
# remote_read.py already provides the authenticated A100 SSH transport.
payload=base64.b64encode(archive.read_bytes()).decode()
upload='from pathlib import Path\nimport base64,hashlib\np=Path(\'/data/libero_mem_baseline/analysis/all10_best_memory_tsne/lgd_memory_subsets.tar.gz\')\np.write_bytes(base64.b64decode('+repr(payload)+'))\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(archive.read_bytes()).hexdigest())+'\nprint(\'Archive received and verified\')\n'
(SESSION/'upload_lgd_memory_archive.py').write_text(upload,encoding='utf-8')
launch=f'''from pathlib import Path
import os,base64,json,tarfile,subprocess
out=Path('/data/libero_mem_baseline/analysis/all10_best_memory_tsne')
with tarfile.open(out/'lgd_memory_subsets.tar.gz') as tar:
 for m in tar.getmembers():
  target=(out/m.name).resolve()
  assert target.is_relative_to(out.resolve()) and m.isfile()
 tar.extractall(out,filter='data')
(out/'selected_checkpoints.json').write_bytes(base64.b64decode({selection!r}))
(out/'extract_all10_memory.py').write_bytes(base64.b64decode({script!r}))
env=os.environ.copy();env.update(PYTHONPATH='/home/nvidia/zyx/PhaseVLA/src:/home/nvidia/zyx/PhaseVLA',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0')
log=open(out/'extract.log','w')
p=subprocess.Popen(['/home/nvidia/zyx/PhaseVLA/environments/futuremamba/.venv/bin/python',str(out/'extract_all10_memory.py'),'--output',str(out),'--selection',str(out/'selected_checkpoints.json'),'--tasks','3,4,5,9'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
'''
(SESSION/'launch_remaining_four.py').write_text(launch,encoding='utf-8')
print('Archive bytes:',archive.stat().st_size)
