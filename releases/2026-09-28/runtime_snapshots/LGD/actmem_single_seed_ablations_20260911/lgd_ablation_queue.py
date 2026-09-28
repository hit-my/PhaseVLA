from pathlib import Path
import sys, os, json, time, subprocess, fcntl, hashlib
R=Path(__file__).parent
D=Path('/pfs/pfs-7jnepv/lgd/libero_multiseed_migration_20260908'); P=D/'PhaseVLA'
M=Path('/pfs/pfs-7jnepv/lgd/actmem_gradientfix_v1_20260910')
sys.path.insert(0,str(D/'ops'))
from target_training_worker import resolve_gpu_identity
lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
jobs=json.loads((R/'jobs.json').read_text())
state=json.loads((R/'status.json').read_text()) if (R/'status.json').exists() else {'jobs':{}}
state['pid']=os.getpid()
children={}
class AdoptedProcess:
    def __init__(self,j,pid):self.job=j;self.pid=pid
    def poll(self):
        p=Path('/proc')/str(self.pid)/'cmdline'
        try:cmd=p.read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:cmd=''
        if str(R/'jobs'/f"{self.job['id']}.json") in cmd:return None
        log=Path(self.job['log_dir'])/'worker.log'
        return 0 if log.exists() and 'accelerated_training_completed' in log.read_text(errors='replace') else 1
for j in jobs:
    s=state['jobs'].get(j['id'],{})
    if s.get('status')=='running':children[j['id']]=AdoptedProcess(j,s['pid'])
def save():
    state['time']=time.time(); p=R/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(R/'status.json')
def gpu_apps():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    return [(r.split(',')[0].strip(),int(r.split(',')[1])) for r in rows]
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
gpus={int(a):b.strip() for a,b in (r.split(',') for r in rows)}
allowed_external=json.loads((R/'shared_gpu_incumbents.json').read_text()) if (R/'shared_gpu_incumbents.json').exists() else {}
selected_gpus=[0,4,7,2,3,5,6] if allowed_external else [2,3,5,6]
maps={g:resolve_gpu_identity(gpus[g]) for g in selected_gpus}
expected=json.loads((R/'source_hashes.json').read_text())
while True:
    for jid,proc in list(children.items()):
        rc=proc.poll()
        if rc is None: continue
        j=next(j for j in jobs if j['id']==jid)
        cp=Path(j['checkpoint_root'])/'3000'
        good=rc==0 and all((cp/n).is_file() for n in ['plugin.safetensors','metadata.json','optimizer.pt','scheduler.pt','rng_state.pt'])
        state['jobs'][jid].update(status='completed' if good else 'failed_requires_review',returncode=rc,finished=time.time())
        del children[jid]
    apps=gpu_apps()
    for gpu in selected_gpus:
        own=[s['pid'] for s in state['jobs'].values() if s.get('gpu')==gpu and s['status']=='running']
        foreign=[pid for uuid,pid in apps if uuid==gpus[gpu] and pid not in own]
        unknown=[]
        for pid in foreign:
            expected_cmd=allowed_external.get(str(gpu),{}).get(str(pid))
            p=Path('/proc')/str(pid)/'cmdline'
            try:actual_cmd=p.read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError:continue
            if expected_cmd is None or expected_cmd!=actual_cmd:unknown.append(pid)
        if unknown:state.setdefault('blocked_gpus',{})[str(gpu)]=unknown;continue
        if len(own)+len(foreign)>=3:continue
        rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if int(rows.strip())<65000:continue
        pending=[j for j in jobs if j['id'] not in state['jobs'] and (j.get('preferred_gpu',gpu)==gpu)
                 and (j['task']!=6 or (R/'task6_cache_ready.json').exists()) and j.get('change')!={'kind':'pae_depth','value':6}]
        if not pending:continue
        j=pending[0]
        for path,digest in expected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,('source_changed',path)
        logs=Path(j['log_dir']);logs.mkdir(parents=True,exist_ok=False)
        path=R/'jobs'/f"{j['id']}.json";path.write_text(json.dumps(j,indent=2))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(maps[gpu]['cuda_ordinal']),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',WANDB_MODE='offline',JAX_PLATFORMS='cpu',PYTHONPATH=f'{M}:{D/"isolated_speedup"}:{P/"src"}:{P}')
        for key,sub in [('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor')]:
            q=logs/sub;q.mkdir();env[key]=str(q)
        if j['family']=='main':
            cmd=['/pfs/pfs-7jnepv/lgd/phasevla-runtime/bin/python',str(M/'lgd_corrected_main_entry.py'),'--job',str(path)]
        else:cmd=['/pfs/pfs-7jnepv/lgd/phasevla-runtime/bin/python',str(R/('pae2_cache_entry.py' if j.get('change')=={'kind':'pae_depth','value':2} else j.get('entry_script','lgd_ablation_entry.py'))),str(path)]
        with (logs/'worker.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[j['id']]=proc
        state['jobs'][j['id']]={'status':'running','gpu':gpu,'pid':proc.pid,'task':j['task'],'seed':j['seed'],'family':j['family'],'started':time.time()}
        save()
    save()
    if len(state['jobs'])==len(jobs) and not children:break
    time.sleep(15)
