"""Operational relocation of verified PhaseVLA recipes; model/trainer unchanged."""
from pathlib import Path
import sys,os,importlib.util,json
R=Path('/pfs/pfs-7jnepv/lgd/libero_fullmatrix_20260909')
D=Path('/pfs/pfs-7jnepv/lgd/libero_multiseed_migration_20260908')
spec=importlib.util.spec_from_file_location('migration_adapter',D/'ops/train_entry_lgd.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
a.D=R;a.MIGRATION_ID=R.name;a.ALLOWED_IDS=frozenset(j['id'] for j in json.loads((R/'jobs.json').read_text()))
a.CONDITIONING=R/'conditioning_cache';a.DATASET=D/'assets/lerobot/libero-mem/LIBERO-Mem-Lerobot'
if '--precompute' in sys.argv:
 import dataclasses
 jp=Path(sys.argv[sys.argv.index('--job')+1]);job,identity,_=a.load_job(jp,None)
 sys.path[:0]=[str(a.ISOLATED),str(a.P/'src'),str(a.P)]
 os.environ['OPENPI_DATA_HOME']=str(D/'assets/openpi')
 from openpi.training import config as tc
 original=tc.get_config;source,config=a.relocate_config(original(job['config']),job,None);a.verify_recipe(source,config,job,identity)
 tc.get_config=lambda name: config if name==job['config'] else original(name)
 from scripts import precompute_handoff04_libero_cache as cache
 import torch
 torch.set_num_threads(4)
 args=cache.parser().parse_args([job['config'],'--output-dir',str(a.CONDITIONING/job['config']),'--microbatch-size','16'])
 cache.precompute(args)
else:
 raise SystemExit(a.main())
