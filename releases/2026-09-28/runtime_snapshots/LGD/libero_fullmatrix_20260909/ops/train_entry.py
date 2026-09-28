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
 os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
 from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
 action_root=D/'assets/action_cache_handoff04_all10'/job['config']
 entries=json.loads((action_root/'manifest.json').read_text())['entries']
 selected_ids=sorted(int(e['episode_id']) for e in entries)
 id_to_ordinal={eid:i for i,eid in enumerate(selected_ids)}
 class TaskDataset(LeRobotDataset):
  def _get_query_indices(self,idx,ep_idx):
   return super()._get_query_indices(idx,id_to_ordinal[ep_idx])
 original_create=cache._episode_loader.create_lerobot_episode_dataset
 def create_task_dataset(**kwargs):
  result=original_create(**kwargs,dataset_factory=lambda *args,**kw:TaskDataset(*args,**kw,episodes=selected_ids),dataset_metadata_factory=LeRobotDatasetMetadata)
  assert len(result.episodes)==len(selected_ids),(len(result.episodes),len(selected_ids))
  records=[]
  for rec in result.episodes:
   eid=selected_ids[rec.episode_index]
   actual=int(result._dataset.hf_dataset[rec.start_frame]['episode_index'].item())
   assert actual==eid,(actual,eid)
   records.append(dataclasses.replace(rec,episode_index=eid,queries=tuple(dataclasses.replace(q,episode_index=eid) for q in rec.queries)))
  result.episodes=tuple(records)
  original_episode_at=result.episode_at
  def checked_episode_at(index):
   episode=original_episode_at(index)
   expected=torch.load(action_root/f'episode_{episode.episode_index:06d}.pt',map_location='cpu',weights_only=True)
   for key in ['actions','action_mask','executed_actions','executed_action_mask']:
    actual=torch.as_tensor(getattr(episode,key));target=torch.as_tensor(expected[key])
    assert actual.shape==target.shape,(episode.episode_index,key,actual.shape,target.shape)
    assert torch.allclose(actual.float(),target.float(),atol=1e-6,rtol=1e-6),(episode.episode_index,key,'action cache alignment mismatch')
   return episode
  result.episode_at=checked_episode_at
  print(json.dumps({'event':'offline_task_dataset_verified','episodes':len(records),'original_episode_ids':selected_ids}),flush=True)
  return result
 cache._episode_loader.create_lerobot_episode_dataset=create_task_dataset
 args=cache.parser().parse_args([job['config'],'--output-dir',str(a.CONDITIONING/job['config']),'--microbatch-size','16'])
 cache.precompute(args)
else:
 raise SystemExit(a.main())
