from pathlib import Path
import sys,dataclasses,json
R=Path(__file__).parent;B=Path('/data/libero_mem_baseline');P=Path('/home/nvidia/zyx/PhaseVLA');sys.path[:0]=[str(B/'isolated_speedup'),str(P/'src'),str(P)]
from openpi.training import config as tc
get=tc.get_config
def config(name):
 c=get(name);return dataclasses.replace(c,model=dataclasses.replace(c.model,progress_depth=6,progress_layer_mapping=None))
tc.get_config=config
assert tuple(config('futuremamba_action_history_handoff04_libero_mem_bowl_t6').model.resolved_progress_layer_indices)==(0,3,7,10,14,17)
import precompute_all10_conditioning_cache as cache
cache.OUTPUT_ROOT=R/'conditioning_cache_pae6'
sys.argv=[cache.__file__,'--task-indices','6','7','8','--microbatch-size','8']
raise SystemExit(cache.main())
