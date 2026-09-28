"""Select actual corrected main-experiment checkpoints from audited evaluations."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'completed_experiments_source.json'
data=json.loads(source.read_text(encoding='utf-8'))
selected=[]
for task in range(1,11):
    candidates=[]
    for seed in [0,1,42]:
        for step in range(500,3001,500):
            rows=[r for r in data if r['model']=='ActMem-VLA' and r['task']==task and r['train_seed']==seed and r['checkpoint']==step]
            assert len(rows)==3 and {r['eval_seed'] for r in rows}=={10001,10002,10003}
            assert all(len(r['rollouts'])==20 and sum(x['success'] for x in r['rollouts'])==r['successes'] for r in rows)
            job=rows[0]['fullmatrix_job'];assert job['training_version']=='differentiable_recurrent_step_v1'
            count=sum(r['successes'] for r in rows)
            candidates.append(dict(task=task,seed=seed,step=step,sr=100*count/60,successes=count,episodes=60,
                                   job=job,evaluation_records=[r['id'] for r in rows]))
    best=max(candidates,key=lambda r:(r['successes'],-r['step'],r['job']['checkpoint_origin']=='A100',r['seed']==42,-r['seed']))
    selected.append(best)
(ROOT/'selected_checkpoints.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
(ROOT/'selection_protocol.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    rule='Max total successes across 3 evaluation seeds; ties: earliest step, then A100-local availability, then seed42, then lower seed.',
    caveat='Selection uses reported evaluation outcomes, not independent validation. For qualitative diagnostics, not an unbiased new performance estimate.'),indent=2))
for r in selected:print(f'T{r["task"]}: seed{r["seed"]} c{r["step"]}, {r["successes"]}/60, {r["job"]["checkpoint_origin"]}')
