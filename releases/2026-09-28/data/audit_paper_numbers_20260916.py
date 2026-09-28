"""Check numerical claims in the manuscript pasted on 2026-09-16."""
from pathlib import Path
import csv,json,statistics,math
R=Path(__file__).resolve().parent
def read(path):
    with (R/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rates=read('figure03_task_budget_double_column_with_average/fig03_success_rates.csv')
means={m:statistics.mean(float(r['budget_sr']) for r in rates if r['method']==m) for m in ['pi05','MemoryVLA','ActMem-VLA']}
for r in rates:assert math.isclose(float(r['budget_sr']),100*int(r['budget_successes'])/int(r['episodes']))
winners={t:[r['method'] for r in rates if int(r['task'])==t and math.isclose(float(r['budget_sr']),max(float(a['budget_sr']) for a in rates if int(a['task'])==t))] for t in range(1,11)}
components=read('figure05_component_ablations/fig05_success_rates.csv')
expected={'no-memory':[38.3,33.3,50.0,40.6],'no-PAE':[58.3,38.3,63.3,53.3],'ActMem-VLA':[80.0,40.0,45.0,55.0]}
for m,values in expected.items():
    a=[float(next(r['success_rate'] for r in components if r['method']==m and int(r['task'])==t)) for t in [6,7,8]]
    assert [round(x,1) for x in a+[statistics.mean(a)]]==values
param=read('figure04_parameter_ablations/table04_parameter_ablations.csv')
expected_params=[[80,40,45,55],[65,36.7,43.3,48.3],[66.7,25,43.3,45],[71.7,28.3,41.7,47.2],[58.3,25,46.7,43.3],[76.7,40,50,55.6],[70,21.7,40,43.9],[71.7,35,55,53.9]]
for row,values in zip(param,expected_params):
    assert [round(float(row[k]),1) for k in ['T6','T7','T8','Average']]==values
    assert math.isclose(float(row['Average']),statistics.mean(float(row[k]) for k in ['T6','T7','T8']))
real=read('figure06_real_world_success/real_world_results.csv')
totals={m:sum(int(r[m+'_successes']) for r in real) for m in ['pi05','actmem']}
trials=sum(int(r['trials_per_method']) for r in real)
raw=read('figure03_task_budget_double_column/fig02_all_episodes.csv')
protocol={}
for m in means:
    rows=[r for r in raw if r['method']==m]
    protocol[m]={'train_seeds':sorted({int(r['train_seed']) for r in rows}),'eval_seeds':sorted({int(r['eval_seed']) for r in rows}),'per_task_trials':{str(t):sum(int(r['task'])==t for r in rows) for t in range(1,11)}}
out={'scope':'Arithmetic and consistency audit against saved local data; not a new independent audit of raw real-robot trials or hardware.', 'simulation_macro_sr':means,'simulation_gain_pp':{m:means['ActMem-VLA']-means[m] for m in ['pi05','MemoryVLA']},'task_winners':winners,'component_table_all_values_match':True,'hyperparameter_table_all_values_match':True,'real_success_counts':totals,'real_trials_per_method':trials,'real_gain_pp':100*(totals['actmem']-totals['pi05'])/trials,'real_relative_gain_percent':100*(totals['actmem']-totals['pi05'])/totals['pi05'],'additional_parameters':115700448,'frozen_base_parameters':3353433872,'additional_parameter_percent':100*115700448/3353433872,'actual_main_protocol':protocol,'task_budgets':[int(r['budget']) for r in read('figure03_task_budget_double_column_with_average/fig03_budgets.csv')]}
(R/'paper_numbers_audit_20260916.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
