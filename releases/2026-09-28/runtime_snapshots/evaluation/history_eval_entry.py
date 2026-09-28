from pathlib import Path
import sys,json,os,runpy
R=Path(__file__).parent;j=json.loads(Path(sys.argv[sys.argv.index('--job')+1]).read_text())
os.environ['ACTMEM_DIAGNOSTIC_HISTORY_WINDOW']=str(j['history_window'])
try:runpy.run_path(str(R/'ops/eval_entry.py'),run_name='__main__')
except SystemExit as e:
 if e.code not in [None,0]:raise
p=R/'evaluations'/j['id']/'audited_summary.json';d=json.loads(p.read_text());d['protocol']['history_window_intervention']=j['history_window'];d['protocol']['full_state_retained_for_bookkeeping']=True;d['interpretation']='Inference stress test; not a retrained memory ablation or proof of generalization';p.write_text(json.dumps(d,indent=2))
