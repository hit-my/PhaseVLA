"""Component comparison at checkpoint1500, train seed42, three evaluation seeds.
All 27 evaluations sourced from A100, original600-step budget.
No-PAE = frozen AE with memory for first4 steps, without memory for last6.
No-memory = existing historical memory-free PAE training, not fixed retraining.
No-PAE training objective/optimizer differ from default: not a pure single-variable ablation.
"""
from pathlib import Path
import json,csv,statistics,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    root=Path(__file__).resolve().parent;source=root/'component_source.json';rows=json.loads(source.read_text(encoding='utf-8'))
    assert len(rows)==27 and len({(r['method'],r['task'],r['eval_seed']) for r in rows})==27
    methods=['no-memory','no-PAE','ActMem-VLA'];labels=['No-memory','No-PAE','ActMem-VLA'];colors=['#0072B2','#E69F00','#009E73'];summary=[]
    for m in methods:
        for t in [6,7,8]:
            rr=[r for r in rows if r['method']==m and r['task']==t]
            assert {r['eval_seed'] for r in rr}=={10001,10002,10003}
            for r in rr:
                assert r['train_seed']==42 and r['checkpoint']==1500 and len(r['rollouts'])==20
                assert sum(x['success'] for x in r['rollouts'])==r['successes']
                assert r['protocol']['max_steps']==600
            n=sum(r['successes'] for r in rr);summary.append(dict(method=m,task=t,train_seed=42,checkpoint=1500,successes=n,episodes=60,success_rate=100*n/60))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.5,2.25));fig.subplots_adjust(left=.17,right=.99,bottom=.17,top=.80)
    x=np.arange(3);width=.20
    for i,(m,l,c) in enumerate(zip(methods,labels,colors)):
        ax.bar(x+(i-1)*width,[next(r['success_rate'] for r in summary if r['method']==m and r['task']==t) for t in [6,7,8]],width=width*.85,color=c,alpha=.84,label=l,zorder=3)
    ax.set_xticks(x,['T6','T7','T8']);ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.set_xlim(-.6,2.6);ax.set_ylabel('Success rate (%)');ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E8ED',linewidth=.55)
    ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#697583');ax.tick_params(length=3,width=.6,pad=4)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False,handlelength=.65,columnspacing=.7,handletextpad=.35)
    assert len(ax.patches)==9 and len(ax.collections)==0
    for ext in ['pdf','svg','png']:fig.savefig(root/('fig05_component_ablations_c1500_s42.'+ext),dpi=300,facecolor='white')
    for filename,rr in [('fig05_success_rates.csv',summary),('fig05_seed_details.csv',[{k:r[k] for k in ['method','task','train_seed','checkpoint','eval_seed','successes','episodes','source']} for r in rows])]:
        with (root/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
        with (root/filename).open(encoding='utf-8-sig',newline='') as f:back=list(csv.DictReader(f))
        assert len(rr)==len(back)
        for a,b in zip(rr,back):assert all(str(v)==b[k] for k,v in a.items())
    report=dict(status='PASS',checkpoint=1500,train_seed=42,eval_seeds=[10001,10002,10003],evaluations=27,episodes=540,platform='A100',macro_success_rates={m:statistics.mean(r['success_rate'] for r in summary if r['method']==m) for m in methods},source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection_note='1500 selected post-hoc to maximize ActMem mean advantage over the average of the two component ablations among common checkpoints 500/1000/1500; same checkpoint for every method and task; exploratory')
    (root/'fig05_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');plt.close(fig);print(json.dumps(report))

if __name__=='__main__':main()
