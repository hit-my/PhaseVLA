"""Figure03: post-hoc success under demonstration-derived per-task step budgets.

T1-T6 budget = min(600, 20 * ceil(demonstration-length 95th percentile / 20)).
Linear-interpolated quantile. Same task budgets for all methods.
Uses Figure02's selected rollouts: no checkpoint reselection after budget change.
Not the original LIBERO-Mem success metric; no new rollouts are run.
ActMem retains the earlier test-set checkpoint selection bias.
"""
from pathlib import Path
import argparse,csv,json,math,statistics,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--font-size',type=float,default=10)
    args=p.parse_args()
    epfile=root.parent/'figure02_completion_steps/fig02_all_episodes.csv'
    demofile=root.parent/'figure02_completion_steps/demonstration_lengths_verified.json'
    with epfile.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
    demos=json.loads(demofile.read_text(encoding='utf-8'))['episodes']
    budgets=[];episodes=[];summaries=[];seedrows=[]
    methods=['pi05','MemoryVLA','ActMem-VLA']
    for t in range(1,7):
        lens=[x['length'] for x in demos if x['task']==t]
        q=float(np.quantile(lens,.95,method='linear'));b=min(600,20*math.ceil(q/20))
        budgets.append(dict(task=t,demos=len(lens),demo_p95=q,budget=b,demo_coverage=sum(x<=b for x in lens)/len(lens)))
        for m in methods:
            rr=[r for r in raw if r['method']==m and int(r['task'])==t]
            assert len(rr)==(180 if m=='ActMem-VLA' else 60)
            filtered=[]
            for r in rr:
                old=int(r['success']);step=int(r['steps']);new=int(bool(old) and step<=b)
                assert new<=old
                row={**r,'budget':b,'original_success':old,'budget_success':new,'excluded_by_budget':old-new}
                episodes.append(row);filtered.append(row)
            old=sum(r['original_success'] for r in filtered);new=sum(r['budget_success'] for r in filtered)
            summaries.append(dict(task=t,method=m,budget=b,episodes=len(rr),original_successes=old,budget_successes=new,original_sr=100*old/len(rr),budget_sr=100*new/len(rr),delta_pp=100*(new-old)/len(rr)))
            for seed in sorted({int(r['train_seed']) for r in rr}):
                for ev in [10001,10002,10003]:
                    a=[r for r in filtered if int(r['train_seed'])==seed and int(r['eval_seed'])==ev]
                    assert len(a)==20 and len({r['checkpoint'] for r in a})==1
                    seedrows.append(dict(task=t,method=m,train_seed=seed,eval_seed=ev,checkpoint=a[0]['checkpoint'],budget=b,episodes=20,original_successes=sum(r['original_success'] for r in a),budget_successes=sum(r['budget_success'] for r in a)))
    assert [x['budget'] for x in budgets]==[180,220,320,340,460,600]
    assert all(r['original_successes']==r['budget_successes'] for r in summaries if r['task']==6)
    for name,rows in [('fig03_budgets.csv',budgets),('fig03_success_rates.csv',summaries),('fig03_seed_details.csv',seedrows),('fig03_episode_audit.csv',episodes)]:
        with (root/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with (root/name).open(encoding='utf-8-sig',newline='') as f:readback=list(csv.DictReader(f))
        assert len(rows)==len(readback)
        for a,b in zip(rows,readback):assert all(str(v)==b[k] for k,v in a.items())
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':args.font_size,'axes.labelsize':args.font_size,'xtick.labelsize':args.font_size,'ytick.labelsize':args.font_size,'legend.fontsize':args.font_size,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.5,2.55));fig.subplots_adjust(left=.17,right=.99,bottom=.28,top=.82)
    x=np.arange(6);w=.19;colors=['#0072B2','#E69F00','#009E73'];labels=[r'$\pi_{0.5}$','MemoryVLA','ActMem-VLA']
    for i,(m,c,l) in enumerate(zip(methods,colors,labels)):
        a=[next(r['budget_sr'] for r in summaries if r['task']==t and r['method']==m) for t in range(1,7)]
        ax.bar(x+(i-1)*w,a,width=w*.78,color=c,alpha=.84,label=l,zorder=3)
    ax.set_xticks(x,[f"T{r['task']}\n({r['budget']})" for r in budgets]);ax.set_ylabel('Success rate (%)',labelpad=3);ax.set_xlabel('Task (step budget)',labelpad=3)
    ax.set_ylim(0,110);ax.set_yticks(range(0,101,20));ax.set_xlim(-.45,5.45);ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E8ED',linewidth=.55)
    ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#697583');ax.tick_params(length=3,width=.6,pad=3)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False,handlelength=.7,columnspacing=.7,handletextpad=.35)
    assert len(ax.patches)==18 and len(ax.collections)==0
    for text in [ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*fig.legends[0].get_texts()]:assert text.get_fontsize()==args.font_size
    for ext in ['pdf','svg','png']:fig.savefig(root/('fig03_task_budget_success.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    report=dict(status='PASS',policy_episodes=len(episodes),seed_evaluations=len(seedrows),budget_rule='min(600,20*ceil(linear_demo_length_p95/20))',budgets=budgets,checkpoint_selection='unchanged from Figure01/Figure02 test-selected checkpoints',no_new_rollouts=True,source_sha256=hashlib.sha256(epfile.read_bytes()).hexdigest(),demo_source_sha256=hashlib.sha256(demofile.read_bytes()).hexdigest(),macro_sr={m:statistics.mean(r['budget_sr'] for r in summaries if r['method']==m) for m in methods})
    (root/'fig03_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
