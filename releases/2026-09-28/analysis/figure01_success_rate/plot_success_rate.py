"""Figure 01: TEST-SELECTED checkpoint success rates; not held-out test performance.

ActMem: per task/train seed select max successes over six checkpoints, then
mean and SAMPLE SD over 3 train seeds. Ties choose the earliest checkpoint.
Baselines: fixed checkpoints, mean and SAMPLE SD over 3 evaluation seeds.
Only mean bars are displayed. No scatter points or error bars.
Replicates and descriptive SD are retained in companion CSV files.
No figure title or caption is drawn. Font sizes are in physical points;
insert PDF at its native width to preserve the requested text size.
"""
from pathlib import Path
import argparse,csv,json,statistics,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def compute(data):
    reps=[];means=[]
    for task in range(1,11):
        for method in ['pi05','MemoryVLA','ActMem-VLA']:
            if method=='ActMem-VLA':
                for seed in [0,1,42]:
                    candidates=[]
                    for cp in range(500,3001,500):
                        rows=[r for r in data if r['model']==method and r['task']==task and r['train_seed']==seed and r['checkpoint']==cp]
                        assert len(rows)==3 and {r['eval_seed'] for r in rows}=={10001,10002,10003}
                        for r in rows:assert len(r['rollouts'])==20 and sum(x['success'] for x in r['rollouts'])==r['successes']
                        candidates.append((sum(r['successes'] for r in rows),cp))
                    count,cp=max(candidates,key=lambda p:(p[0],-p[1]))
                    reps.append(dict(task=task,method=method,replicate_type='training_seed',replicate_seed=seed,checkpoint=cp,successes=count,episodes=60,sr=100*count/60))
            else:
                cp=49999 if method=='pi05' else 40000
                rows=[r for r in data if r['model']==method and r['task']==task and r['checkpoint']==cp]
                assert len(rows)==3 and {r['eval_seed'] for r in rows}=={10001,10002,10003}
                for r in sorted(rows,key=lambda r:r['eval_seed']):
                    assert len(r['rollouts'])==20 and sum(x['success'] for x in r['rollouts'])==r['successes']
                    reps.append(dict(task=task,method=method,replicate_type='evaluation_seed',replicate_seed=r['eval_seed'],checkpoint=cp,successes=r['successes'],episodes=20,sr=5*r['successes']))
            values=[r['sr'] for r in reps if r['task']==task and r['method']==method]
            assert len(values)==3
            means.append(dict(task=task,method=method,mean=statistics.mean(values),sd=statistics.stdev(values),n=3))
    return reps,means

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=here.parent/'completed_experiments_source.json')
    ap.add_argument('--output-dir',type=Path,default=here)
    ap.add_argument('--font-size',type=float,default=10,help='All visible font sizes in pt; match paper body text.')
    ap.add_argument('--width',type=float,default=7.16,help='Native figure width in inches (two-column default).')
    ap.add_argument('--height',type=float,default=2.65)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.input.read_text(encoding='utf-8'));reps,summary=compute(data)
    # Okabe-Ito palette: blue, orange, bluish green; colorblind-friendly.
    methods=['pi05','MemoryVLA','ActMem-VLA'];colors=['#0072B2','#E69F00','#009E73']
    labels=[r'$\pi_{0.5}$','MemoryVLA','ActMem-VLA']
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':args.font_size,'axes.labelsize':args.font_size,'xtick.labelsize':args.font_size,'ytick.labelsize':args.font_size,'legend.fontsize':args.font_size,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65,'hatch.linewidth':.5})
    fig,ax=plt.subplots(figsize=(args.width,args.height))
    # Explicit margins keep native dimensions; tight cropping would change size.
    fig.subplots_adjust(left=.078,right=.992,bottom=.16,top=.80)
    x=np.arange(10);barw=.235
    for i,(method,c,label) in enumerate(zip(methods,colors,labels)):
        rr=[next(r for r in summary if r['task']==t and r['method']==method) for t in range(1,11)]
        positions=x+(i-1)*barw
        ax.bar(positions,[r['mean'] for r in rr],barw*.92,color=c,alpha=.84,label=label,zorder=3,linewidth=0)
    assert len(ax.collections)==0 and len(ax.patches)==30
    ax.set_xticks(x,[f'T{i}' for i in range(1,11)])
    ax.set_ylabel('Success rate (%)')
    # Leave visual headroom above 100% bars.
    ax.set_ylim(0,112);ax.set_yticks(np.arange(0,101,20));ax.set_xlim(-.6,9.6)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E8ED',linewidth=.55)
    ax.tick_params(axis='both',length=3,width=.6,pad=4)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#697583')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.535,.995),ncol=3,frameon=False,handlelength=1.25,columnspacing=2,handletextpad=.55)
    stem='fig01_success_rate_bars_only'
    for ext in ['pdf','svg','png']:fig.savefig(args.output_dir/f'{stem}.{ext}',dpi=300,facecolor='white')
    for filename,rr in [('fig01_replicates.csv',reps),('fig01_mean_sd.csv',summary)]:
        with (args.output_dir/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    avg=statistics.mean(r['mean'] for r in summary if r['method']=='ActMem-VLA')
    assert abs(avg-87.33333333333333)<1e-8
    assert all(abs(text.get_fontsize()-args.font_size)<1e-10 for text in [ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*fig.legends[0].get_texts()])
    meta=dict(selection='per-task per-training-seed best checkpoint on the same evaluation set; optimistic descriptive result',error_bars='none',seed_markers='none',actmem_macro_mean=avg,font_size_pt=args.font_size,width_inches=args.width,height_inches=args.height,source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    (args.output_dir/'fig01_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    plt.close(fig);print(json.dumps(meta))

if __name__=='__main__':main()
