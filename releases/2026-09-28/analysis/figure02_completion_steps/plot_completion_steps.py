"""Figure02: successful rollout steps versus FULL demonstration trajectory lengths.

ActMem checkpoints: same per-task/per-training-seed test-set selection as Figure01.
Only successes are plotted for policies. All 961 dataset episodes are included
as demonstration-length references, NOT independently verified first-success times.
No title/caption is drawn. Plot physical fonts are 10pt at native width.
"""
from pathlib import Path
import argparse,json,csv,statistics,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def extract(data, demos):
    selected=[];samples=[];summaries=[]
    for t in range(1,11):
        for seed in [0,1,42]:
            candidates=[]
            for cp in range(500,3001,500):
                rr=[r for r in data if r['model']=='ActMem-VLA' and r['task']==t and r['train_seed']==seed and r['checkpoint']==cp]
                assert len(rr)==3 and {r['eval_seed'] for r in rr}=={10001,10002,10003}
                candidates.append((sum(r['successes'] for r in rr),cp,rr))
            _,_,rr=max(candidates,key=lambda x:(x[0],-x[1]));selected.extend(rr)
    selected.extend(r for r in data if (r['model'],r['checkpoint']) in [('pi05',49999),('MemoryVLA',40000)])
    assert len(selected)==150
    for r in selected:
        assert len(r['rollouts'])==20 and sum(x['success'] for x in r['rollouts'])==r['successes']
        for x in r['rollouts']:
            samples.append(dict(method=r['model'],task=r['task'],train_seed=r['train_seed'],eval_seed=r['eval_seed'],checkpoint=r['checkpoint'],episode=x['episode'],success=int(x['success']),steps=x['steps'],included_in_box=int(x['success']),source=r['path']))
    assert len(demos)==961 and sum(d['length'] for d in demos)==321405
    for d in demos:samples.append(dict(method='Demonstrations',task=d['task'],train_seed='',eval_seed='',checkpoint='',episode=d['episode'],success='',steps=d['length'],included_in_box=1,source=d['source']))
    for t in range(1,11):
        for m in ['Demonstrations','pi05','MemoryVLA','ActMem-VLA']:
            allr=[x for x in samples if x['task']==t and x['method']==m];a=np.array([x['steps'] for x in allr if x['included_in_box']],dtype=float)
            if len(a):
                q1,med,q3=np.quantile(a,[.25,.5,.75]);iqr=q3-q1;inside=a[(a>=q1-1.5*iqr)&(a<=q3+1.5*iqr)]
                stats=dict(mean=float(a.mean()),q1=q1,median=med,q3=q3,whisker_low=float(inside.min()),whisker_high=float(inside.max()),outlier_count=int(len(a)-len(inside)))
            else:stats={k:'' for k in ['mean','q1','median','q3','whisker_low','whisker_high','outlier_count']}
            summaries.append(dict(task=t,method=m,total_episodes=len(allr),box_samples=len(a),success_rate='' if m=='Demonstrations' else 100*len(a)/len(allr),**stats))
    return samples,summaries,selected

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=root.parent/'completed_experiments_source.json')
    p.add_argument('--demos',type=Path,default=root/'demonstration_lengths_verified.json')
    p.add_argument('--font-size',type=float,default=10)
    p.add_argument('--width',type=float,default=3.5)
    p.add_argument('--height',type=float,default=3.05)
    args=p.parse_args();data=json.loads(args.input.read_text(encoding='utf-8'));demos=json.loads(args.demos.read_text(encoding='utf-8'))
    samples,summary,selected=extract(data,demos['episodes'])
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':args.font_size,'axes.labelsize':args.font_size,'xtick.labelsize':args.font_size,'ytick.labelsize':args.font_size,'legend.fontsize':args.font_size,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65})
    fig,axes=plt.subplots(3,2,figsize=(args.width,args.height),sharey=False)
    fig.subplots_adjust(left=.18,right=.985,bottom=.10,top=.80,hspace=.38,wspace=.45)
    methods=['Demonstrations','pi05','MemoryVLA','ActMem-VLA']
    colors=['#9AA4AE','#0072B2','#E69F00','#009E73']
    labels=['Demonstrations',r'$\pi_{0.5}$','MemoryVLA','ActMem-VLA']
    missing=[]
    for t,ax in enumerate(axes.flat,1):
        for j,(method,color) in enumerate(zip(methods,colors),1):
            a=[x['steps'] for x in samples if x['task']==t and x['method']==method and x['included_in_box']]
            if not a:
                missing.append(dict(task=t,method=method))
                ax.text(j,.07,'N/A',transform=ax.get_xaxis_transform(),color=color,ha='center',va='bottom',fontsize=args.font_size)
                continue
            ax.boxplot([a],positions=[j],widths=.56,patch_artist=True,manage_ticks=False,whis=1.5,showfliers=False,
                boxprops=dict(facecolor=color,edgecolor=color,alpha=.65,linewidth=.75),medianprops=dict(color='#26323D',linewidth=.85),whiskerprops=dict(color=color,linewidth=.75),capprops=dict(color=color,linewidth=.75),flierprops=dict(marker='o',markersize=1.7,markeredgewidth=.4,markeredgecolor=color,markerfacecolor='white',alpha=.7))
        ax.set_xlim(.4,4.6);ax.set_xticks([]);ax.set_xlabel(f'T{t}',labelpad=3)
        values=[float(r[k]) for r in summary if r['task']==t and r['box_samples'] for k in ['whisker_low','whisker_high']]
        low,high=min(values),max(values)
        padding=max(15,(high-low)*.08)
        bottom=max(0,50*np.floor((low-padding)/50))
        top=50*np.ceil((high+padding)/50)
        ax.set_ylim(bottom,top)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3,steps=[1,2,2.5,5,10],integer=True))
        ax.set_axisbelow(True);ax.grid(axis='y',color='#E2E7ED',linewidth=.55)
        ax.tick_params(length=2.5,width=.6,pad=3)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#697583')
    fig.text(.025,.48,'Environment steps',rotation=90,va='center',ha='center',fontsize=args.font_size)
    fig.legend(handles=[Patch(facecolor=c,edgecolor=c,alpha=.65,label=l) for c,l in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,frameon=False,handlelength=.9,columnspacing=.8,handletextpad=.4)
    for ax in axes.flat:
        assert all(text.get_fontsize()==args.font_size for text in [ax.xaxis.label,*ax.get_yticklabels()])
    stem='fig02_completion_steps_boxplot'
    for ext in ['pdf','svg','png']:fig.savefig(root/(stem+'.'+ext),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.01)
    for name,rows in [('fig02_all_episodes.csv',samples),('fig02_distribution_summary.csv',summary)]:
        with (root/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    chosen=[dict(task=r['task'],train_seed=r['train_seed'],checkpoint=r['checkpoint'],eval_seed=r['eval_seed'],id=r['id']) for r in selected]
    (root/'fig02_selected_evaluations.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    # Re-read exported samples: exact integer steps, flags and identities.
    with (root/'fig02_all_episodes.csv').open(encoding='utf-8-sig',newline='') as f:back=list(csv.DictReader(f))
    assert len(back)==3961
    for a,b in zip(samples,back):assert all(str(v)==b[k] for k,v in a.items())
    assert missing==[dict(task=6,method='MemoryVLA')]
    fig01=root.parent/'figure01_success_rate/fig01_replicates.csv'
    if fig01.exists():
        with fig01.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        for a in rr:
            if a['method']=='ActMem-VLA':assert {r['checkpoint'] for r in selected if r['model']=='ActMem-VLA' and r['task']==int(a['task']) and r['train_seed']==int(a['replicate_seed'])}=={int(a['checkpoint'])}
    report=dict(status='PASS',policy_episodes=3000,policy_successes=sum(x['success'] for x in samples if x['method']!='Demonstrations'),demo_episodes=961,demo_frames=321405,missing_boxes=missing,units='environment steps; demonstrations are full trajectory lengths, not first-success time',selection='same test-selected checkpoints as Figure01',font_size_pt=args.font_size,layout='3 rows x 2 columns; row-major T1-T6',width_inches=args.width,height_inches=args.height,axis_scaling='independent task limits from whisker extents with padding; outlier markers hidden, all data retained in box statistics',task_axis_limits={str(t):list(ax.get_ylim()) for t,ax in enumerate(axes.flat,1)},source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    (root/'fig02_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');plt.close(fig);print(json.dumps(report))

if __name__=='__main__':main()
