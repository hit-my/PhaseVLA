"""Figure04: parameter ablations at one shared checkpoint, training seed 42.
Each bar averages evaluation seeds 10001,10002,10003 (60 episodes).
No test-based per-variant checkpoint selection. No error bars or titles.
Panels: handoff ratio r; Mamba width d_m / depth L_m; PAE depth P.
Green denotes the same default in all panels: r=.4, d_m=1024,L_m=2,P=4.
"""
from pathlib import Path
import argparse,json,csv,hashlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=int,default=3000,choices=[500,1000,1500,2000,2500,3000])
    p.add_argument('--font-size',type=float,default=10)
    args=p.parse_args();source=root.parent/'completed_experiments_source.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    panels=[
        [('handover03',r'$r=0.3$','#0072B2'),('ActMem-VLA',r'$r=0.4$','#009E73'),('handover05',r'$r=0.5$','#E69F00'),('handover06',r'$r=0.6$','#CC79A7')],
        [('ActMem-VLA',r'$d_m=1024,\ L_m=2$','#009E73'),('mamba_depth4',r'$d_m=1024,\ L_m=4$','#0072B2'),('mamba_width1536',r'$d_m=1536,\ L_m=2$','#E69F00')],
        [('pae_depth2',r'$P=2$','#0072B2'),('ActMem-VLA',r'$P=4$','#009E73'),('pae_depth6',r'$P=6$','#E69F00')]]
    rows=[];summaries=[];lookup={}
    for m in dict.fromkeys(m for panel in panels for m,_,_ in panel):
        for t in [6,7,8]:
            rr=[r for r in data if r['model']==m and r['task']==t and r['train_seed']==42 and r['checkpoint']==args.checkpoint]
            assert len(rr)==3 and {r['eval_seed'] for r in rr}=={10001,10002,10003},(m,t)
            for r in sorted(rr,key=lambda r:r['eval_seed']):
                assert len(r['rollouts'])==20 and sum(x['success'] for x in r['rollouts'])==r['successes']
                rows.append(dict(method=m,task=t,train_seed=42,checkpoint=args.checkpoint,eval_seed=r['eval_seed'],successes=r['successes'],episodes=20,success_rate=5*r['successes'],source=r['path']))
            count=sum(r['successes'] for r in rr);rate=100*count/60
            summaries.append(dict(method=m,task=t,train_seed=42,checkpoint=args.checkpoint,successes=count,episodes=60,success_rate=rate));lookup[m,t]=rate
    assert len(rows)==72 and len(summaries)==24
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':args.font_size,'axes.labelsize':args.font_size,'xtick.labelsize':args.font_size,'ytick.labelsize':args.font_size,'legend.fontsize':args.font_size,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65})
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.05),sharey=True)
    fig.subplots_adjust(left=.082,right=.993,bottom=.15,top=.70,wspace=.20)
    for ax,panel in zip(axes,panels):
        width=.78/len(panel);x=np.arange(3)
        for j,(method,label,color) in enumerate(panel):
            ax.bar(x+(j-(len(panel)-1)/2)*width,[lookup[method,t] for t in [6,7,8]],width=width*.92,color=color,alpha=.84,label=label,zorder=3)
        ax.set_xticks(x,['T6','T7','T8']);ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.set_xlim(-.58,2.58)
        ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E8ED',linewidth=.55);ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#697583');ax.tick_params(length=3,width=.6,pad=4)
        legend=ax.legend(loc='lower center',bbox_to_anchor=(.5,1.04),ncol=2 if len(panel)==4 else 1,frameon=False,handlelength=1,columnspacing=.8,handletextpad=.45,labelspacing=.3,borderaxespad=0)
        for text in [*ax.get_xticklabels(),*ax.get_yticklabels(),*legend.get_texts()]:assert text.get_fontsize()==args.font_size
    axes[0].set_ylabel('Success rate (%)')
    assert sum(len(ax.patches) for ax in axes)==30 and all(len(ax.collections)==0 for ax in axes)
    stem=f'fig04_parameter_ablations_c{args.checkpoint}'
    for ext in ['pdf','svg','png']:fig.savefig(root/(stem+'.'+ext),dpi=300,facecolor='white')
    for suffix,records in [('seed_details',rows),('success_rates',summaries)]:
        path=root/(stem+'_'+suffix+'.csv')
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        with path.open(encoding='utf-8-sig',newline='') as f:back=list(csv.DictReader(f))
        for r,b in zip(records,back):assert all(str(v)==b[k] for k,v in r.items())
        assert len(records)==len(back)
    meta=dict(status='PASS',checkpoint=args.checkpoint,training_seed=42,evaluation_seeds=[10001,10002,10003],episodes_per_bar=60,unique_evaluations=72,unique_episodes=1440,default_reused_across_panels=True,font_size_pt=args.font_size,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),mean_success_rate={m:statistics.mean(lookup[m,t] for t in [6,7,8]) for m in dict.fromkeys(m for panel in panels for m,_,_ in panel)})
    (root/(stem+'_validation.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8');plt.close(fig);print(json.dumps(meta))

if __name__=='__main__':main()
