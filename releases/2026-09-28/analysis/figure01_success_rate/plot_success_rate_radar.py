"""Ten-task radar plot of test-selected success rates, not held-out selection.
Uses the same audited data and checkpoint selection as plot_success_rate.py.
No title/caption, no error bars. Native-size fonts default to 10 pt.
"""
from pathlib import Path
import argparse,json,csv,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_success_rate import compute

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=root.parent/'completed_experiments_source.json')
    parser.add_argument('--font-size',type=float,default=10)
    parser.add_argument('--size',type=float,default=4.5)
    args=parser.parse_args()
    _,summary=compute(json.loads(args.input.read_text(encoding='utf-8')))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':args.font_size,'axes.labelsize':args.font_size,'xtick.labelsize':args.font_size,'ytick.labelsize':args.font_size,'legend.fontsize':args.font_size,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(args.size,args.size))
    ax=fig.add_axes([.12,.10,.76,.76],projection='polar')
    ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
    angles=np.linspace(0,2*np.pi,10,endpoint=False);closed=np.r_[angles,angles[0]]
    methods=['pi05','MemoryVLA','ActMem-VLA'];colors=['#0072B2','#E69F00','#009E73']
    labels=[r'$\pi_{0.5}$','MemoryVLA','ActMem-VLA']
    styles=['--',':','-'];lines=[]
    for method,color,label,style in zip(methods,colors,labels,styles):
        values=np.array([next(r['mean'] for r in summary if r['task']==t and r['method']==method) for t in range(1,11)])
        assert len(values)==10 and np.all((values>=0)&(values<=100))
        line,=ax.plot(closed,np.r_[values,values[0]],color=color,lw=1.5,ls=style,label=label,zorder=3)
        ax.fill(closed,np.r_[values,values[0]],color=color,alpha=.07,zorder=2)
        lines.append(line)
    ax.set_xticks(angles,[f'T{i}' for i in range(1,11)])
    ax.tick_params(axis='x',pad=7)
    ax.set_ylim(0,105);ax.set_yticks([20,40,60,80,100],['20%','40%','60%','80%','100%'])
    ax.set_rlabel_position(18)
    ax.tick_params(axis='y',labelsize=args.font_size,colors='#637181')
    ax.grid(color='#DDE3E9',linewidth=.6)
    ax.spines['polar'].set_visible(False)
    fig.legend(handles=lines,loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,handlelength=1.6,columnspacing=1.1,handletextpad=.4)
    for text in [*ax.get_xticklabels(),*ax.get_yticklabels(),*fig.legends[0].get_texts()]:assert text.get_fontsize()==args.font_size
    stem=root/'fig01_success_rate_radar'
    for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.01)
    with (root/'fig01_radar_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task','method','mean']);w.writeheader();w.writerows({k:r[k] for k in ['task','method','mean']} for r in summary)
    meta=dict(selection='same test-selected per-task per-training-seed checkpoint means as bars-only figure',actmem_macro_mean=statistics.mean(r['mean'] for r in summary if r['method']=='ActMem-VLA'),font_size_pt=args.font_size,size_inches=args.size,note='Radar area depends on task ordering and is not an aggregate performance score.')
    assert abs(meta['actmem_macro_mean']-87.33333333333333)<1e-8
    (root/'fig01_radar_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    plt.close(fig);print(json.dumps(meta))

if __name__=='__main__':main()
