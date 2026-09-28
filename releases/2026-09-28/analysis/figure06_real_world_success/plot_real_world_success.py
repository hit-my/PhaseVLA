"""Real-robot success rates: user-reported counts, 20 trials per task/method.
Run: python plot_real_world_success.py
PDF/SVG/PNG are written beside this script. No simulated data is used.
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT=Path(__file__).resolve().parent
LABELS=['R1','R2','R3','R4']
COLORS=['#0072B2','#009E73']
FONT=10*72/72.27  # Match 10 TeX pt body text in physical units.

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width-mm',type=float,default=88.9)
    args=parser.parse_args()
    source=ROOT/'real_world_results.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert [r['task_id'] for r in rows]==['R1','R2','R3','R4']
    records=[]
    for r in rows:
        n=int(r['trials_per_method']);assert n==20
        for method,key in [('pi05','pi05_successes'),('ActMem-VLA','actmem_successes')]:
            successes=int(r[key]);assert 0<=successes<=n
            records.append(dict(task_id=r['task_id'],method=method,successes=successes,
                                trials=n,success_rate=100*successes/n))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
       'font.size':FONT,'axes.labelsize':FONT,'xtick.labelsize':FONT,'ytick.labelsize':FONT,
       'legend.fontsize':FONT,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
       'svg.fonttype':'none','axes.linewidth':.65})
    width=args.width_mm/25.4
    fig,ax=plt.subplots(figsize=(width,2.38))
    fig.subplots_adjust(left=.155,right=.994,bottom=.235,top=.83)
    x=np.arange(4);bar_offset=.23;annotations=[]
    for i,(method,label,color) in enumerate(zip(['pi05','ActMem-VLA'],[r'$\pi_{0.5}$','ActMem-VLA'],COLORS)):
        values=[r['success_rate'] for r in records if r['method']==method]
        bars=ax.bar(x+(2*i-1)*bar_offset,values,width=.34,color=color,alpha=.84,label=label,zorder=3)
        annotations.extend(ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=FONT,color='#333333'))
    ax.set_xticks(x,LABELS)
    ax.set_xlim(-.6,3.6);ax.set_ylim(0,115);ax.set_yticks([0,20,40,60,80,100])
    ax.set_ylabel('Success rate (%)',labelpad=3)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E8ED',linewidth=.55)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#697583')
    ax.tick_params(length=3,width=.6,pad=3)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.995),
               ncol=2,frameon=False,handlelength=.8,columnspacing=1.4,handletextpad=.4,borderaxespad=0)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for texts in [annotations,ax.get_xticklabels()]:
        boxes=[t.get_window_extent(renderer) for t in texts]
        assert all(not a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]),'Overlapping labels'
    tight=fig.get_tightbbox(renderer)
    assert tight.x0>=0 and tight.x1<=width,('Horizontal overflow',tight.bounds)
    # Tight vertical crop, fixed physical column width: no post-export scaling.
    bounds=Bbox.from_extents(0,tight.y0-.01,width,tight.y1+.01)
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/f'fig06_real_world_success.{ext}',dpi=300,facecolor='white',bbox_inches=bounds,pad_inches=0)
    plt.close(fig)
    with (ROOT/'real_world_success_rates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report={'status':'PASS','data_origin':'User supplied task-order percentages, converted exactly to counts out of 20',
       'width_mm':args.width_mm,'body_font_tex_pt':10,'font_pdf_pt':FONT,'colors':dict(zip(['pi05','ActMem-VLA'],COLORS)),
       'task_count':4,'trials_per_task_per_method':20,'total_trials':160,'labeled_bars':8,
       'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'macro_success_rates':{m:float(np.mean([r['success_rate'] for r in records if r['method']==m])) for m in ['pi05','ActMem-VLA']}}
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
