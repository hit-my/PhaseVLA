"""Recolor unchanged sphere coordinates by reviewed, equal-weight task milestones."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import proj3d
ROOT=Path(__file__).resolve().parent
SPHERE=ROOT.parent/'figure15_memory_spherical_trajectories'
SOURCE=ROOT.parent/'figure12_all_tasks_memory_tsne'
sys.path.insert(0,str(SPHERE))
from plot_memory_sphere import arc,NAMES
CMAP=LinearSegmentedColormap.from_list('milestone_green',['#ccece6','#66c2a4','#238b45','#00441b'])
EXPECTED=[1,1,3,3,5,7,3,4,2,2]
EXCLUSIONS={(4,1):[1],(10,4):[2]}

def main():
    (ROOT/'individual').mkdir(exist_ok=True);(ROOT/'coordinates').mkdir(exist_ok=True)
    rows=json.loads((ROOT/'milestone_candidates.json').read_text())
    manifest=json.loads((SOURCE/'extraction_manifest.json').read_text())
    web=[];report=[];csvrows=[]
    for task in range(1,11):
        with np.load(SPHERE/'coordinates'/f'T{task:02d}_sphere.npz') as f:d={k:f[k] for k in f.files}
        src=next(r for r in manifest['tasks'] if r['task']==task)
        demos=[];all_progress=np.zeros(len(d['unit']))
        for e in range(5):
            row=next(r for r in rows if r['task']==task and r['episode']==e)
            assert row['id']==src['episodes'][e]['id'] and row['actions_sha256']==src['episodes'][e]['actions_sha256']
            excluded=EXCLUSIONS.get((task,e),[])
            cycles=[c for c in row['cycles'] if c['cycle'] not in excluded]
            assert len(cycles)==EXPECTED[task-1]
            events=[]
            for i,c in enumerate(cycles,1):
                # G: observed lifted-object reference frame. P: query after release,
                # with resulting placement checked in the settled evidence frame.
                events.append(dict(label=f'G{i}',query=c['peak_observation'],description=f'第{i}次抓取（抬起参考帧）',kind='grasp',cycle=i))
                events.append(dict(label=f'P{i}',query=c['release_action']+1,description=f'第{i}次放置',kind='place',cycle=i))
            assert all(a['query']<b['query'] for a,b in zip(events[:-1],events[1:]))
            ids=np.flatnonzero(d['episode']==e);q=d['query'][ids];u=d['unit'][ids]
            progress=np.searchsorted([v['query'] for v in events],q,side='right')/len(events)
            all_progress[ids]=progress
            stages=[]
            for count in np.rint(progress*len(events)).astype(int):
                stages.append('初始接近' if count==0 else events[count-1]['description'])
            for j,event in enumerate(events):
                event['progress']=(j+1)/len(events)
                valid=np.flatnonzero(q>=event['query'])
                event['point_index']=int(valid[0]) if len(valid) else None
                event['sample_query']=int(q[valid[0]]) if len(valid) else None
                csvrows.append(dict(task=task,episode=row['id'],**event))
            demo=dict(id=row['id'],points=u.tolist(),arcs=[arc(a,b).tolist() for a,b in zip(u[:-1],u[1:])],
                progress=progress.tolist(),query=q.tolist(),colors=[matplotlib.colors.to_hex(CMAP(float(p))) for p in progress],
                stages=stages,events=events,excluded_command_cycles=excluded)
            demos.append(demo)
            report.append(dict(task=task,episode=row['id'],excluded_command_cycles=excluded,required_milestones=len(events),
                final_sample_progress=float(progress[-1]),events=events,actions_sha256=row['actions_sha256']))
        np.savez_compressed(ROOT/'coordinates'/f'T{task:02d}_progress.npz',unit=d['unit'],progress=all_progress,query=d['query'],episode=d['episode'])
        assert np.array_equal(np.load(ROOT/'coordinates'/f'T{task:02d}_progress.npz')['unit'],d['unit'])
        web.append(dict(task=task,name=NAMES[task-1],demos=demos))
        print(f'T{task}: {EXPECTED[task-1]*2} task milestones; sphere coordinates unchanged',flush=True)
    (ROOT/'progress_annotations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (ROOT/'milestones.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    def panel(ax,task,e):
        a=np.linspace(0,2*np.pi,80)
        for lat in [-np.pi/3,-np.pi/6,0,np.pi/6,np.pi/3]:ax.plot(np.cos(a)*np.cos(lat),np.sin(a)*np.cos(lat),np.full_like(a,np.sin(lat)),color='#d5dfe5',lw=.4)
        for lon in np.linspace(0,np.pi,6,endpoint=False):ax.plot(np.cos(a)*np.cos(lon),np.cos(a)*np.sin(lon),np.sin(a),color='#d5dfe5',lw=.4)
        demo=web[task-1]['demos'][e];u=np.asarray(demo['points'])
        for j,points in enumerate(demo['arcs']):ax.plot(*np.asarray(points).T,color=CMAP(demo['progress'][j+1]),alpha=.6,lw=.8)
        ax.scatter(*u.T,c=demo['progress'],cmap=CMAP,vmin=0,vmax=1,s=16,edgecolors='white',linewidths=.3,depthshade=False)
        ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),zlim=(-1.12,1.12));ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-55);ax.set_proj_type('ortho');ax.set_axis_off();ax.set_title(f'Demo {e+1}',pad=0)
        labels=[]
        for ev in demo['events']:
            if ev['kind']!='grasp' or ev['point_index'] is None:continue
            xyz=u[ev['point_index']];xp,yp,_=proj3d.proj_transform(*xyz,ax.get_proj());labels.append((xp,yp,ev['label']))
        for side in [-1,1]:
            group=sorted([v for v in labels if (-1 if v[0]<0 else 1)==side],key=lambda v:v[1])
            for (xp,yp,label),y in zip(group,np.linspace(.28,.76,len(group)) if len(group)>1 else [.52]):
                ax.annotate(label,xy=(xp,yp),xycoords='data',xytext=(.04 if side<0 else .96,y),textcoords='axes fraction',
                    ha='right' if side<0 else 'left',va='center',fontsize=10,color='#143d29',
                    arrowprops=dict(arrowstyle='-',color='#668375',lw=.55),bbox=dict(facecolor='white',edgecolor='none',pad=.3,alpha=.8))
    scalar=plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=CMAP)
    with PdfPages(ROOT/'fig16_milestone_progress_10pages.pdf') as pages:
        for task in range(1,11):
            fig=plt.figure(figsize=(7.16,5.2))
            for e in range(5):panel(fig.add_subplot(2,3,e+1,projection='3d'),task,e)
            ax=fig.add_subplot(236);ax.axis('off');ax.text(.08,.75,'G1, G2, ...: grasp milestones\nP1, P2, ...: placement milestones\n\nProgress = completed milestones\n                 / required milestones\n\nEqual weight per milestone.\nArcs indicate visual interpolation.',va='top',fontsize=10,linespacing=1.5)
            fig.suptitle(f'T{task} · {NAMES[task-1]}',fontsize=10,y=.99);fig.subplots_adjust(left=.025,right=.965,top=.91,bottom=.12,wspace=.2,hspace=.05)
            cb=fig.colorbar(scalar,cax=fig.add_axes([.27,.085,.46,.014]),orientation='horizontal');cb.set_ticks([0,.25,.5,.75,1],labels=['0%','25%','50%','75%','100%']);cb.set_label('Task milestone progress',labelpad=1)
            for ext in ['pdf','png','svg']:fig.savefig(ROOT/'individual'/f'T{task:02d}_milestone_progress.{ext}',bbox_inches='tight',pad_inches=0,dpi=300)
            pages.savefig(fig,bbox_inches='tight',pad_inches=0);plt.close(fig)
    template=(ROOT/'viewer_template.html').read_text(encoding='utf-8')
    (ROOT/'memory_milestone_progress.html').write_text(template.replace('__MEMORY_DATA__',json.dumps(web,ensure_ascii=False)),encoding='utf-8')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
