"""Uniform label-independent t-SNE protocol for all ten task-specific memories.

Run normally only when all ten NPZs exist. --available generates individual
available-task plots and an explicitly named PARTIAL overview.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
NAMES=['Bowl ×1','Bottle ×1','Bowl ×3','Bottle ×3','Bowl ×5','Bowl ×7',
       'Bowl swap','Bowl rotation','Filled basket','Empty basket']

def save(fig,stem):
    for ext in ['pdf','svg','png']:
        fig.savefig(str(stem)+'.'+ext,bbox_inches='tight',pad_inches=0,dpi=350)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--available',action='store_true');args=ap.parse_args()
    present=[i for i in range(1,11) if (ROOT/f'T{i:02d}_memory.npz').exists()]
    if len(present)!=10 and not args.available:raise RuntimeError(f'Incomplete input: only {present}; no complete ten-task figure generated.')
    (ROOT/'individual').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':10,'axes.titlesize':10,'axes.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    info=json.loads((ROOT/'extraction_manifest.json').read_text())
    cache=ROOT/'embeddings';cache.mkdir(exist_ok=True)
    results={};metrics={}
    for t in present:
        path=ROOT/f'T{t:02d}_memory.npz';d=np.load(path)
        x=d['memory'].astype(np.float64);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        meta_path=cache/f'T{t:02d}.json';xy_path=cache/f'T{t:02d}.npz'
        old=json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if xy_path.exists() and old.get('source_sha256')==digest:
            xy=np.load(xy_path)['xy'];stats=old
        else:
            assert len(x)>15 and np.isfinite(x).all()
            with threadpool_limits(limits=2):
                pca=PCA(n_components=min(30,len(x)-1),random_state=42)
                reduced=pca.fit_transform(x)
                model=TSNE(n_components=2,perplexity=15,init='pca',learning_rate='auto',max_iter=1500,random_state=42)
                xy=model.fit_transform(reduced)
                quality=trustworthiness(x,xy,n_neighbors=5)
            stats=dict(source_sha256=digest,samples=len(x),dimensions=x.shape[1],pca_components=pca.n_components_,
                       pca_variance_retained=float(pca.explained_variance_ratio_.sum()),perplexity=15,random_state=42,
                       max_iter=1500,init='pca',learning_rate='auto',trustworthiness_k5=float(quality),
                       kl_divergence=float(model.kl_divergence_))
            np.savez_compressed(xy_path,xy=xy,query=d['query'],episode=d['episode'],fraction=d['fraction'])
            meta_path.write_text(json.dumps(stats,indent=2),encoding='utf-8')
        results[t]=(xy,d);metrics[t]=stats
        print(f'T{t} projected: {len(x)} points',flush=True)
    def panel(ax,t,individual=False):
        if t not in results:
            ax.text(.5,.5,'Checkpoint access\npending',ha='center',va='center',transform=ax.transAxes,color='.5',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
        else:
            xy,d=results[t]
            for e in np.unique(d['episode']):
                ids=np.flatnonzero(d['episode']==e)
                segments=np.stack([xy[ids[:-1]],xy[ids[1:]]],axis=1)
                ax.add_collection(LineCollection(segments,colors='#95A1AB',linewidth=.45,alpha=.25))
            ax.scatter(*xy.T,c=d['fraction'],cmap='turbo',vmin=0,vmax=1,s=15 if individual else 9,
                       linewidths=.15,edgecolors='white',alpha=.9)
            ax.set_aspect('equal',adjustable='datalim');ax.margins(.1)
            ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'T{t}\n{NAMES[t-1]}',pad=5)
        for spine in ax.spines.values():spine.set_visible(False)
    scalar=plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='turbo')
    fig,axs=plt.subplots(2,5,figsize=(7.16,3.65),layout='constrained')
    for t,ax in enumerate(axs.flat,1):panel(ax,t)
    cb=fig.colorbar(scalar,ax=axs.ravel().tolist(),orientation='horizontal',fraction=.035,pad=.035,aspect=45)
    cb.set_label('Normalized elapsed time');cb.set_ticks([0,.25,.5,.75,1])
    name='fig12_all10_memory_tsne' if len(present)==10 else 'fig12_PARTIAL_memory_tsne'
    save(fig,ROOT/name)
    for t in present:
        fig,ax=plt.subplots(figsize=(3.5,3.1),layout='constrained');panel(ax,t,True)
        ax.set_xlabel('t-SNE 1');ax.set_ylabel('t-SNE 2')
        cb=fig.colorbar(scalar,ax=ax,orientation='horizontal',fraction=.045,pad=.03)
        cb.set_label('Normalized elapsed time')
        save(fig,ROOT/'individual'/f'T{t:02d}_memory_tsne')
    (ROOT/'projection_report.json').write_text(json.dumps(dict(complete=len(present)==10,tasks=metrics,
        note='Each task is fitted separately. Coordinates and global distances are not comparable across panels. Colors are elapsed time, not verified semantic progress.'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
