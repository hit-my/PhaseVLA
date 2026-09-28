from pathlib import Path
import sys,runpy
R=Path(__file__).parent;P=Path('/home/nvidia/zyx/PhaseVLA')
sys.path[:0]=[str(R),str(P/'src'),str(P)]
from mamba_differentiable_step import install_training_fix
install_training_fix()
runpy.run_path(str(R/'train_entry.py'),run_name='__main__')
