from pathlib import Path
import subprocess
r=Path('/pfs/pfs-7jnepv/lgd')
for name in ['phasevla-runtime','PhaseVLA','datasets','libero_fullmatrix_20260909','actmem_gradientfix_v1_20260910']:
 p=r/name;print('DIR',p);print('\n'.join(str(x) for x in p.iterdir()))
print('HDF5',subprocess.run(['find',str(r/'datasets'),'-maxdepth','5','-name','KITCHEN_SCENE1_3_*.hdf5'],capture_output=True,text=True).stdout)
print('PYTHON',subprocess.run(['find',str(r/'phasevla-runtime'),'-maxdepth','4','-name','python'],capture_output=True,text=True).stdout)
for p in (r/'actmem_gradientfix_v1_20260910').glob('*.py'):
 print('SCRIPT',p, p.read_text()[:5000])
