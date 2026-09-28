from pathlib import Path
import subprocess
for p in [Path('/pfs/pfs-7jnepv/lgd'),Path('/home/lgd')]:
 print('ROOT',p)
 if p.exists():print('\n'.join(str(x) for x in p.iterdir()))
print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv'],capture_output=True,text=True).stdout)
