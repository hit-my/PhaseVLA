from pathlib import Path
import base64,io,zipfile,json
import numpy as np
root=Path(__file__).resolve().parent
encoded=(root/'fetch_lgd_features.result.txt').read_text().strip()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded,validate=True))) as z:
    allowed={f'T{t:02d}_memory.npz' for t in [3,4,5,9]}|{'lgd_extraction_manifest.json'}
    assert set(z.namelist())==allowed
    z.extractall(root)
manifest=json.loads((root/'extraction_manifest.json').read_text())
lgd=json.loads((root/'lgd_extraction_manifest.json').read_text())
manifest['tasks']=[r for r in manifest['tasks'] if r['task'] not in [3,4,5,9]]+lgd['tasks']
manifest['tasks'].sort(key=lambda r:r['task'])
manifest['lgd_replay_input_sha256']=lgd['input_file_sha256']
assert [r['task'] for r in manifest['tasks']]==list(range(1,11))
for row in manifest['tasks']:
    with np.load(root/f'T{row["task"]:02d}_memory.npz') as d:
        assert list(d['memory'].shape)==row['memory_shape']
        assert np.isfinite(d['memory']).all()
        assert set(d['episode'])==set(range(5))
    print('T',row['task'],row['memory_shape'],row['seed'],row['step'])
(root/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
