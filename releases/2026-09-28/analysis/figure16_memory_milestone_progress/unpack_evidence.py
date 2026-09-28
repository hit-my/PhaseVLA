from pathlib import Path
import base64,zipfile,io,json
p=Path(__file__).resolve().parent
payload=Path('C:/Users/lenovo/phasevla-session-migration-20260907/memory_milestones_payload.txt').read_text().strip()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload,validate=True))) as z:
 for name in z.namelist():assert name=='milestone_candidates.json' or name in {f'evidence/T{t:02d}_milestones.jpg' for t in range(1,11)}
 z.extractall(p)
rows=json.loads((p/'milestone_candidates.json').read_text())
for t in range(1,11):print(t,[len(r['cycles']) for r in rows if r['task']==t])
