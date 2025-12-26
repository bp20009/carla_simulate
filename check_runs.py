# check_runs.py
import csv, json, re
from pathlib import Path

root = Path("results_grid_accident")

def parse_key(path_str: str):
    m = re.search(r"(?:^|[\\/])(autopilot|lstm)[\\/]+lead_(\d+)[\\/]+rep_(\d+)[\\/]+logs[\\/]+collisions\.csv$", path_str)
    if not m:
        return None
    return (m.group(1), int(m.group(2)), int(m.group(3)))

rows = []
for cpath in root.glob("**/logs/collisions.csv"):
    k = parse_key(str(cpath))
    if not k:
        continue
    meta = cpath.parent / "meta.json"
    if not meta.exists():
        continue

    d = json.loads(meta.read_text(encoding="utf-8", errors="ignore"))
    spf = int(float(d["switch_payload_frame_observed"]))

    cnt = 0
    mx = -1.0
    with cpath.open(encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                pf = int(float(row.get("payload_frame")))
                it = float(row.get("intensity"))
            except Exception:
                continue
            if pf < spf:
                continue
            cnt += 1
            mx = max(mx, it)

    rows.append((k, cnt, mx))

print("runs_parsed=", len(rows))
reps = sorted({k[2] for (k,_,_) in rows})
print("reps_seen=", reps, "max_rep=", (max(reps) if reps else None))

# 例：rep>=6で、after_switchの最大強度が1000未満のrun数
bad = sum(1 for (k,cnt,mx) in rows if k[2] >= 6 and mx < 1000.0)
print("rep>=6 AND max_intensity_after_switch<1000 runs=", bad)
