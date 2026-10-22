"""Copy a hide and seek run into web/public/hideseek: a spread of replays, the training curves, detected stages.

python3 publish_game.py runs/hs1 [--max 24]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--max", type=int, default=24, help="checkpoints to publish, spread evenly over the run")
args = p.parse_args()

run = Path(args.run)
dest = Path(__file__).resolve().parents[1] / "web" / "public" / "hideseek"

rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
# A resumed run appends to the same log; keep the last row for each update.
by_update = {r["update"]: r for r in rows}
rows = [by_update[k] for k in sorted(by_update)]

KEYS = ["hidden", "in_room", "sealed", "locked_prep", "unsealed", "key_unlocks", "key_denied", "grabs_h", "grabs_s", "throws", "flips"]


def smooth(xs, alpha=0.1):
    out, m = [], None
    for x in xs:
        m = x if m is None else m + alpha * (x - m)
        out.append(m)
    return out


ema = {k: smooth([r.get(k, 0.0) for r in rows]) for k in KEYS + ["r_hider"]}

# Stage onsets: the smoothed metric first holds above its threshold for 20 consecutive updates.
STAGES = [
    ("hide", "Hiders hide in the room", "in_room", 0.5),
    ("seal", "Hiders seal the doors", "sealed", 0.6),
    ("lock", "Hiders lock the seal", "locked_prep", 1.0),
    ("break", "Seekers break in", "unsealed", 0.3),
    ("key", "Seekers use the key", "key_unlocks", 0.2),
    ("deny", "Hiders take the key away", "key_denied", 0.3),
]
stages = []
for sid, label, k, thr in STAGES:
    run_len = 0
    for i, v in enumerate(ema[k]):
        run_len = run_len + 1 if v > thr else 0
        if run_len == 20:
            r = rows[i - 19]
            stages.append({"id": sid, "label": label, "metric": k, "decisions": r["decisions"], "hours": r["elapsed"] / 3600})
            break

n = len(rows)
step = max(1, n // 400)
curves = {"decisions": [r["decisions"] for r in rows][::step], "hours": [round(r["elapsed"] / 3600, 3) for r in rows][::step]}
for k, v in ema.items():
    curves[k] = [round(x, 4) for x in v[::step]]

replays = sorted(run.glob("replay_*.json"))
if len(replays) > args.max:
    idx = sorted({round(i * (len(replays) - 1) / (args.max - 1)) for i in range(args.max)})
    replays = [replays[i] for i in idx]

tmp = dest.with_name("hideseek.tmp")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
checkpoints = []
for f in replays:
    doc = json.loads(f.read_text())
    m = doc["metrics"]
    shutil.copy(f, tmp / f.name)
    checkpoints.append(
        {
            "id": f.stem.removeprefix("replay_"),
            "file": f.name,
            "decisions": m["decisions"],
            "episodes_seen": int(sum(r["episodes"] for r in rows if r["update"] <= m["update"])),
            "hours": round(m["elapsed"] / 3600, 2),
            "metrics": {k: round(m.get(k, 0.0), 3) for k in KEYS + ["r_hider"]},
        }
    )
(tmp / "manifest.json").write_text(json.dumps({"checkpoints": checkpoints, "stages": stages}, indent=1))
(tmp / "curves.json").write_text(json.dumps(curves, separators=(",", ":")))
# Swap in one rename so the dev server never serves a half-written directory.
old = dest.with_name("hideseek.old")
shutil.rmtree(old, ignore_errors=True)
if dest.exists():
    os.rename(dest, old)
os.rename(tmp, dest)
shutil.rmtree(old, ignore_errors=True)
print(f"published {len(checkpoints)} checkpoints, {len(stages)} stages to {dest}")
for s in stages:
    print(f"  {s['hours']:.2f} h  {s['decisions'] / 1e6:.0f}M decisions  {s['label']}")
