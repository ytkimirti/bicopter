"""Copy brains from a run into the web app and write its manifest. Standard library only.

python3 publish.py runs/main
"""

import json
import shutil
import sys
from pathlib import Path

run = Path(sys.argv[1])
dest = Path(__file__).resolve().parents[1] / "web" / "public" / "brains"
dest.mkdir(parents=True, exist_ok=True)
for old in dest.glob("*.json"):
    old.unlink()

hours = ""
log = run / "log.tsv"
if log.exists():
    last = log.read_text().strip().splitlines()[-1].split("\t")
    hours = f" in {int(last[2].rstrip('s')) / 60:.0f} min on 4 CPU cores"


def fmt(steps):
    return f"{steps / 1e6:.0f}M" if steps >= 1e6 else f"{steps / 1e3:.0f}k"


entries = []
checkpoints = sorted(run.glob("brain_*M.json"), key=lambda p: int(p.stem.split("_")[1][:-1]))
for p in checkpoints:
    b = json.loads(p.read_text())
    sid = f"ppo-{p.stem.split('_')[1]}"
    shutil.copy(p, dest / f"{sid}.json")
    entries.append(
        {
            "id": sid,
            "label": f"PPO · {fmt(b['steps'])} steps",
            "file": f"{sid}.json",
            "steps": b["steps"],
            "info": f"An early checkpoint from the same training run, after {fmt(b['steps'])} simulated steps.",
        }
    )
best = json.loads((run / "brain_best.json").read_text())
shutil.copy(run / "brain_best.json", dest / "ppo-best.json")
entries.append(
    {
        "id": "ppo-best",
        "label": f"PPO · {fmt(best['steps'])} steps (best)",
        "file": "ppo-best.json",
        "steps": best["steps"],
        "info": (
            f"Trained with PPO on {fmt(best['steps'])} simulated steps{hours}, with randomized mass, motors, lag,"
            " latency, wind and sensor noise, so it copes when you change them."
        ),
    }
)
entries.reverse()
(dest / "manifest.json").write_text(json.dumps({"default": "ppo-best", "brains": entries}, indent=2))
print(f"published {len(entries)} brains to {dest}")
