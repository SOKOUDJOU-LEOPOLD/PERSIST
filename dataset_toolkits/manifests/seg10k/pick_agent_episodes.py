"""Pick validation episodes where agents are clearly visible, for the pixel-denoiser trial rollouts.

Score = sum over agent classes (kind "entity" in seg_classes.json) of pixel_fraction x frames, i.e. how much screen the
agents cover over the episode. Keeps the top N voxelized validation episodes and writes their sha256s, one per line.
Usage: python pick_agent_episodes.py [N=6]   (after preprocessing has assigned the validation split)
"""
import json
import sys

import pandas as pd

DS = "datasets/seg10k/train"
OUT = "dataset_toolkits/manifests/seg10k/rollout_sha256.txt"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6

m = pd.read_csv(f"{DS}/metadata.csv")
m = m[m.validation_set & m.voxelized]
rows = []
for sha, path in zip(m.sha256, m.local_path):
    classes = json.load(open(f"{DS}/{path}/seg_classes.json"))["classes"]
    agents = [c for c in classes if c["kind"] == "entity"]
    rows.append(dict(sha256=sha, episode=path.rsplit("/", 1)[1], n_agents=len(agents),
                     score=sum(c["pixel_fraction"] * c["frames"] for c in agents),
                     agents=",".join(c["name"].split(":")[-1] for c in agents)))
top = pd.DataFrame(rows).sort_values("score", ascending=False).head(n)
print(top.to_string(index=False))
open(OUT, "w").write("\n".join(top.sha256) + "\n")
print(f"wrote {OUT}")
