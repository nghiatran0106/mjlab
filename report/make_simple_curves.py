"""Learning curve rút gọn (3 ô) cho báo cáo ngắn.

Usage: ~/venvs/mjlab/bin/python report/make_simple_curves.py fair_results-20260928-040113
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_train_curves import ema, find, load  # noqa: E402

RUNS = {
  "gail-v7c": ("v7c", "#1baf7a", 2.2),
  "gail-v7b": ("v7b", "#eda100", 1.4),
  "gail-v7a": ("v7a", "#eb6834", 1.4),
}
MUTED, GRID, EXPERT_C = "#52514e", "#e6e5e1", "#2a78d6"
plt.rcParams.update(
  {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "axes.edgecolor": MUTED,
    "legend.frameon": False,
  }
)

roots = [Path(a) for a in sys.argv[1:]] + [Path("logs")]
data = {k: load(find(roots, k)) for k in RUNS}
expert = load(find(roots, "expert-step"))
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
for k, (label, color, lw) in RUNS.items():
  d = data[k]
  x, y = d["Episode_Reward/track_linear_velocity"]
  axes[0].plot(x[x <= 2000], ema(y)[x <= 2000], color=color, lw=lw, label=label)
  x, y = d["Loss/amp_style_reward"]
  axes[1].plot(x[x <= 2000], ema(y)[x <= 2000], color=color, lw=lw)
  x, e = d["Loss/amp_d_expert"]
  _, p = d["Loss/amp_d_policy"]
  axes[2].plot(x[x <= 2000], ema(e - p)[x <= 2000], color=color, lw=lw)
axes[0].axhline(
  expert["Episode_Reward/track_linear_velocity"][1][-1],
  color=EXPERT_C,
  ls="--",
  lw=1.2,
  label="expert",
)
titles = [
  "(a) Reward bám vận tốc dài",
  "(b) Style reward $r^S$",
  "(c) $\\bar d_E - \\bar d_\\pi$",
]
for ax, t in zip(axes, titles, strict=True):
  ax.set_title(t, fontsize=8.5)
  ax.set_xlabel("Iteration")
fig.legend(
  *axes[0].get_legend_handles_labels(),
  loc="upper center",
  ncol=4,
  bbox_to_anchor=(0.5, 1.1),
  fontsize=8,
)
fig.tight_layout()
out = Path(__file__).parent / "figures" / "simple_curves.pdf"
fig.savefig(out, bbox_inches="tight")
print("->", out)
