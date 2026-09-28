"""Learning curve của vòng 7 từ log TensorBoard.

Usage (từ thư mục repo; ROOT là thư mục chứa log đã giải nén, ví dụ fair_result):
  ~/venvs/mjlab/bin/python report/make_train_curves.py ROOT [ROOT2 ...]
Tìm các thư mục *_gail-v7a, *_gail-v7b, *_gail-v7c, *_amp-v7d và *_expert-step.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
  EventAccumulator,
)

OUT = Path(__file__).resolve().parent / "figures" / "v7_train_curves.pdf"
RUNS = {
  "gail-v7a": ("v7a: trộn 30/70 + replay", "#eb6834"),
  "gail-v7b": ("v7b: + discriminator biết lệnh", "#eda100"),
  "gail-v7c": ("v7c: + RSI", "#1baf7a"),
  "amp-v7d": ("v7d: như v7c, loss AMP", "#e87ba4"),
}
PANELS = [
  ("Episode_Reward/track_linear_velocity", "Reward bám vận tốc dài ↑", True),
  ("Train/mean_episode_length", "Độ dài episode (bước) ↑", True),
  ("Policy/mean_std", "Action std $\\sigma$", True),
  ("Loss/amp_style_reward", "Style reward $r^S$ ↑", False),
  ("Loss/amp_d_expert", "Điểm discriminator: expert $\\bar d_E$", False),
  ("Loss/amp_d_policy", "Điểm discriminator: policy $\\bar d_\\pi$", False),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
EXPERT_C = "#2a78d6"

plt.rcParams.update(
  {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
  }
)


def find(roots: list[Path], suffix: str) -> Path | None:
  hits = sorted(
    p for r in roots for p in r.rglob(f"*_{suffix}") if list(p.glob("events.out.*"))
  )
  return hits[-1] if hits else None


def load(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  tags = set(ea.Tags()["scalars"])
  out = {}
  for tag, _, _ in PANELS:
    if tag in tags:
      ev = ea.Scalars(tag)
      out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
  return out


def ema(y: np.ndarray, alpha: float = 0.05) -> np.ndarray:
  out = np.empty_like(y, dtype=float)
  acc = y[0]
  for i, v in enumerate(y):
    acc = alpha * v + (1 - alpha) * acc
    out[i] = acc
  return out


def main() -> None:
  roots = [Path(a) for a in sys.argv[1:]] or [Path("logs")]
  data = {}
  for suffix in RUNS:
    d = find(roots, suffix)
    print(f"[INFO] {suffix}: {d}")
    if d is not None:
      data[suffix] = load(d)
  expert_dir = find(roots + [Path("logs")], "expert-step")
  expert = load(expert_dir) if expert_dir is not None else {}
  print(f"[INFO] expert-step: {expert_dir}")

  fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
  for ax, (tag, title, show_expert) in zip(axes.flat, PANELS, strict=True):
    for suffix, (label, color) in RUNS.items():
      if suffix in data and tag in data[suffix]:
        x, y = data[suffix][tag]
        keep = x <= 2000
        ax.plot(x[keep], ema(y)[keep], color=color, lw=1.8, label=label)
    if show_expert and tag in expert:
      ax.axhline(
        expert[tag][1][-1], color=EXPERT_C, lw=1.2, ls="--", label="Expert step (cuối)"
      )
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_xlabel("Iteration")
  axes[0, 2].set_yscale("log")
  handles, labels = [], []
  for ax in axes.flat:
    for h, lbl in zip(*ax.get_legend_handles_labels(), strict=True):
      if lbl not in labels:
        handles.append(h)
        labels.append(lbl)
  fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.07),
    fontsize=7.5,
  )
  fig.tight_layout()
  OUT.parent.mkdir(exist_ok=True)
  fig.savefig(OUT, bbox_inches="tight")
  print(f"-> {OUT}")


if __name__ == "__main__":
  main()
