"""Iterations-to-target from checkpoints evaluated with evaluate.py.

Policies must be named ``<config>_s<seed>_it<iteration>`` (plus a reference
``expert`` entry). A checkpoint meets the target when it walks as asked and
close to the expert:

  - tracking error <= MAX_LIN_ERR m/s,
  - true reward >= MIN_REWARD_FRAC * expert true reward,
  - swing peak >= MIN_LIFT_CM,
  - pelvis down (crawling) <= MAX_DOWN of the time, falls <= MAX_FALLS per minute.

With ``--reference-config`` (used for robot variants, where the nominal expert
is no longer the right yardstick) the target is relative to the converged
checkpoints of that config instead: true reward >= MIN_REWARD_FRAC and swing
peak >= MIN_LIFT_FRAC of their mean, tracking error <= their mean + LIN_ERR_MARGIN.

Usage:
  python -m mjlab.tasks.velocity_amp.scripts.convergence results.json \
    [--expert expert_step | --reference-config ppo] [--out summary.json]
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

MAX_LIN_ERR = 0.22
MIN_REWARD_FRAC = 0.90
MIN_LIFT_CM = 5.0
MAX_DOWN = 0.01
MAX_FALLS = 0.1
MIN_LIFT_FRAC = 0.8
LIN_ERR_MARGIN = 0.05

_NAME = re.compile(r"^(?P<config>.+)_s(?P<seed>\d+)_it(?P<it>\d+)$")


def meets_target(r: dict, ref: dict) -> bool:
  return (
    r["lin_vel_error_m_s"] <= ref["max_lin_err"]
    and r["true_reward_rate"] >= ref["min_reward_frac"] * ref["reward"]
    and r["swing_peak_cm"] >= ref["min_lift"]
    and r["down_fraction"] <= MAX_DOWN
    and r["falls_per_minute"] <= MAX_FALLS
  )


def summarize(
  results: dict,
  expert: str,
  reference_config: str | None = None,
  min_reward_frac: float = MIN_REWARD_FRAC,
) -> dict:
  runs: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
  for name, r in results.items():
    m = _NAME.match(name)
    if m:
      runs[(m["config"], int(m["seed"]))][int(m["it"])] = r

  if reference_config is None:
    ref = {
      "reward": results[expert]["true_reward_rate"],
      "max_lin_err": MAX_LIN_ERR,
      "min_lift": MIN_LIFT_CM,
    }
  else:
    finals = [c[max(c)] for (cfg, _), c in runs.items() if cfg == reference_config]
    if not finals:
      raise ValueError(f"no runs of reference config {reference_config!r}")
    mean = lambda k: sum(r[k] for r in finals) / len(finals)  # noqa: E731
    ref = {
      "reward": mean("true_reward_rate"),
      "max_lin_err": mean("lin_vel_error_m_s") + LIN_ERR_MARGIN,
      "min_lift": MIN_LIFT_FRAC * mean("swing_peak_cm"),
    }
  expert_reward = ref["reward"]
  ref["min_reward_frac"] = min_reward_frac

  per_run = {}
  for (config, seed), ckpts in sorted(runs.items()):
    iters = sorted(ckpts)
    hit = next((it for it in iters if meets_target(ckpts[it], ref)), None)
    last = ckpts[iters[-1]]
    per_run[f"{config}_s{seed}"] = {
      "config": config,
      "seed": seed,
      "iterations_to_target": hit,
      "final_iteration": iters[-1],
      "final_reward_frac": last["true_reward_rate"] / expert_reward,
      "final_lin_err": last["lin_vel_error_m_s"],
      "final_lift_cm": last["swing_peak_cm"],
    }

  per_config: dict[str, dict] = {}
  for config in sorted({v["config"] for v in per_run.values()}):
    rows = [v for v in per_run.values() if v["config"] == config]
    hits = [v["iterations_to_target"] for v in rows]
    reached = [h for h in hits if h is not None]
    fracs = [v["final_reward_frac"] for v in rows]
    per_config[config] = {
      "seeds": len(rows),
      "reached": len(reached),
      "iterations_to_target_mean": sum(reached) / len(reached) if reached else None,
      "iterations_to_target_all": hits,
      "final_reward_frac_mean": sum(fracs) / len(fracs),
      "final_reward_frac_std": _std(fracs),
    }
  return {"reference": ref, "per_run": per_run, "per_config": per_config}


def _std(xs: list[float]) -> float:
  if len(xs) < 2:
    return 0.0
  mean = sum(xs) / len(xs)
  return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("results", type=Path)
  parser.add_argument("--expert", default="expert_step")
  parser.add_argument("--reference-config", default=None)
  parser.add_argument("--min-reward-frac", type=float, default=MIN_REWARD_FRAC)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()
  summary = summarize(
    json.loads(args.results.read_text()),
    args.expert,
    args.reference_config,
    args.min_reward_frac,
  )
  print(f"Target: true reward >= {100 * args.min_reward_frac:.0f}% of the reference")
  print()
  print(
    "| config | seeds reached | iterations to target (each seed) "
    "| final reward / reference |"
  )
  print("|---|---|---|---|")
  for config, c in summary["per_config"].items():
    hits = ", ".join(
      "-" if h is None else str(h) for h in c["iterations_to_target_all"]
    )
    print(
      f"| {config} | {c['reached']}/{c['seeds']} | {hits} "
      f"| {100 * c['final_reward_frac_mean']:.0f}% "
      f"± {100 * c['final_reward_frac_std']:.0f} |"
    )
  if args.out is not None:
    args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
