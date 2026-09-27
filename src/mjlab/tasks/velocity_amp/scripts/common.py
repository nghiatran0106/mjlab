"""Shared helpers for the AMP expert-collection and evaluation scripts."""

from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

EXPERT_TASK = "Mjlab-Velocity-Flat-Unitree-G1-Expert"


def make_env(
  num_envs: int, seed: int, device: str, pushes: bool, task: str = EXPERT_TASK
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
  """G1 flat env with the original reward and the ``amp`` feature group."""
  cfg: ManagerBasedRlEnvCfg = load_env_cfg(task)
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  if not pushes:
    cfg.events.pop("push_robot", None)
  env = ManagerBasedRlEnv(cfg, device=device)
  agent_cfg = load_rl_cfg(EXPERT_TASK)
  return env, RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def load_policy(wrapped: RslRlVecEnvWrapper, checkpoint: str, device: str):
  """Loads only the actor, so PPO, task-only and AMP checkpoints all work."""
  agent_cfg = asdict(load_rl_cfg(EXPERT_TASK))
  runner = VelocityOnPolicyRunner(wrapped, agent_cfg, device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def default_device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"
