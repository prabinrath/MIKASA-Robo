import argparse

import gymnasium as gym
import numpy as np
import torch

import mikasa_robo_suite  # noqa: F401 - registers MIKASA-Robo environments
from mikasa_robo_suite.memory_envs.remember_color import (
    RememberColor3Env,
    RememberColor5Env,
    RememberColor9Env,
)
from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper


REMEMBER_COLOR_ENVS = {
    "RememberColor3-v0": RememberColor3Env,
    "RememberColor5-v0": RememberColor5Env,
    "RememberColor9-v0": RememberColor9Env,
}


def as_step_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="RememberColor3-v0")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-time", type=int, default=250)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--obs-mode", default="state", choices=["state", "rgb"])
    parser.add_argument("--render-mode", default="all")
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-dir", default="videos/remember_color")
    args = parser.parse_args()

    if args.delta_time < 0:
        raise ValueError("--delta-time must be non-negative")
    if args.env_id not in REMEMBER_COLOR_ENVS:
        raise ValueError(f"{args.env_id} is not a RememberColor env")

    env_cls = REMEMBER_COLOR_ENVS[args.env_id]
    max_episode_steps = args.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = env_cls.max_episode_steps_for_delta_time(args.delta_time)
    rollout_steps = args.steps if args.steps is not None else max_episode_steps

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        max_episode_steps=max_episode_steps,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_delta_pos",
        render_mode=args.render_mode,
        reward_mode="normalized_dense",
        sim_backend="gpu",
        delta_time=args.delta_time,
    )

    if args.record:
        from mani_skill.utils.wrappers import RecordEpisode

        env = RecordEpisode(
            env,
            args.video_dir,
            save_trajectory=False,
            max_steps_per_video=rollout_steps,
        )

    env = StateOnlyTensorToDictWrapper(env)

    cue_steps = env.unwrapped.TIME_OFFSET
    delay_steps = env.unwrapped.DELTA_TIME
    option_start = cue_steps + delay_steps
    option_steps = max_episode_steps - option_start
    color_mapping = env.unwrapped.COLOR_MAPPING

    try:
        obs, info = env.reset(seed=args.seed)

        oracle = as_step_array(info["oracle_info"]).reshape(-1)
        targets = [color_mapping[int(idx)][0] for idx in oracle]

        print(f"env={args.env_id} num_envs={args.num_envs}")
        print(
            f"cue_steps={cue_steps} delay_steps={delay_steps} "
            f"options_start_step={option_start} option_steps={option_steps} "
            f"max_episode_steps={max_episode_steps}"
        )
        print(f"targets={targets}")

        for _ in range(rollout_steps):
            elapsed = as_step_array(info["elapsed_steps"]).reshape(-1)
            step_id = int(elapsed[0])
            if step_id < cue_steps:
                phase = "cue"
            elif step_id < option_start:
                phase = "delay"
            else:
                phase = "options"

            action = torch.from_numpy(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)

            reward_np = as_step_array(reward).reshape(-1)
            success_np = as_step_array(info["success"]).reshape(-1)
            print(
                f"step={step_id:02d} phase={phase:<7} "
                f"reward0={float(reward_np[0]):.3f} success0={bool(success_np[0])}"
            )

            if bool(as_step_array(terminated | truncated).all()):
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
