import os
from typing import Callable

import gymnasium as gym
import imageio
import numpy as np
import torch


def evaluate_ppo(model_path: str,
                 make_env: Callable,
                 env_id: str,
                 eval_episodes: int,
                 run_name: str,
                 Model: torch.nn.Module,
                 device: torch.device = torch.device("cpu"),
                 capture_video: bool = True,
                 gamma: float = 0.99,
):

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    frames = envs.render()

    # Create gif with imageio
    print("Saving GIF file")
    with imageio.get_writer(
            os.path.join(f"./runs/{run_name}"),
            mode="I",
            fps=30) as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
    print("Finish")

    return episodic_returns

