import os

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv, EpisodicLifeEnv, FireResetEnv


def wrap_env(env):
    # env = NoopResetEnv(env, noop_max=40)
    # env = EpisodicLifeEnv(env)
    # if "FIRE" in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


# Evaluation
eval_env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
model_path = f"C:/Users/SKG/GitHub/RL_MIPT/HW2/models/breakout/BreakOut_8000000.tar"
eval_env = wrap_env(eval_env)


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = QNetwork(eval_env).to(device)
pretrained_weights = torch.load(model_path)
q_network.load_state_dict(pretrained_weights)
print("weights loaded")

frames = []
scores = 0
(s, _), done, ret = eval_env.reset(), False, 0
while not done:
    frames.append(eval_env.render())
    q_values = q_network(torch.Tensor(np.array(s)).unsqueeze(0).to(device))
    actions = torch.argmax(q_values, dim=1).cpu().item()
    s_next, r, terminated, truncated, info = eval_env.step(actions)
    s = s_next
    ret += r
    done = terminated or truncated
scores += ret

# Create gif with imageio
print("Saving GIF file")
with imageio.get_writer(
        os.path.join("./video", f"{model_path.split('/')[-1].split('.')[0]}.gif"),
        mode="I",
        fps=30) as writer:
    for idx, frame in enumerate(frames):
        writer.append_data(frame)
print("Finish")
