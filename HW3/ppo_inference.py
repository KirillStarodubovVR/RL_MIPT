import os
import random
from typing import Callable
import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def make_env(env_id, mode):
    env = gym.make(env_id, render_mode=mode)
    env = gym.wrappers.RenderCollection(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# TRY NOT TO MODIFY: seeding
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
envs = make_env("HalfCheetah-v4", "rgb_array")
agent = Agent(envs).to(device)
model_path = "./runs/HalfCheetah-v4__HalfCheetah__1__1710836467/HalfCheetah_999424.tar"
agent.load_state_dict(torch.load(model_path, map_location=device))
agent.eval()

# for mean and std
episodic_returns = []
for i in range(50):
    obs, _ = envs.reset()
    returns = 0
    while True:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(device))
        next_obs, reward, term, trun, infos = envs.step(actions.squeeze(0).cpu().numpy())
        obs = next_obs
        returns += reward
        if trun:
            break
    episodic_returns.append(returns)

mean_reward = np.array(episodic_returns).mean()
std_reward = np.array(episodic_returns).std()
print(mean_reward, std_reward)

# for gif
obs, _ = envs.reset()
frames = []
while True:
    frame = envs.render()
    frames.append(np.array(frame).squeeze(0))
    # im = cv2.imread(frames[-1])
    # cv2.imshow("im", frames[-1])
    # cv2.waitKey(0)
    actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(device))
    next_obs, _, term, trun, infos = envs.step(actions.squeeze(0).cpu().numpy())
    obs = next_obs
    if trun:
        break

# Create gif with imageio
print("Saving GIF file")
with imageio.get_writer(
        os.path.join(f"./runs/test.gif"),
        mode="I",
        fps=30) as writer:
    for idx, frame in enumerate(frames):
        writer.append_data(frame)
print("Finish")
