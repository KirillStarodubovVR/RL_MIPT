import os
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RL_MIPT_HWs"
    """the wandb's project name"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    load_model: bool = False
    """load the model"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    total_timesteps: int = 6_000_000
    """total timesteps of the experiments"""
    num_checkpoints: int = 11
    """number of checkpoint to save weights and create gif"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 6
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


"""
-------------------------------------------------------------------------------------
Action Space
There are four discrete actions available:
0: do nothing
1: fire left orientation engine
2: fire main engine
3: fire right orientation engine
-------------------------------------------------------------------------------------
Observation Space
The state is an 8-dimensional vector: the coordinates of the lander in x & y, 
its linear velocities in x & y, its angle, its angular velocity, and two booleans 
that represent whether each leg is in contact with the ground or not.
Example:
                1         2      3    4      5     6      7          8
observation: x-coord | y-coord | Vx | Vy | alpha | w | Leg_Left | Leg_Right
-------------------------------------------------------------------------------------
Rewards
After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.
For each step, the reward:
* is increased/decreased the closer/further the lander is to the landing pad. (distance)
* is increased/decreased the slower/faster the lander is moving. (speed)
* is decreased the more the lander is tilted (angle not horizontal). (angle)
* is increased by 10 points for each leg that is in contact with the ground. (contact left leg)
* is decreased by 0.03 points each frame a side engine is firing. (decrease the amount of side engine firing)
* is decreased by 0.3 points each frame the main engine is firing. (decrease the amount of main engine firing)
* The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively. (goal of fail)
An episode is considered a solution if it scores at least 200 points.

continues action:
If continuous=True is passed, continuous actions (corresponding to the throttle of the engines) will be 
used and the action space will be Box(-1, +1, (2,), dtype=np.float32). The first coordinate of an action 
determines the throttle of the main engine, while the second coordinate specifies the throttle of the lateral boosters. 
Given an action np.array([main, lateral]), the main engine will be turned off completely if main < 0 and the throttle 
scales affinely from 50% to 100% for 0 <= main <= 1 (in particular, the main engine doesn’t work with less than 50% power). 
Similarly, if -0.5 < lateral < 0.5, the lateral boosters will not fire at all. If lateral < -0.5, the left booster will 
fire, and if lateral > 0.5, the right booster will fire. Again, the throttle scales affinely from 50% to 100% 
between -1 and -0.5 (and 0.5 and 1, respectively).
"""


def make_env(env_id,
             idx,
             capture_video,
             gamma,
             render_mode: str = "rgb_array",
             continuous: bool = False,
             gravity: float = -10.0,
             wind: bool = True,
             wind_power: float = 15.0,
             turbulence_power: float = 1.5):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id,
                           render_mode=render_mode,
                           continuous=continuous,
                           gravity=gravity,
                           enable_wind=wind,
                           wind_power=wind_power,
                           turbulence_power=turbulence_power)
            env = gym.wrappers.RenderCollection(env)
        else:
            env = gym.make(env_id,
                           render_mode=render_mode,
                           continuous=continuous,
                           gravity=gravity,
                           enable_wind=wind,
                           wind_power=wind_power,
                           turbulence_power=turbulence_power)

        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


def evaluate(agent, run_name, iteration, seed):
    # create env
    def make_env_eval(env_id, mode):
        env = gym.make(env_id, render_mode=mode)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    with torch.no_grad():
        envs_eval = make_env_eval("HalfCheetah-v4", "rgb_array")
        # for mean and std
        episodic_returns = []

        for i in range(10):
            obs, _ = envs_eval.reset(seed=seed)
            returns = 0
            while True:
                actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(device))
                next_obs, reward, term, trun, infos = envs_eval.step(actions.squeeze(0).cpu().numpy())
                obs = next_obs
                returns += reward
                if trun:
                    break
            episodic_returns.append(returns)

        mean_reward = np.array(episodic_returns).mean()
        std_reward = np.array(episodic_returns).std()
        print(mean_reward, std_reward)

        # for gif
        obs, _ = envs_eval.reset(seed=seed)
        frames = []
        while True:
            frame = envs_eval.render()
            # frames.append(np.array(frame).squeeze(0))
            frames.append(np.array(frame))
            # im = cv2.imread(frames[-1])
            # cv2.imshow("im", frames[-1])
            # cv2.waitKey(0)
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(device))
            next_obs, _, term, trun, infos = envs_eval.step(actions.squeeze(0).cpu().numpy())
            obs = next_obs
            if trun:
                break

        # Create gif with imageio
        with imageio.get_writer(
                os.path.join(f"./runs/{run_name}/eval_{iteration}.gif"),
                mode="I",
                fps=30) as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    deq = deque(maxlen=20)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    save_range = np.linspace(start=0.05 * args.num_iterations, stop=args.num_iterations, num=args.num_checkpoints,
                             dtype=np.uint32)
    print(run_name)
    print(save_range)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    if args.load_model:
        model_path = "C:/Users/SKG/GitHub/RL_MIPT/HW3/runs/"
        agent.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        deq.append(info["episode"]["r"])

                        run_mean = np.array(deq).mean()
                        run_std = np.array(deq).std()
                        writer.add_scalar("charts/running_mean", run_mean, global_step)
                        writer.add_scalar("charts/running_std", run_std, global_step)

                        if args.track:
                            wandb.log({"charts/episodic_return": info["episode"]["r"],
                                       "charts/episodic_length": info["episode"]["l"]},
                                      step=global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if args.track:
            wandb.log({"charts/learning_rate": optimizer.param_groups[0]["lr"],
                       "losses/value_loss": v_loss.item(),
                       "losses/policy_loss": pg_loss.item(),
                       "losses/entropy": entropy_loss.item(),
                       "losses/old_approx_kl": old_approx_kl.item(),
                       "losses/approx_kl": approx_kl.item(),
                       "losses/clipfrac": np.mean(clipfracs),
                       "losses/explained_variance": explained_var},
                      step=global_step)

        if iteration in save_range:
            model_path = f"runs/{run_name}/{args.exp_name}_{global_step}.tar"
            torch.save(agent.state_dict(), model_path)
            # evaluate(agent, run_name, global_step, args.seed)

    envs.close()
    writer.close()
