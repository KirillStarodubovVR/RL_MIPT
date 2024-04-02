import os
import random

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Model:
    def __init__(self, n_states, n_actions, seed):
        self.mask_state = np.zeros([n_states], dtype=int)
        self.mask_state_action = np.zeros([n_states, n_actions], dtype=int)
        self.r = np.zeros_like(self.mask_state_action, dtype=float)
        self.next_s = np.zeros_like(self.mask_state_action, dtype=int)
        self._rng = np.random.default_rng(seed)

    def add(self, s: int, a: int, r: float, next_s: int) -> float:
        self.mask_state[s] = 1
        self.mask_state_action[s][a] = 1
        self.r[s][a] = r
        self.next_s[s][a] = next_s
        return r

    def sample(self) -> tuple[int, int, float, int]:
        """
        returns s, a, r, next_s
        """
        s = self._rng.choice(np.where(self.mask_state > 0)[0])
        a = self._rng.choice(np.where(self.mask_state_action[s] > 0)[0])
        return s, a, self.r[s][a], self.next_s[s][a]


class DynaQAgent:
    def __init__(self, n_states, n_actions, lr, gamma, max_epsilon, min_epsilon, decay_rate, f_model, seed):
        self.Q = np.zeros((n_states, n_actions))
        self.model = f_model(n_states, n_actions, seed=seed)
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate

    def greedy_policy(self, state):
        # Exploitation: take the action with the highest state, action value
        action = np.argmax(self.Q[state][:])
        return action

    def epsilon_random_policy(self, state, i):
        """make new probability distribution of action"""
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * i)
        prob = [0.15, 0.15, 0.15, 0.15]
        if self._rng.uniform(0, 1) > epsilon:
            action = self.greedy_policy(state)
            prob[action] = 0.55
            action = random.choices(population=[0, 1, 2, 3], weights=prob, k=1)[0]
        else:
            action = env.action_space.sample()
            prob[action] = 0.55
            action = random.choices(population=[0, 1, 2, 3], weights=prob, k=1)[0]

        return action

    def epsilon_greedy_policy(self, state, i):

        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * i)

        if self._rng.uniform(0, 1) > epsilon:
            action = self.greedy_policy(state)
        else:
            action = env.action_space.sample()

        return action

    def update(self, s, a, r, s_n, update_model: bool):
        # Обновите модель, если нужно, реализуйте шаг Q-обучения
        if update_model:
            r = self.model.add(s, a, r, s_n)

        # получаем old_value (Q(s,a)) и next_max (max(Q(s', a')))
        Q_s_a = self.Q[s, a]
        V_sn = np.max(self.Q[s_n])

        td_error = r + self.gamma * V_sn - Q_s_a
        self.Q[s, a] += self.lr * td_error

    def dream(self, max_steps, **_):
        for _ in range(max_steps):
            m_s, m_a, m_r, m_next_s = self.model.sample()
            self.update(m_s, m_a, m_r, m_next_s, update_model=False)

def save_q_function(q_function, episode, path):
    """
    Convenient function to visualize the progress of training.
    """
    plt.figure(figsize=(12, 12))
    q = q_function.mean(axis=1).reshape((8, 8)).T
    sns.heatmap(q, annot=True, square=True, fmt=".2f", cbar=True)
    plt.grid()
    image_path = os.path.join(path, f"q_function_{str(episode).zfill(5)}.png")
    plt.savefig(image_path)
    plt.close("all")


def save_avg_return(avg_returns, episode, path):
    """
    Удобная функция, которая отображает прогресс обучения.
    """
    plt.figure(figsize=[12, 4])
    plt.subplot(1, 1, 1)
    plt.plot(*zip(*avg_returns), label='Mean return')
    plt.legend(loc=4)
    plt.grid()
    image_path = os.path.join(path, f"mean_return_{str(episode).zfill(5)}.png")
    plt.savefig(image_path)
    plt.close("all")


def create_gif(folder_path):
    # List the PNG files in the directory
    png_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

    # Sort the PNG files in ascending order
    png_files.sort()

    # Create a list to store the images
    images = []
    for filename in png_files:
        images.append(imageio.v3.imread(filename))

    # Output GIF file path
    output_gif = os.path.join(folder_path, "output.gif")

    # Save the images as a GIF
    imageio.mimsave(output_gif, images)
    print(f'GIF saved to: {output_gif}')


def train(env, agent, n_episodes, on_model_updates, seed, show_progress_schedule=50):
    path_average_return = "C:/Users/SKG/GitHub/RL_MIPT/HW4/videos/average_return"
    path_q_function = "C:/Users/SKG/GitHub/RL_MIPT/HW4/videos/q_function"
    path_to_video = "C:/Users/SKG/GitHub/RL_MIPT/HW4/videos/inference.gif"
    avg_returns, returns_batch = [], []
    rng = np.random.default_rng(seed)

    for i in range(1, n_episodes):
        state, _ = env.reset(seed=int(rng.integers(10000000)))
        reward, episode_return = 0, 0

        while True:

            action = agent.epsilon_greedy_policy(state, i)
            # action = agent.epsilon_random_policy(state, i)

            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, update_model=True)
            state = next_state
            episode_return += reward
            done = terminated or truncated
            if done:
                break

            agent.dream(on_model_updates, state=state)

        returns_batch.append(episode_return)

        if i % show_progress_schedule == 0:
            avg_returns.append((i, np.mean(returns_batch)))
            returns_batch = []

            save_avg_return(avg_returns, i, path_average_return)
            save_q_function(agent.Q, i, path_q_function)

            print(
                f"Episode: {i}, Return: {episode_return}, "
                f"AvgReturn[{show_progress_schedule}]: {avg_returns[-1][1]:.0f}, "
                f"Action: {action}"
            )

    create_gif(path_average_return)
    create_gif(path_q_function)
    record_video(env, agent.Q, path_to_video, fps=10)
    # draw_policy(env, agent.Q, gamma=0.9)

    return avg_returns


def record_video(env, Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 use 1)
    """
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)

    i = 0
    while not terminated or truncated:
        i += 1
        action = np.argmax(Qtable[state][:])

        # prob = [0.15, 0.15, 0.15, 0.15]
        # prob[action] = 0.55
        # action = random.choices(population=[0, 1, 2, 3], weights=prob, k=1)[0]

        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
        if i > 200:
            break

    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


# Training parameters
# n_training_episodes = 20000  # Total training episodes

desc = ["SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"]

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array", desc=desc)
env.reset(seed=47)
plt.imshow(env.render())

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")

agent = DynaQAgent(state_space,
                   action_space,
                   lr=0.7,
                   gamma=0.95,
                   max_epsilon=1.0,
                   min_epsilon=0.05,
                   decay_rate=0.0005,
                   f_model=Model,
                   seed=47
                   )

log_q = train(env, agent, n_episodes=5000, on_model_updates=20, seed=47, show_progress_schedule=50)
