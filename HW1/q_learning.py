import random
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from taxi_grid import TaxiAgent

agent = TaxiAgent(5, 5, 3)

# Extract data for Q-learning
state_space = len(agent.P.keys())
action_space = len(agent.actions)


def initialize_q_table(nrows, ncols, actions):

    Qtable = dict()
    for r in range(nrows):
        for c in range(ncols):
            Qtable[(r, c)] = dict()
            Qtable[(r, c)][0] = dict()
            Qtable[(r, c)][1] = dict()
            for a in actions:
                Qtable[(r, c)][0][a] = 0
                Qtable[(r, c)][0][1] = dict()

    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    q_list = list(Qtable[state].values())
    action = q_list.index(max(q_list))

    return action


def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = random.randint(0, action_space - 1)

    return action


def train(agent, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable):
    Qtable_plot = []
    episode_rewards = []
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # Reset the environment
        state = agent.reset()

        total_rewards_ep = 0

        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated = agent.step(action)
            total_rewards_ep += reward

            Qnew = max(list(Qtable[new_state].values()))

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                    reward + gamma * Qnew - Qtable[state][action])

            # Qtable_plot.append(Qtable.mean())
            # If terminated or truncated finish the episode
            if terminated:
                break

            # Our next state is the new state
            state = new_state

        episode_rewards.append(total_rewards_ep)

    return Qtable, Qtable_plot, episode_rewards


Qtable_taxi = initialize_q_table(agent.nrows, agent.ncols, agent.actions)

# Training parameters
n_training_episodes = 100000  # Total training episodes
learning_rate = 0.9  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

gif_frames = []

Qtable_taxi, Qtable_plot, episode_rewards = train(agent,
                                                  n_training_episodes,
                                                  min_epsilon,
                                                  max_epsilon,
                                                  decay_rate,
                                                  max_steps,
                                                  Qtable_taxi)

sns.lineplot(episode_rewards)
plt.show()