from collections import defaultdict
from random import randint, random
import numpy as np

MAP = [
    "+---------+",
    "|R: : : :G|",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "|Y: : : :B|",
    "+---------+",
]

"""  
    Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
"""

actions = [0, 1, 2, 3, 4, 5]


def random_location(num_rows, num_cols, num_locs) -> list:
    """Create 2 location of pickup and destination"""
    h = [x for x in range(num_rows)]
    w = [y for y in range(num_cols)]

    list_with_locations = []

    for n in range(num_locs):
        x = h.pop(randint(0, len(h) - 1))
        y = w.pop(randint(0, len(w) - 1))
        list_with_locations.append((x, y))

    return list_with_locations


def generate_field(num_rows=4, num_cols=4, num_locs=2):
    """generate environment field with location of pickup and final destination"""

    field = (num_rows, num_rows)
    pickup, dropoff = random_location(num_rows, num_cols, num_locs)

    return field, pickup, dropoff


field, pickup, dropoff = generate_field(3, 3, 2)


def generate_transition_matrix(field, pickup, dropoff, actions):
    """
    create dictionary with all states and all 6 actions
    structure of state
    P[state][action] = [destination, reward, passenger, done]
    Initially everywhere reward = -1, done = False
    """

    check = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # south, north, east, west

    # Define dictionary for transition matrix Assumptions: 1. Everywhere, reward are -1 2. Initially all action lead
    # to the same state, so we don't need separately define drop off, pickup and out of boundary

    P = dict()
    rows, cols = field
    for r in range(rows):
        for c in range(cols):
            P[(r, c)] = dict()
            for a in actions:
                P[(r, c)][a] = [(r, c), -1, False, False]

    # define transition between states with only movement actions
    for r in range(rows):
        for c in range(cols):
            for a in actions[:4]:
                x, y = check[a]
                if (r + x, c + y) in P:
                    if (r + x, c + y) == pickup:
                        # automatically pickup passenger without pickup
                        P[r, c][a] = [(r + x, c + y), -1, True, False]  # change in the future on -1

                    else:
                        P[r, c][a] = [(r + x, c + y), -1, False, False]

    # Define dropoff point
    P[dropoff][5] = [dropoff, +10, False, True]

    # Change passanger flag for taxi from False to True if taxi achieves pickup point without

    return P


def generate_transition_matrix_v2(field, pickup, dropoff, actions):
    """
    create dictionary with all states and all 6 actions
    structure of state
    P[state][action] = [destination, reward, passenger, done]
    Initially everywhere reward = -1, done = False
    """
    # decoder in x,y coordinates
    check = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # south, north, east, west

    # passenger flag (3d dimension)
    empty = 0
    non_empty = 1

    # Define dictionary for transition matrix Assumptions: 1. Everywhere, reward are -1 2. Initially all action lead
    # to the same state, so we don't need separately define drop off, pickup and out of boundary

    P = dict()
    rows, cols = field
    for r in range(rows):
        for c in range(cols):
            P[(r, c)] = dict()
            P[(r, c)][empty] = dict()
            for a in actions:
                P[(r, c)][empty][a] = [(r, c), -1, False]
                P[(r, c)][non_empty][a] = [(r, c), -1, False]

    # define transition between states with only movement actions
    for r in range(rows):
        for c in range(cols):
            for a in actions[:4]:
                x, y = check[a]
                if (r + x, c + y) in P:
                    if (r + x, c + y) == pickup:
                        # automatically pickup passenger without pickup
                        P[r, c][1][a] = [(r + x, c + y), -1, False]  # change in the future on -1

                    else:
                        P[r, c][0][a] = [(r + x, c + y), -1, False]

    # Define dropoff point
    P[dropoff][5] = [dropoff, +10, False, True]

    # Change passanger flag for taxi from False to True if taxi achieves pickup point without

    return P


"""  
    Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
"""

P = generate_transition_matrix(field, pickup, dropoff, actions)


# Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])

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
        action = randint(0, 5)

    return action
