from collections import defaultdict
from random import randint, random
import numpy as np


class TaxiAgent:
    """
        Map
        "|R: : : :G|",
        "| : : : : |",
        "| : : : : |",
        "| : : : : |",
        "|Y: : : :B|"


        Actions
        There are 6 discrete deterministic actions:
        - 0: move south
        - 1: move north
        - 2: move east
        - 3: move west
    """

    def __init__(self, nrows, ncols, nlocs):
        self.nrows = nrows
        self.ncols = ncols
        self.nlocs = nlocs

        self.actions = [0, 1, 2, 3]
        self.checks = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # decoder in x,y coordinates

        self.empty = 0
        self.non_empty = 1

        # generate pickup and dropoff location
        self.pickup = None
        self.dropoff = None
        self.start = None
        self.taxi = None
        self.passenger = False
        self.random_location()

        self.P = None
        self.generate_transition_matrix()

    def random_location(self):
        """Create 2 location of pickup and destination"""
        h = [x for x in range(self.nrows)]
        w = [y for y in range(self.ncols)]

        list_with_locations = []

        for n in range(self.nlocs):
            x = h.pop(randint(0, len(h) - 1))
            y = w.pop(randint(0, len(w) - 1))
            list_with_locations.append((x, y))

        self.pickup = list_with_locations[0]
        self.dropoff = list_with_locations[1]
        self.start = list_with_locations[2]

    def generate_transition_matrix(self):
        """
        create dictionary with all states and all 6 actions
        structure of state
        P[state][action] = [destination, reward, passenger, done]
        Initially everywhere reward = -1, done = False
        """

        # Define dictionary for transition matrix Assumptions: 1. Everywhere, reward are -1 2. Initially all action lead
        # to the same state, so we don't need separately define drop off, pickup and out of boundary

        self.P = dict()

        for r in range(self.nrows):
            for c in range(self.ncols):
                self.P[(r, c)] = dict()
                self.P[(r, c)][self.empty] = dict()
                self.P[(r, c)][self.non_empty] = dict()
                for a in self.actions:
                    self.P[(r, c)][self.empty][a] = [(r, c), -1, False]
                    self.P[(r, c)][self.non_empty][a] = [(r, c), -1, False]

        # define transition between states with only movement actions
        for r in range(self.nrows):
            for c in range(self.ncols):
                for a in self.actions:
                    x, y = self.checks[a]
                    if (r + x, c + y) in self.P:
                        if (r + x, c + y) == self.pickup:
                            self.P[r, c][self.empty][a] = [(r + x, c + y), +10, False]

                        elif (r + x, c + y) == self.dropoff:
                            self.P[r, c][self.non_empty][a] = [(r + x, c + y), +10, True]

                        else:
                            self.P[r, c][self.empty][a] = [(r + x, c + y), -1, False]
                            self.P[r, c][self.non_empty][a] = [(r + x, c + y), -1, False]

    def reset(self):
        self.taxi = self.start
        self.passenger = False
        return self.taxi

    def step(self, action):
        new_state, reward, terminated = self.P[self.taxi][int(self.passenger)][action]
        self.taxi = new_state

        if new_state == self.pickup:
            self.passenger = True

        return new_state, reward, terminated




