"""
    Contains the implementation class of the 2048 game.

    Implemented according to: https://github.com/georgwiese/2048-rl/blob/master/py_2048_rl/game/game.py
"""

import numpy as np
from numpy.random import RandomState

DEFAULT_BOARD_SIZE = 4
NUMBER_OF_ACTIONS = 4

ACTION_NAMES = ["left", "up", "right", "down"]
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


class Game2048(object):
    """Implements the 2048 Game logic.

    Game states are represented as (4, 4) numpy arrays,
    and its values are 0 for empty fields and log_2(value) for any tiles.
    """

    def __init__(self, state=None, initial_score=0, seed=None):
        """Initialize the Game2048 object.

        :arg state: (4, 4) numpy array to initialize the state with. If None, the game will be initialized with
                two random tiles.

        :arg seed: int, array like or optional, used to initialize RandomState.
            Must be convertible to 32 bit unsigned int.

        :arg: initial_score: score to initialize the game with.
        """

        self._score = initial_score

        if seed is None:
            self._random = RandomState()
        else:
            self._random = RandomState(seed=seed)

        if state is None:
            self._state = np.zeros((DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE), dtype=np.int)
            self.__add_random_tile()
            self.__add_random_tile()
        else:
            self._state = state

    def state(self):
        return self._state

    def score(self):
        return self._score

    def copy(self):
        """Returns a copy of itself."""

        return Game2048(state=self._state, initial_score=self._score)

    def is_game_over(self):
        """Returns whether or not the game is over."""

        for action in range(NUMBER_OF_ACTIONS):
            if self.is_action_available(action):
                return False

        return True

    def available_actions(self):
        """Returns the list of actions that are available."""

        return [action for action in range(NUMBER_OF_ACTIONS) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determines if action is action is available, i.e. it will change the state."""

        temp_state = np.rot90(self._state, action)
        return Game2048._is_action_available_left(temp_state)

    def do_action(self, action):
        """Execute the action, add a new tile, update the score and return reward."""

        temp_state = np.rot90(self._state, action)
        reward = Game2048._do_action_left(temp_state)
        self._state = np.rot90(temp_state, -action)
        self._score += reward

        self.__add_random_tile()

        return reward

    def __add_random_tile(self):
        """Adds random tile to the grid. Assumes that it has empty field."""

        row_positions, col_positions = np.where(self._state == 0)
        assert len(row_positions) > 0

        empty_index = self._random.choice(len(row_positions))
        value = self._random.choice([1, 2], p=[0.9, 0.1])

        self._state[row_positions[empty_index], col_positions[empty_index]] = value

    def show(self):
        """Returns the human-readable string representation of the state."""

        def tile_string(value):
            """Convert value to string."""
            if value > 0:
                return '{: 5d}'.format(2 ** value)
            return '     '

        width, height = self._state.shape

        horizontal_line = "-" * (6 * width + 1)

        pretty_string = horizontal_line

        for row in range(height):
            values_string = '|'.join([tile_string(v) for v in self._state[row, :]])
            row_string = '|{:s}|'.format(values_string)

            pretty_string += '\n{:s}'.format(row_string)
            pretty_string += '\n{:s}'.format(horizontal_line)
        return pretty_string

    @staticmethod
    def _is_action_available_left(state):
        """Determines whether action 'LEFT' is available given the state."""

        height, width = state.shape
        for row in range(height):
            has_empty = False
            for col in range(width):
                has_empty |= state[row, col] == 0
                if state[row, col] != 0 and has_empty:
                    return True

                if (state[row, col] != 0 and col > 0 and
                        state[row, col] == state[row, col - 1]):
                    return True
        return False

    @staticmethod
    def _do_action_left(state):
        """Executes action left on the given state and returns reward."""

        height, width = state.shape
        reward = 0

        for row in range(height):
            # Always the rightmost tile in the current row that was already moved
            merge_candidate = -1
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                if state[row, col] == 0:
                    continue

                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                            state[row, merge_candidate] == state[row, col]):
                    # Merge tile with merge_candidate
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile to the left
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward
