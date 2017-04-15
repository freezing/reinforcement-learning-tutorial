"""Algorithms and strategies to play 2048."""

import numpy as np
from freezing.g2048.game_2048 import *


def min_preference_strategy(_, actions):
    """Strategy that prefers LEFT over TOP over RIGHT over BOTTOM."""
    return min(actions)


def highest_reward_strategy(state, actions):
    """Strategy that always chooses the action of highest immediate reward.
    If there are any ties, the strategy prefers left over up over right over down.
    """

    sorted_actions = np.sort(actions)[::-1]
    rewards = map(lambda action: Game2048(np.copy(state)).do_action(action), sorted_actions)
    action_index = np.argsort(rewards, kind="mergesort")[-1]
    return sorted_actions[action_index]


def random_strategy(_, actions):
    """Strategy that always chooses actions at random.

        :arg actions: List of available actions.

        :return Random action from the list of available actions.
    """

    return np.random.choice(actions)


def make_greedy_strategy(q_values_provider, verbose=False):
    """Makes greedy strategy.

        :arg q_values_provider: A function that returns list of Q-values for the given state.
        :arg verbose: If True, prints state, Q-values per action and taken action.
    """

    def greedy_strategy(state, actions):
        """A strategy that always picks an action of maximum Q(state, action).

            :arg state: Representation of the Game2048 inner state to run the strategy against.
            :arg actions: List of available actions.

            :return An action from the available actions to be played.
        """

        q_values = q_values_provider(state)

        if verbose:
            print("State:")
            print(state)
            print("Q-Values:")
            for action, q_value, action_name in zip(range(NUMBER_OF_ACTIONS), q_values, ACTION_NAMES):
                not_available_string = "" if action in actions else "(not available)"
                print("{:s}:\t{:.2f} {:s}".format(action_name, q_value, not_available_string))

        sorted_actions = np.argsort(q_values)
        action = [a for a in sorted_actions if a in actions][-1]

        if verbose:
            print("-->", ACTION_NAMES[action])

        return action

    return greedy_strategy


def make_epsilon_greedy_strategy(q_values_provider, epsilon):
    """Makes epsilon_greedy_strategy.

        :arg q_values_provider: A function that returns list of Q-values for the given state.
        :arg epsilon: A probability to pick a random action.
    """

    greedy_strategy = make_greedy_strategy(q_values_provider)

    def epsilon_greedy_strategy(state, actions):
        """Picks a random action with probability epsilon, otherwise uses greedy_strategy.

            :arg state: Representation of the Game2048 inner state to run the strategy against.
            :arg actions: List of available actions.

            :return An action from the available actions to be played.
        """

        do_random_action = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
        if do_random_action:
            return random_strategy(state, actions)
        else:
            return greedy_strategy(state, actions)

    return epsilon_greedy_strategy


