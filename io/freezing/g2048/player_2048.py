"""A high level implementation of the play method that runs a strategy against the game."""

from freezing.g2048.game_2048 import ACTION_NAMES, NUMBER_OF_ACTIONS, Game2048
from freezing.g2048.experience_2048 import Experience2048


def play(strategy, game=None, allow_unavailable_actions=True, verbose=False):
    """Plays the game with the given strategy.

        :arg strategy: A function that takes as argument Game2048 and a list of available actions and
            returns an action from the list.

        :arg game: An optional state of the 2048 game as Game2048 instance.
            If None is provided then the new (random by default) Game2048 is initialized.

        :arg allow_unavailable_actions: Whether strategy is passed all actions or just available ones.

        :arg verbose: If True, prints game states, scores, actions.

        :return score, experience
            where score is the final score and experiences is the list of Experience instances.
    """

    if game is None:
        game = Game2048()

    state = game.state().copy()
    is_game_over = game.is_game_over()
    experiences = []

    while not is_game_over:
        if verbose:
            print('Score: {:d}'.format(game.score()))
            print(game.show())

        # By reference
        old_state = state

        list_of_actions = __generate_actions(game, allow_unavailable_actions)
        next_action = strategy(old_state, list_of_actions)

        if game.is_action_available(next_action):
            reward = game.do_action(next_action)
            state = game.state().copy()
            is_game_over = game.is_game_over()

            if verbose:
                print('Action: {:s}'.format(ACTION_NAMES[next_action]))
                print('Reward: {:d}'.format(reward))

            experiences.append(Experience2048(old_state, next_action, reward, state, is_game_over,
                                              False, game.available_actions()))
        else:
            experiences.append(Experience2048(old_state, next_action, 0, state, is_game_over,
                                              is_not_available=True,
                                              next_state_available_actions=game.available_actions()))

    if verbose:
        print('Score: {:d}'.format(game.score()))
        print(game.show())
        print('Game Over.')

    return game.score(), experiences


def __generate_actions(game, allow_unavailable_actions):
    """Returns actions to be passed to the strategy."""

    if allow_unavailable_actions:
        return range(NUMBER_OF_ACTIONS)
    else:
        return game.available_actions()
