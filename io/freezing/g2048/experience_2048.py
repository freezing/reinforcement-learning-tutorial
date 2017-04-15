"""A data-structure that encapsulates the experience of a single turn."""


class Experience2048(object):
    """A data-structure that encapsulates the experience of a single turn."""

    def __init__(self, state, action, reward, next_state, is_game_over,
                 is_not_available, next_state_available_actions):

        """Initialize Experience

          :arg state: (NUMBER_OF_ACTIONS, NUMBER_OF_ACTIONS) numpy array, the state before the action was executed.
          :arg action: Number in range(NUMBER_OF_ACTIONS), action that was taken.
          :arg reward: Number, experienced reward
          :arg next_state: (NUMBER_OF_ACTIONS, NUMBER_OF_ACTIONS) numpy array, the state after the action was
              executed
          :arg is_game_over: boolean, whether next_state is a terminal state
          :arg is_not_available: boolean, whether action was not available from state
          :arg next_state_available_actions: Available actions from the next state
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_game_over = is_game_over
        self.is_not_available = is_not_available
        self.next_state_available_actions = next_state_available_actions

    def __str__(self):
        return str((self.state, self.action, self.reward, self.next_state,
                    self.is_game_over, self.is_not_available, self.next_state_available_actions))

    def __repr__(self):
        return self.__str__()

