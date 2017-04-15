"""Dummy test script."""

from freezing.g2048.strategy_2048 import *
from freezing.g2048.player_2048 import *

strategy = min_preference_strategy
game = Game2048(seed=1234)

play(strategy, game, verbose=True)

# print(s.show())
# print()
#
# for action in range(NUMBER_OF_ACTIONS):
#     s.do_action(action)
#     print(ACTION_NAMES[action])
#     print(s.show())
#     print()