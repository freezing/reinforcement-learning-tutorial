"""Dummy test script."""

from freezing.g2048.strategy_2048 import *
from freezing.g2048.player_2048 import *

strategy = min_preference_strategy
game = Game2048()
score, experiences = play(strategy, game, verbose=True, allow_unavailable_actions=False)