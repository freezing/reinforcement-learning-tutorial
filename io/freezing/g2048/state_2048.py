# Represents the state and actions for 2048 game

import numpy as np

HEIGHT_2048 = 4
WIDTH_2048 = 4


class State2048(object):
    """Represents one state of the 2048 game.
        Tiles board is a matrix of integers where integer K represents a value 2^K, for K > 0.
        Tiles board field is empty, for K = 0.
    """

    def __init__(self, height=HEIGHT_2048, width=WIDTH_2048, tiles=None):
        self.height = height
        self.width = width

        if tiles:
            self.tiles = tiles

        self.__new_game()

    def __new_game(self):
        """Initializes the tiles board by setting one random field to 1."""

        self.tiles = np.array((self.height, self.width))

        row = np.random.randint(0, self.height)
        col = np.random.randint(0, self.width)

        self.tiles[row, col] = 1

    

