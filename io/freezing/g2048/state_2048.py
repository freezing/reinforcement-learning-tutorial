# Represents the state and actions for 2048 game

import numpy as np

HEIGHT_2048 = 4
WIDTH_2048 = 4

TILE_SCORES = [0, 2 * 1, 4 * 2, 8 * 3, 16 * 4, 32 * 5, 64 * 6, 128 * 7, 256 * 8, 512 * 9, 1024 * 10, 2048 * 11,
               4096 * 12, 8192 * 13, 16384 * 14, 32768 * 15]


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
        self.__calculate_total_score()

    def __new_game(self):
        """Initializes the tiles board by setting one random field to 1."""

        self.tiles = np.array((self.height, self.width))

        row = np.random.randint(0, self.height)
        col = np.random.randint(0, self.width)

        self.tiles[row, col] = 1

    def __calculate_total_score(self):
        """Total score can be approximated from the current state only (i.e. no need to know the previous actions).
           Formula is:
            total_score = sum of scores for each field
            field_score = N * Log_2 N

            E.g. 16 can be obtained by merging 8 and 8. At the point of merge, player receives 16 points.
            The same would happen for 2 eights which is in total 16, for 4 fours, which is in total 16.
            This means that we have 16 (N) for  16, 8, 4.

            NOTE: Obviously this is not 100% correct because the initial tiles can be 4s, but it's negligible.

            TILE_SCORES is a global list of tile scores.
        """
        self.total_score = 0

        for i in range(0, self.height):
            for j in range(0, self.width):
                field_score = TILE_SCORES[self.tiles[i, j]]
                self.total_score += field_score

    def reward(self):
        """Reward is the total score of the current state."""
        return self.total_score

    # TODO: Improve performance
    def run_action(self, action):
        """Runs the given action for the current state of the game.
        
            It's implemented by calling rotate matrix by 90 degrees (counter-clockwise / left) # of times per action. 
            After that, we apply MOVE UP method.
            Finally, rotate the matrix in the original state, which is (4 - action) % 4 times.
            
            Performance can be improved., but it's not priority at the moment.
        
            :arg action - UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3
                
            :return New game state after applied action.
                
            Note: run_action includes some randomness, i.e. after action is executed, random empty field gets
                a new tile. Tile 2 with probability 0.9 OR Tile 4 with probability 0.1.
        """
        normalized_tiles = np.rot90(self.tiles, action)

        # Runs the MOVE UP action in place
        State2048.__move_up(normalized_tiles)

        return State2048(tiles=normalized_tiles)

    @staticmethod
    def __move_up(tiles):

        return np.copy(tiles)

