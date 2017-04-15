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

    def __init__(self, tiles=None):
        if tiles is not None:
            self.tiles = tiles
            self.height, self.width = tiles.shape[0], tiles.shape[1]
        else:
            self.height = WIDTH_2048
            self.width = WIDTH_2048

            self.__new_game()

        self.__calculate_total_score()
        self.as_vector = self.tiles.reshape(1, self.height * self.width)

    def is_game_over(self):
        """Checks if there are any available moves"""
        return len(self.available_actions()) == 0

    def available_actions(self):
        actions = []
        for action in range(4):
            if self.__is_available_action(action):
                actions.append(action)
        return actions

    def __is_available_action(self, action):
        rotated = np.rot90(self.tiles, action)
        return State2048.__is_available_left(rotated)

    def __new_game(self):
        """Initializes the tiles board by setting one random field to 1."""

        self.tiles = np.zeros((self.height, self.width), dtype=np.int)
        self.__add_random_tile()
        self.__add_random_tile()

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

    def pretty_print(self):
        """Returns the human-readable representation of the state"""

        def tile_string(value):
            """Convert value to string."""
            if value > 0:
                return '{: 5d}'.format(2 ** value)
            return '     '

        horizontal_line = "-" * (6 * self.width + 1)

        pretty_string = horizontal_line

        for row in range(self.height):
            values_string = '|'.join([tile_string(v) for v in self.tiles[row, :]])
            row_string = '|{:s}|'.format(values_string)

            pretty_string += '\n{:s}'.format(row_string)
            pretty_string += '\n{:s}'.format(horizontal_line)
        return pretty_string

    def run_action(self, action):
        """Runs the given action for the current state of the game.
        
            It's implemented by calling rotate matrix by 90 degrees (counter-clockwise / left) # of times per action. 
            After that, we apply MOVE LEFT method.
            Finally, rotate the matrix in the original state, by -action.
        
            :arg action - LEFT = 0, UP = 1, RIGHT = 2, DOWN = 3
                
            :return New game state after applied action.
                
            Note: run_action includes some randomness, i.e. after action is executed, random empty field gets
                a new tile. Tile 2 with probability 0.9 OR Tile 4 with probability 0.1.
        """
        if not self.__is_available_action(action):
            return self

        normalized_tiles = np.rot90(self.tiles, action)

        # Runs the MOVE LEFT action in place
        State2048.__move_left(normalized_tiles)

        state = State2048(tiles=np.rot90(normalized_tiles, -action))
        state.__add_random_tile()
        return state

    def __add_random_tile(self):
        """Adds a random tile to the grid at random empty position.
           
           Tile 2 is added with probability 0.9.
           Tile 4 is added with probability 0.1.
        """
        row_positions, col_positions = np.where(self.tiles == 0)
        assert len(row_positions) > 0

        empty_index = np.random.choice(len(row_positions))
        tile_value = np.random.choice([1, 2], p=[0.9, 0.1])

        self.tiles[row_positions[empty_index], col_positions[empty_index]] = tile_value

    @staticmethod
    def __is_available_left(tiles):
        """Checks if LEFT is valid move (it will change the state)."""
        height, width = tiles.shape[0], tiles.shape[1]

        for row in range(height):
            empty_found = False
            for col in range(width):
                empty_found |= tiles[row][col] == 0

                # Check move to the empty cell
                if empty_found and tiles[row][col] != 0:
                    return True

                # Check merge
                if col > 0 and tiles[row][col] != 0 and tiles[row][col - 1] == tiles[row][col]:
                    return True

        return False

    @staticmethod
    def __move_left(tiles):
        """Moves the tiles LEFT.
        
          NOTE: It runs in-place so the tiles array will be modified.
        """

        row_count = tiles.shape[0]

        for row in range(row_count):
            State2048.__move_left_row(tiles[row])

        return tiles

    @staticmethod
    def __move_left_row(tiles_row):
        """Moves the tiles in a single row to the LEFT.
            
           NOTE: It runs in-place so the tiles array will be modified.
        """

        length = tiles_row.shape[0]

        # Represents the index of the interesting column (that is the column of the tile that can be merged;
        # or the column of the most left empty field)
        last_idx = 0
        # NOTE: Starting from 1 (skipping first cell)
        for col in range(1, length):
            # Is there anything to move? Skip if not.
            if tiles_row[col] == 0:
                continue

            # Can I move the current cell to the left?
            # There are two cases:
            #   - last_idx is empty cell, so I just move it there
            #   - last_idx cell has the same value, so I merge them
            # In both cases, last_idx cell becomes the next one
            #
            # If I can't move it to the last_idx, that means I can't move this cell at all
            # In that case last_idx == col - 1 (assert that), and it becomes col

            if tiles_row[last_idx] == 0:
                # Empty - this is the next candidate for marge now, so don't increase last_idx
                tiles_row[last_idx] = tiles_row[col]
                tiles_row[col] = 0
            elif tiles_row[last_idx] == tiles_row[col]:
                # Merge them
                tiles_row[last_idx] += 1
                tiles_row[col] = 0

                # We don't want to consider this anymore, move to the next cell
                last_idx += 1
            else:
                last_idx += 1

                if tiles_row[last_idx] == 0:
                    tiles_row[last_idx] = tiles_row[col]
                    tiles_row[col] = 0
                else:
                    assert last_idx == col

# st = np.zeros((4, 4), dtype=np.int)
# st[0, 0] = 2
# st[0, 1] = 1
# st[0, 2] = 3
# s = State2048(st)
# s.run_action(0)
#
# s = State2048()
# print(s.pretty_print())
# print()
#
# s1 = s.run_action(0)
# print(s1.pretty_print())
# print()
#
# s2 = s1.run_action(1)
# print(s2.pretty_print())
# print()
#
# s3 = s2.run_action(2)
# print(s3.pretty_print())
# print()
#
# s4 = s3.run_action(3)
# print(s3.pretty_print())
# print()

