# -*- coding: utf-8 -*-

import numpy as np

PLAYER_IDX = 3
WALL_IDX = 2
PIT_IDX = 1
GOAL_IDX = 0

BOARD_SIZE = 4

PLAYER_ARRAY = np.array([0, 0, 0, 1])
WALL_ARRAY = np.array([0, 0, 1, 0])
PIT_ARRAY = np.array([0, 1, 0, 0])
GOAL_ARRAY = np.array([1, 0, 0, 0])

MOVES_ARRAY = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class State(object):
    def __init__(self, state=None, height=BOARD_SIZE, width=BOARD_SIZE):
        self.height = height
        self.width = width

        if state is None:
            self.__init_static()
        else:
            self.state = state

        self.as_vector = self.state.reshape(1, height * width * 4)

    def __init_static(self):
        self.state = np.zeros((self.height, self.width, 4))

        self.state[0, 1] = PLAYER_ARRAY
        self.state[2, 2] = WALL_ARRAY
        self.state[1, 1] = PIT_ARRAY
        self.state[3, 3] = GOAL_ARRAY

    def __find_location(self, idx):
        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.state[i, j][idx] == 1:
                    return i, j

    def run_action(self, action):
        """
        Locates player in the grid and determines what object is in the target
        field:
            - Out of bounds: State doens't change
            - WALL: State doesn't change
            - Otherwise: Move player to the target field

        Returns new state, which represents the state after the action.
        """

        player_loc = self.__find_location(PLAYER_IDX)
        wall_loc = self.__find_location(WALL_IDX)

        target_field = tuple(np.array(player_loc) + np.array(MOVES_ARRAY[action]))

        new_state = np.copy(self.state)

        if self.__is_out(target_field) or (np.array(target_field) == wall_loc).all():
            return State(new_state)

        # Remove player
        new_state[player_loc] = new_state[player_loc] - PLAYER_ARRAY

        # Move it to the target field
        new_state[target_field] = new_state[target_field] + PLAYER_ARRAY

        return State(new_state)

    def __is_out(self, field):
        np_field = np.array(field)
        return not ((np_field >= (0, 0)).all() and (np_field < (self.height, self.width)).all())

    def reward(self):
        player_loc = self.__find_location(PLAYER_IDX)
        pit_loc = self.__find_location(PIT_IDX)
        goal_loc = self.__find_location(GOAL_IDX)

        if player_loc == pit_loc:
            return -10
        elif player_loc == goal_loc:
            return 10
        else:
            return -1

    def display_grid(self):
        grid = np.zeros((self.height, self.width), dtype='<U2')

        player_loc = self.__find_location(PLAYER_IDX)
        wall_loc = self.__find_location(WALL_IDX)
        pit_loc = self.__find_location(PIT_IDX)
        goal_loc = self.__find_location(GOAL_IDX)

        print("Player, Wall, Pit, Goal")
        print (player_loc)
        print(wall_loc)
        print(pit_loc)
        print(goal_loc)
        print()

        if player_loc:
            grid[player_loc] = 'P'

        if wall_loc:
            grid[wall_loc] = 'W'

        # TODO: Handle player is on Goal or Pit
        if pit_loc:
            grid[pit_loc] = '-'

        if goal_loc:
            grid[goal_loc] = '+'

        return grid

