# -*- coding: utf-8 -*-

import numpy as np

PLAYER_IDX = 3
WALL_IDX = 2
PIT_IDX = 1
GOAL_IDX = 0

PLAYER_ARRAY = np.array([0, 0, 0, 1])
WALL_ARRAY = np.array([0, 0, 1, 0])
PIT_ARRAY = np.array([0, 1, 0, 0])
GOAL_ARRAY = np.array([1, 0, 0, 0])

class State(object):
    
    def __init__(self):
        self.state = np.zeros((4, 4, 4))
        self.__init_static()
        
        
    def __init_static(self):
        self.state[0, 1] = PLAYER_ARRAY
        self.state[2, 2] = WALL_ARRAY
        self.state[1, 1] = PIT_ARRAY
        self.state[3, 3] = GOAL_ARRAY
            
    def __find_location(self, idx):
        for i in range(0, 4):
            for j in range(0, 4):
                if (self.state[i,j][idx] == 1):
                    return i,j
        
    def reward(self):
        player_loc = self.__find_location(PLAYER_ARRAY)
        
        
    def display_grid(self):
        grid = np.zeros((4, 4), dtype='<U2')

        player_loc = self.__find_location(PLAYER_IDX)
        wall_loc = self.__find_location(WALL_IDX)
        pit_loc = self.__find_location(PIT_IDX)
        goal_loc = self.__find_location(GOAL_IDX)
        
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
    