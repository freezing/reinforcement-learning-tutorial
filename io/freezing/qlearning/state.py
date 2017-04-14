# -*- coding: utf-8 -*-

import numpy as np

PLAYER_ARRAY = [0, 0, 0, 1]
WALL_ARRAY = [0, 0, 1, 0]
PIT_ARRAY = [0, 1, 0, 0]
GOAL_ARRAY = [1, 0, 0, 0]

class State(object):
    
    def __init__(self):
        self.state = np.zeros((4, 4, 4))
        self.__init_static()
        
        
    def __init_static(self):
        self.state[0, 1] = np.array(PLAYER_ARRAY)
        self.state[2, 2] = np.array(WALL_ARRAY)
        self.state[1, 1] = np.array(PIT_ARRAY)
        self.state[3, 3] = np.array(GOAL_ARRAY)
        
    def find_location(self, obj):
        for i in range(0, 4):
            for j in range(0, 4):
                if (self.state[i,j] == obj).all():
                    return i,j
        
    def display_grid(self):
        grid = np.zeros((4, 4), dtype='<U2')

        player_loc = self.find_location(PLAYER_ARRAY)
        wall_loc = self.find_location(WALL_ARRAY)
        pit_loc = self.find_location(PIT_ARRAY)
        goal_loc = self.find_location(GOAL_ARRAY)
        
        if player_loc:
            grid[player_loc] = 'P'
            
        if wall_loc:
            grid[wall_loc] = 'W'
            
        if pit_loc:
            grid[pit_loc] = '-'
                
        if goal_loc:
            grid[goal_loc] = '+'
            
        return grid
    