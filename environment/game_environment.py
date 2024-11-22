import random
import numpy as np

GRID_SIZE = 10
NUM_TREASURES = 5
NUM_GUARDS = 4
NUM_TURNS = 50

class GameEnvironment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.minimax_player = [0, 0]
        self.dqn_player = [GRID_SIZE - 1, GRID_SIZE - 1]
        self.treasures = self.generate_positions(NUM_TREASURES)
        self.guards = self.generate_positions(NUM_GUARDS)

    def generate_positions(self, num):
        positions = []
        while len(positions) < num:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in positions:
                positions.append(pos)
        return positions

    def is_valid_move(self, x, y):
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE
