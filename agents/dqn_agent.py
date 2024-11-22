import random
from utils.helpers import is_valid_move
from environment.constants import GUARD_MOVES   


class DQNAgent:
    def __init__(self, env):
        self.env = env

    def get_best_move(self):
        # Placeholder: Random move for now
        x, y = self.env.dqn_player
        possible_moves = [(x + dx, y + dy) for dx, dy in GUARD_MOVES if is_valid_move(x + dx, y + dy)]
        return random.choice(possible_moves)