from utils.helpers import is_valid_move, manhattan_distance 
from environment.constants import GRID_SIZE,GUARD_MOVES # Ensure this import is at the top of the file

class MinimaxAgent:
    def __init__(self, env):
        self.env = env

    def heuristic(self, position):
        # Calculate distances to treasures and guards
        distances_to_treasures = [manhattan_distance(position, t) for t in self.env.treasures]
        distances_to_guards = [manhattan_distance(position, g) for g in self.env.guards]

        # Calculate scores
        treasure_score = -min(distances_to_treasures, default=GRID_SIZE)  # Closer to treasures is better
        guard_penalty = min(distances_to_guards, default=GRID_SIZE)      # Farther from guards is better
        visibility_bonus = self.env.visibility_radius - manhattan_distance(position, self.env.minimax_player)

        return treasure_score + guard_penalty + visibility_bonus

    def get_best_move(self):
        x, y = self.env.minimax_player
        best_move = None
        best_score = float("-inf")
        for dx, dy in GUARD_MOVES:
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny):
                score = self.heuristic((nx, ny))
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)
        return best_move