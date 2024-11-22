from utils.helpers import manhattan_distance

class MinimaxAgent:
    def __init__(self, env):
        self.env = env

    def heuristic(self, position):
        distances_to_treasures = [manhattan_distance(position, t) for t in self.env.treasures]
        treasure_score = -min(distances_to_treasures, default=0)
        return treasure_score

    def get_best_move(self):
        x, y = self.env.minimax_player
        best_move = None
        best_score = float('-inf')
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.env.is_valid_move(nx, ny):
                score = self.heuristic((nx, ny))
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)
        return best_move
