from utils.helpers import is_valid_move, manhattan_distance
import random
import numpy as np
from environment.constants import GRID_SIZE, NUM_TREASURES, NUM_GUARDS, GUARD_MOVES



class GameEnvironment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.treasures = []
        self.guards = []
        self.treasures = self.generate_positions(NUM_TREASURES)
        self.guards = self.generate_positions(NUM_GUARDS)
        self.minimax_player = [0, 0]
        self.dqn_player = [GRID_SIZE - 1, GRID_SIZE - 1]
        self.scores = {"Minimax": 0, "DQN": 0}
        self.visibility_radius = 2 

    def generate_positions(self, num):
        positions = []
        while len(positions) < num:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            # Ensure no conflicts with existing positions
            if pos not in positions and pos not in self.treasures and pos not in self.guards:
                positions.append(pos)
        return positions


    def move_guards(self):
        new_positions = set()  # To prevent guards from overlapping

        # Probabilities (can be tuned)
        stay_probability = 0.2  # 20% chance of staying still
        random_move_probability = 0.4  # 40% chance of moving randomly
        weighted_move_probability = 0.4  # 40% chance of moving toward objectives

        for guard in self.guards:
            action = random.choices(
                ["stay", "random_move", "weighted_move"],
                weights=[stay_probability, random_move_probability, weighted_move_probability],
                k=1
            )[0]

            if action == "stay":
                new_positions.add(guard)

            elif action == "random_move":
                # Randomly select a valid move
                valid_moves = [(guard[0] + dx, guard[1] + dy) for dx, dy in GUARD_MOVES if is_valid_move(guard[0] + dx, guard[1] + dy)]
                if valid_moves:
                    new_positions.add(random.choice(valid_moves))
                else:
                    new_positions.add(guard)  # Stay in place if no valid move

            elif action == "weighted_move":
                moves = []
                weights = []
                for dx, dy in GUARD_MOVES:
                    nx, ny = guard[0] + dx, guard[1] + dy
                    if is_valid_move(nx, ny) and (nx, ny) not in new_positions:
                        moves.append((nx, ny))
                        # Calculate weights for players and treasures
                        distance_to_minimax = manhattan_distance((nx, ny), self.minimax_player)
                        distance_to_dqn = manhattan_distance((nx, ny), self.dqn_player)
                        distance_to_nearest_treasure = (
                            min([manhattan_distance((nx, ny), t) for t in self.treasures]) if self.treasures else GRID_SIZE
                        )
                        # Higher weights for closer proximity to players/treasures
                        weight = 1 / (distance_to_minimax + 1) + 1 / (distance_to_dqn + 1) + 1 / (distance_to_nearest_treasure + 1)
                        weights.append(weight)

                if moves:
                    selected_move = random.choices(moves, weights=weights, k=1)[0]
                    new_positions.add(selected_move)
                else:
                    new_positions.add(guard)  # Stay in place if no valid weighted move

        self.guards = list(new_positions)


    def check_collision(self, player):
        return player in self.guards

    def check_treasure(self, player):
        if player in self.treasures:
            self.treasures.remove(player)
            return True
        return False

    def display(self):
        grid = np.full((GRID_SIZE, GRID_SIZE), "?", dtype=str)  # '?' for fog of war

        # Mark visible area for both players
        for player in [self.minimax_player, self.dqn_player]:
            px, py = player
            for x in range(px - self.visibility_radius, px + self.visibility_radius + 1):
                for y in range(py - self.visibility_radius, py + self.visibility_radius + 1):
                    if is_valid_move(x, y):
                        grid[x, y] = "."  # Default visible cell
                        if (x, y) in self.treasures:
                            grid[x, y] = "T"
                        elif (x, y) in self.guards:
                            grid[x, y] = "G"
                        if [x, y] == self.minimax_player:
                            grid[x, y] = "M"
                        elif [x, y] == self.dqn_player:
                            grid[x, y] = "D"

        print("\n".join([" ".join(row) for row in grid]))
        print(f"Scores: {self.scores}")