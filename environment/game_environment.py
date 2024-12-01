import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
            if pos not in positions and pos not in self.treasures and pos not in self.guards:
                positions.append(pos)
        return positions

    def move_guards(self):
        new_positions = set()
        stay_probability = 0.2
        random_move_probability = 0.4
        weighted_move_probability = 0.4

        for guard in self.guards:
            action = random.choices(
                ["stay", "random_move", "weighted_move"],
                weights=[stay_probability, random_move_probability, weighted_move_probability],
                k=1
            )[0]

            if action == "stay":
                new_positions.add(guard)

            elif action == "random_move":
                valid_moves = [(guard[0] + dx, guard[1] + dy) for dx, dy in GUARD_MOVES if is_valid_move(guard[0] + dx, guard[1] + dy)]
                if valid_moves:
                    new_positions.add(random.choice(valid_moves))
                else:
                    new_positions.add(guard)

            elif action == "weighted_move":
                moves = []
                weights = []
                for dx, dy in GUARD_MOVES:
                    nx, ny = guard[0] + dx, guard[1] + dy
                    if is_valid_move(nx, ny) and (nx, ny) not in new_positions:
                        moves.append((nx, ny))
                        distance_to_minimax = manhattan_distance((nx, ny), self.minimax_player)
                        distance_to_dqn = manhattan_distance((nx, ny), self.dqn_player)
                        distance_to_nearest_treasure = (
                            min([manhattan_distance((nx, ny), t) for t in self.treasures]) if self.treasures else GRID_SIZE
                        )
                        weight = 1 / (distance_to_minimax + 1) + 1 / (distance_to_dqn + 1) + 1 / (distance_to_nearest_treasure + 1)
                        weights.append(weight)

                if moves:
                    selected_move = random.choices(moves, weights=weights, k=1)[0]
                    new_positions.add(selected_move)
                else:
                    new_positions.add(guard)

        self.guards = list(new_positions)

    def check_collision(self, player):
        return player in self.guards

    def check_treasure(self, player):
        if player in self.treasures:
            self.treasures.remove(player)
            return True
        return False

    def display(self):
        grid = np.full((GRID_SIZE, GRID_SIZE), "?", dtype=str)
        for player in [self.minimax_player, self.dqn_player]:
            px, py = player
            for x in range(px - self.visibility_radius, px + self.visibility_radius + 1):
                for y in range(py - self.visibility_radius, py + self.visibility_radius + 1):
                    if is_valid_move(x, y):
                        grid[x, y] = "."
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

    def calculate_guard_distances(self, player_pos):
        """
        Calculate Manhattan distances between player and all guards.
        
        :param player_pos: Current player position (x, y)
        :return: List of distances to each guard
        """
        distances = []
        for guard in self.guards:
            # Calculate Manhattan distance
            distance = abs(player_pos[0] - guard[0]) + abs(player_pos[1] - guard[1])
            distances.append(distance)
        return distances

    def get_state(self, player_position):
    # Initialize a grid with zeros
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Place player on the grid
        grid[player_position[0], player_position[1]] = 1  # Player is represented as 1

        # Place treasures on the grid
        for treasure in self.treasures:
            grid[treasure[0], treasure[1]] = 2  # Treasures are represented as 2

        # Place guards on the grid
        for guard in self.guards:
            grid[guard[0], guard[1]] = 3  # Guards are represented as 3

        # Flatten the grid into a 1D vector
        return grid.flatten()


    def get_next_position(self, player, action):
        x, y = player
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = actions[action]
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            return (new_x, new_y)
        else:
            return None

    def update_position(self, player, action):
        next_position = self.get_next_position(player, action)
        if next_position is not None:
            player[0], player[1] = next_position

    def initialize_game(self):
        self.grid = self.create_grid()
        self.dqn_player = [GRID_SIZE - 1, GRID_SIZE - 1]
        self.minimax_player = [0, 0]
        self.guards = self.generate_positions(NUM_GUARDS)
        self.treasures = self.generate_positions(NUM_TREASURES)
        self.scores = {"DQN": 0, "Minimax": 0}

    def reset(self):
        self.initialize_game()
        return self.get_state(self.dqn_player)

    def step(self, action):
        self.update_position(self.dqn_player, action)
        reward = 0
        if self.check_treasure(self.dqn_player):
            self.scores["DQN"] += 1
            reward += 10
        self.move_guards()
        if self.check_collision(self.dqn_player):
            return self.get_state(self.dqn_player), -10, True
        if not self.treasures:
            return self.get_state(self.dqn_player), reward + 50, True
        return self.get_state(self.dqn_player), reward, False

    def create_grid(self):
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
