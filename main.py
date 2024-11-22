import pygame
from environment.game_environment import GameEnvironment
from environment.constants import NUM_TURNS
from agents.minimax_agent import MinimaxAgent
from agents.dqn_agent import DQNAgent
from environment.pygame_visualizer import PygameVisualizer
import numpy as np  


def decode_action(action):
    """Map action index to grid movement (dx, dy)."""
    action_mapping = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }
    return action_mapping[action]

def get_state_representation(env, player_type):
    """
    Get the state representation for the DQN player.
    Flatten the grid and encode the player's position.
    """
    state = env.grid.flatten()
    if player_type == "dqn":
        state = np.append(state, env.dqn_player)
    elif player_type == "minimax":
        state = np.append(state, env.minimax_player)
    return state

def main():
    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()
    
    # Initialize Environment and Agents
    env = GameEnvironment()
    minimax_agent = MinimaxAgent(env)
    dqn_agent = DQNAgent(env)
    visualizer = PygameVisualizer(env)

    # Game Loop
    running = True
    turn = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Stop the game after reaching the turn limit
        if turn >= NUM_TURNS:
            print("Turn limit reached! Game over.")
            break

        # Alternate turns between Minimax and DQN agents
        minimax_move = minimax_agent.get_best_move()
        if minimax_move:
            env.minimax_player = minimax_move
        if env.check_collision(env.minimax_player):
            print("Minimax hit a guard!")
            break
        if env.check_treasure(env.minimax_player):
            env.scores["Minimax"] += 1

        # DQN Agent Move
        dqn_move = dqn_agent.get_best_move()
        if dqn_move:
            env.dqn_player = dqn_move
        if env.check_collision(env.dqn_player):
            print("DQN hit a guard!")
            break
        if env.check_treasure(env.dqn_player):
            env.scores["DQN"] += 1

        # Move Guards
        env.move_guards()

        # Render the updated environment
        visualizer.render()

        # Advance the turn
        turn += 1
        clock.tick(2)  # Limit to 2 frames per second for visibility

    pygame.quit()
    print("Game Over!")
    print(f"Final Scores: {env.scores}")

if __name__ == "__main__":
    main()
