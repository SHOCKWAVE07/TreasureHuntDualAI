import pygame
import os
from environment.game_environment import GameEnvironment
from environment.constants import NUM_TURNS, GRID_SIZE
from agents.minimax_agent import MinimaxAgent
from agents.dqn_agent import TrainedDQNAgent
from environment.pygame_visualizer import PygameVisualizer
from utils.helpers import is_valid_move, log_game_stats

def main():
    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Initialize Environment
    env = GameEnvironment()

    # Calculate state size and action size
    state_size = GRID_SIZE * GRID_SIZE
    action_size = 4

    # Find the latest trained model
    model_path = None
    saved_models_dir = 'saved_models'
    
    # Check if saved_models directory exists and has model files
    if os.path.exists(saved_models_dir) and os.listdir(saved_models_dir):
        model_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.pth')]
        
        if model_files:
            # If multiple model files, choose the first one
            model_path = os.path.join(saved_models_dir, model_files[0])
            print(f"Using trained model: {model_path}")
        else:
            print("No trained model found in saved_models directory.")
            return
    else:
        print("saved_models directory does not exist or is empty.")
        return

    # Initialize Agents
    minimax_agent = MinimaxAgent(env)
    trained_dqn_agent = TrainedDQNAgent(env, state_size, action_size, model_path)

    # Visualizer
    visualizer = PygameVisualizer(env)

    # Game Loop
    running = True
    turn = 0
    max_turns = NUM_TURNS
    winner = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Alternate turns between Minimax and DQN agents
        if turn % 2 == 0:  # Minimax's turn
            move = minimax_agent.get_best_move()
            if move:
                env.minimax_player = move
                if env.check_treasure(env.minimax_player):
                    env.scores["Minimax"] += 1
        else:  # Trained DQN's turn
            state = env.get_state(env.dqn_player)
            action = trained_dqn_agent.select_action(state)
            move = env.get_next_position(env.dqn_player, action)
            
            if move and is_valid_move(*move):
                prev_player_pos = env.dqn_player
                env.dqn_player = move

                # Additional logging for DQN agent's move
                guard_distances = env.calculate_guard_distances(env.dqn_player)
                # print(f"Turn {turn}: DQN moved to {env.dqn_player}")
                # print(f"Guard distances: {guard_distances}")

                if env.check_treasure(env.dqn_player):
                    env.scores["DQN"] += 1

        # Check for collisions
        if env.check_collision(env.minimax_player):
            print("Minimax hit a guard! Game over.")
            winner = "DQN"
            running = False
        
        if env.check_collision(env.dqn_player):
            print("DQN hit a guard! Game over.")
            winner = "Minimax"
            running = False

        # Move guards
        env.move_guards()

        # Render the updated environment
        visualizer.render()

        # Check for game end conditions
        turn += 1
        if turn >= max_turns:
            running = False
            winner = (
                "Minimax" if env.scores["Minimax"] > env.scores["DQN"] else
                "DQN" if env.scores["DQN"] > env.scores["Minimax"] else "Draw"
            )

        clock.tick(2)  # Limit to 2 frames per second for visibility

    # End of game
    pygame.quit()
    print("Game Over!")
    print(f"Final Scores: {env.scores}")
    print(f"Winner: {winner}")

    # Log the game stats
    log_game_stats(env, winner, turn)

if __name__ == "__main__":
    main()