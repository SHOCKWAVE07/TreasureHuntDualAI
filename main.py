from environment.game_environment import GameEnvironment
from agents.dqn_agent import DQNAgent
from agents.minimax_agent import MinimaxAgent
from environment.pygame_visualizer import visualize_game

def main():
    env = GameEnvironment()
    minimax_agent = MinimaxAgent(env)
    dqn_agent = DQNAgent(env)

    visualize_game(env, minimax_agent, dqn_agent)

if __name__ == "__main__":
    main()
