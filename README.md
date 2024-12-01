# Treasure Hunt AI 🗺️🤖

A grid-based **Treasure Hunt Game** where two AI agents—**DQN Agent** and **Minimax Agent**—compete to collect treasures while avoiding guards. The project is implemented using **Pygame** and **PyTorch**.

---

## Features 🚀

- **Dynamic Gameplay**: Two AI agents navigate the grid to collect treasures.
- **DQN Agent**: Trained using Deep Q-Learning with replay memory.
- **Minimax Agent**: Uses the minimax algorithm to plan its moves.
- **Obstacles and Guards**: Both agents must avoid guards to survive.
- **Scoring System**: Tracks the performance of both agents.
- **Customizable Parameters**: Number of treasures, guards, grid size, and turns.

---

## Requirements 📦

- Python 3.7+
- Libraries: 
  - `torch`
  - `pygame`

---

## Installation 🛠️

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/TreasureHuntDualAI.git
   cd treasure-hunt-dqn-minimax
   ```

2. **Install dependencies**:
   Use `pip` to install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run 🏃‍♂️

### Run the Game
To play/watch the game, run:

```bash
python main.py
```

* The **DQN Agent** (trained model) will compete with the **Minimax Agent**.
* Results, including scores and game logs, will be displayed in the terminal and saved in `logs/game_stats.txt`.

### Retrain the DQN Agent
To retrain the DQN Agent:

```bash
python models/dqn_training.py
```

* Training logs are saved in `logs/training_log.txt`.
* The trained model is automatically saved for future gameplay.

## Game Rules 🕹️

1. **Objective**: Collect as many treasures as possible.
2. **Guards**:
   * If an agent collides with a guard, it loses the game immediately.
3. **Treasure Collection**:
   * Agents collect treasures for points. The agent with the highest score wins.
4. **Turns**:
   * The game ends after a fixed number of turns (configurable).

## File Structure 📂

```
TreasureHuntDualAI/
├── main.py                 # Main script to run the game
├── agents/                 # AI agent implementations
│   ├── __init__.py         # Init file for the agents package
│   ├── dqn_agent.py        # DQN agent implementation
│   ├── minimax_agent.py    # Minimax agent implementation
├── environment/            # Game environment and logic
│   ├── __init__.py         # Init file for the environment package
│   ├── constants.py        # Game-related constants
│   ├── game_environment.py # Core game environment logic
│   ├── pygame_visualizer.py# Pygame-based game visualization
├── logs/                   # Logs directory
│   ├── dqn_training.txt    # Logs for DQN training progress
│   ├── game_stats.txt      # Logs for game outcomes and scores
├── models/                 # DQN model training and management
│   ├── __init__.py         # Init file for the models package
│   ├── dqn_training.py     # Training script for the DQN agent
├── saved_models/           # Directory for saving trained DQN models
│   ├── final_dqn_model.pth # Trained DQN model checkpoint
├── utils/                  # Utility scripts
│   ├── __init__.py         # Init file for the utils package
│   ├── helpers.py          # Helper functions for game logic
├── README.md               # Project documentation
├── requirements.txt        # Packages required for the project
└── .gitignore              # Git configuration to exclude unnecessary files
```

## Customization ⚙️

You can customize the game settings in the `treasure_hunt.py` file:
* **Grid Size**: Change the `GRID_SIZE` variable.
* **Number of Treasures and Guards**: Adjust `NUM_TREASURES` and `NUM_GUARDS`.
* **Max Turns**: Modify the `NUM_TURNS` variable.

## Example Gameplay 🎮

* **DQN Agent** vs. **Minimax Agent** on a 10x10 grid.
* **Treasures**: 6
* **Guards**: 3
* **Turns**: 50

## Results 📊

* After training, the **DQN Agent** learns to avoid guards and optimize treasure collection.
* Training results are logged in `training_log.txt`.

## Future Improvements 🌟

* Add player vs. AI mode for interactive gameplay.
* Implement more advanced reinforcement learning algorithms (e.g., PPO, A3C).
* Introduce different types of obstacles and rewards.

## Contributing 🤝

Feel free to submit pull requests or open issues to improve the project!

## License 📜

This project is licensed under the MIT License.

## Credits 🙌

Created by **Varun Raskar**. Special thanks to the creators of PyTorch and Pygame!

### Steps for Usage:
1. Copy the entire content above into a README.md file.
2. Replace the placeholder **your-username** and the repository URL with your details.
3. Commit and push it to your GitHub repository. 🎉