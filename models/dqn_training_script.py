import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.game_environment import GameEnvironment
from environment.constants import GRID_SIZE, NUM_TURNS
from utils.helpers import is_valid_move
import os

class DQNModel(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQNModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.fc(x)

class DQNTrainer:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, 
                 replay_size=10000, batch_size=64, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.target_update = target_update

        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)

        # Memory
        self.memory = []

        # Models
        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.update_target_model()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Training logs
        self.log_file = 'logs/dqn_training.txt'

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.replay_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return loss.item()

    def log_training_progress(self, episode, total_reward, loss):
        with open(self.log_file, 'a') as f:
            f.write(f"Episode: {episode}, Total Reward: {total_reward}, Loss: {loss}, Epsilon: {self.epsilon}\n")

    def save_model(self, filename='final_dqn_model.pth'):
        # Ensure saved_models directory exists
        os.makedirs('saved_models', exist_ok=True)
        
        # Save the model
        filepath = os.path.join('saved_models', filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

def train_dqn_agent(num_episodes=1000):
    # Initialize environment and trainer
    env = GameEnvironment()
    state_size = GRID_SIZE * GRID_SIZE
    action_size = 4  # Up, Down, Left, Right

    trainer = DQNTrainer(state_size, action_size)

    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        env.reset()
        state = env.get_state(env.dqn_player)
        total_reward = 0
        done = False
        turn = 0
        loss = 0

        while turn < NUM_TURNS and not done:
            # Select action
            action = trainer.select_action(state)
            
            # Get next position
            move = env.get_next_position(env.dqn_player, action)
            
            # Check if move is valid
            if move and is_valid_move(*move):
                prev_player_pos = env.dqn_player
                env.dqn_player = move

                # Check for treasure
                reward = 1 if env.check_treasure(env.dqn_player) else 0

                # Calculate distance penalty for proximity to guards
                guard_distances = env.calculate_guard_distances(env.dqn_player)
                proximity_penalty = 0
                for distance in guard_distances:
                    if distance <= 2:  # Within 2 grid units
                        proximity_penalty = -0.5 * (3 - distance)  # Penalty increases as distance decreases

                # Check for collision
                if env.check_collision(env.dqn_player):
                    reward = -10
                    done = True
                
                # Combine rewards
                reward += proximity_penalty

                # Get next state
                next_state = env.get_state(env.dqn_player)

                # Store experience
                trainer.store_experience(state, action, reward, next_state, done)

                # Update state and total reward
                state = next_state
                total_reward += reward

                # Train
                loss = trainer.train()

            # Move guards
            env.move_guards()
            turn += 1

        # Log progress
        trainer.log_training_progress(episode, total_reward, loss)

        # Periodically update target model
        if episode % trainer.target_update == 0:
            trainer.update_target_model()

    # Save final model
    trainer.save_model()

if __name__ == "__main__":
    train_dqn_agent()