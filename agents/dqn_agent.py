# import random
# from collections import deque
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Neural network for approximating Q-values
# class DQNModel(nn.Module):
#     def __init__(self, input_size, num_actions):
#         super(DQNModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_actions)
#         )

#     def forward(self, x):
#         return self.fc(x)

# class DQNAgent:
#     def __init__(self, env, state_size, action_size, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, replay_size=1000, batch_size=32, target_update=10):
#         self.env = env
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.lr = lr
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_min = epsilon_min
#         self.replay_size = replay_size
#         self.batch_size = batch_size
#         self.target_update = target_update

#         # Replay buffer
#         self.memory = deque(maxlen=self.replay_size)

#         # Models
#         self.model = DQNModel(state_size, action_size)
#         self.target_model = DQNModel(state_size, action_size)
#         self.update_target_model()

#         # Optimizer
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#         self.loss_fn = nn.MSELoss()

#         # Step counter for target updates
#         self.step_count = 0

#     def update_target_model(self):
#         self.target_model.load_state_dict(self.model.state_dict())

#     def select_action(self, state):
#         # Epsilon-greedy action selection
#         if random.random() < self.epsilon:
#             return random.randint(0, self.action_size - 1)  # Explore
#         else:
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#             with torch.no_grad():
#                 q_values = self.model(state_tensor)
#             return q_values.argmax().item()  # Exploit

#     def store_experience(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def train(self):
#         if len(self.memory) < self.batch_size:
#             return

#         # Sample a batch from the replay buffer
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         states = torch.tensor(states, dtype=torch.float32)
#         actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
#         rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
#         next_states = torch.tensor(next_states, dtype=torch.float32)
#         dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

#         # Compute target Q-values
#         with torch.no_grad():
#             max_next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
#             target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

#         # Compute current Q-values
#         current_q_values = self.model(states).gather(1, actions)

#         # Compute loss and optimize
#         loss = self.loss_fn(current_q_values, target_q_values)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # Update epsilon (exploration rate)
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

#         # Update target model periodically
#         self.step_count += 1
#         if self.step_count % self.target_update == 0:
#             self.update_target_model()

import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNModel(torch.nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQNModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.fc(x)

class TrainedDQNAgent:
    def __init__(self, env, state_size, action_size, model_path):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        
        # Load the trained model
        self.model = DQNModel(state_size, action_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode

    def select_action(self, state):
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        # Return the action with the highest Q-value
        return q_values.argmax().item()