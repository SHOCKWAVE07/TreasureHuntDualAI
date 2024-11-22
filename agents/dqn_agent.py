import random
import torch
import numpy as np

class DQNAgent:
    def __init__(self, model, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.model = model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state, num_actions):
        if random.random() < self.epsilon:
            return random.randint(0, num_actions - 1)  # Explore
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()  # Exploit

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
