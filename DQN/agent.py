# dqn_optimizer.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ---------- Q-Network ----------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Consider making the network slightly deeper or wider if performance is low
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# ---------- DQN Agent ----------
class DQNAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=1e-3,              # Learning rate (can experiment with 1e-4, 3e-4)
                 gamma=0.99,           # Discount factor
                 epsilon=1.0,          # Initial epsilon
                 decay=0.995,          # Epsilon decay rate per training step
                 epsilon_min=0.01,     # Minimum epsilon value
                 target_update_freq=100, # How often to update the target network (steps)
                 buffer_size=10000):   # Size of the experience replay buffer

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

        # Initialize Networks
        self.model = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()  # Set target network to evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Memory
        self.memory = deque(maxlen=buffer_size)

        # Training step counter for target network updates
        self.train_step_counter = 0

    def select_action(self, state):
        """Selects an action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: choose the best action based on current Q-network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad(): # No need to track gradients for action selection
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        """Performs a single training step"""
        if len(self.memory) < batch_size:
            # Not enough experiences in memory to train
            return None # Return None or 0 for loss if needed

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, batch_size)
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert batch elements to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1) # Action indices need to be Long and shape [batch_size, 1] for gather
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        # --- Calculate Q-values for current states ---
        # Get Q(s, a) for the actions that were actually taken
        current_q_values = self.model(states_t).gather(1, actions_t)

        # --- Calculate Target Q-values ---
        # Get max Q(s', a') from target network
        with torch.no_grad(): # Target values are detached from the computation graph
            next_q_values_target = self.target(next_states_t).max(1, keepdim=True)[0] # [0] gets the max values

        # Calculate the target Q-value: R + gamma * max Q(s', a')
        # If done is true, the target is just the reward
        target_q_values = rewards_t + (~dones_t) * self.gamma * next_q_values_target

        # --- Calculate Loss ---
        # Using Mean Squared Error (MSE) loss, or Huber loss for robustness
        loss = nn.MSELoss()(current_q_values, target_q_values)
        # loss = nn.SmoothL1Loss()(current_q_values, target_q_values) # Huber loss

        # --- Optimize the Model ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping can help stabilize training
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # --- Decay Epsilon ---
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

        # --- Update Target Network Periodically ---
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            # print(f"Updating target network at step {self.train_step_counter}") # Optional debug print
            self.target.load_state_dict(self.model.state_dict())

        return loss.item() # Return the loss value for monitoring