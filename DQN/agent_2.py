import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ---------- Q-Network (outputs: 1 discrete + n continuous) ----------
class QNetwork(nn.Module):
    def __init__(self, state_dim, phase_levels, num_clusters):
        super(QNetwork, self).__init__()
        self.num_clusters = num_clusters
        self.phase_levels = phase_levels
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1 + num_clusters)  # 1 for phase shift index, rest for power alloc
        )

    def forward(self, x):
        return self.model(x)

# ---------- DQN Agent ----------
class DQNAgent:
    def __init__(self, state_dim, phase_levels, num_clusters, total_power,
                 lr=1e-3, gamma=0.99, epsilon=1.0, decay=0.995,
                 epsilon_min=0.01, target_update_freq=100, buffer_size=10000):

        self.phase_levels = phase_levels
        self.num_clusters = num_clusters
        self.total_power = total_power
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

        self.model = QNetwork(state_dim, phase_levels, num_clusters)
        self.target = QNetwork(state_dim, phase_levels, num_clusters)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.train_step_counter = 0

    def select_action(self, state):
        """Selects a hybrid action: [phase_shift_idx, power_cluster_1, ..., power_cluster_n]"""
        if random.random() < self.epsilon:
            # Random action
            phase_shift_idx = random.randint(0, self.phase_levels - 1)
            random_power = np.random.rand(self.num_clusters)
            random_power = self.total_power * random_power / np.sum(random_power)
            return np.concatenate([[phase_shift_idx], random_power])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = self.model(state_tensor).squeeze(0)
            # Phase shift index (rounded and clamped)
            phase_shift_idx = int(torch.clamp(torch.round(output[0]), 0, self.phase_levels - 1).item())

            # Power allocations
            power_alloc = output[1:].detach().numpy()
            power_alloc = np.maximum(power_alloc, 0)
            if power_alloc.sum() > 0:
                power_alloc = self.total_power * power_alloc / np.sum(power_alloc)
            else:
                power_alloc = np.ones(self.num_clusters) * (self.total_power / self.num_clusters)

            return np.concatenate([[phase_shift_idx], power_alloc])

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = np.array(actions)

        # Split hybrid actions
        phase_indices = actions[:, 0].astype(int)
        power_allocs = actions[:, 1:]

        phase_indices_t = torch.tensor(phase_indices, dtype=torch.float32).unsqueeze(1)
        power_allocs_t = torch.tensor(power_allocs, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        outputs = self.model(states_t)
        phase_pred = outputs[:, 0].unsqueeze(1)
        power_pred = outputs[:, 1:]

        # Target calculation
        with torch.no_grad():
            next_outputs = self.target(next_states_t)
            next_phase = next_outputs[:, 0].unsqueeze(1)
            next_power = next_outputs[:, 1:]

        target_values = rewards_t + (~dones_t) * self.gamma * (next_phase + next_power.sum(dim=1, keepdim=True))  # Very rough approx

        # Compute loss
        phase_loss = nn.MSELoss()(phase_pred, phase_indices_t)
        power_loss = nn.MSELoss()(power_pred, power_allocs_t)
        loss = phase_loss + power_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

        # Target network update
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())

        return loss.item()