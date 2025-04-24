# lstm_position_predictor.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------- LSTM Model for Position Prediction ----------
class PositionLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(PositionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# ---------- Trajectory Simulation ----------
def generate_user_trajectories(num_users=10, steps=50, area_size=20):
    """Simulate random walk for each user."""
    trajectories = []
    for _ in range(num_users):
        pos = np.random.rand(1, 2) * area_size
        path = [pos]
        for _ in range(steps - 1):
            movement = np.random.randn(1, 2) * 0.5  # small random step
            pos = pos + movement
            pos = np.clip(pos, 0, area_size)
            path.append(pos)
        trajectories.append(np.vstack(path))
    return np.array(trajectories)  # Shape: (num_users, steps, 2)

# ---------- Train LSTM on Simulated Trajectories ----------
def train_lstm_model(trajectories, seq_len=10, epochs=100, batch_size=16, lr=0.001):
    model = PositionLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X, Y = [], []
    for user_path in trajectories:
        for i in range(len(user_path) - seq_len):
            X.append(user_path[i:i+seq_len])
            Y.append(user_path[i+seq_len])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model

# ---------- Predict Future Positions ----------
def predict_positions(model, trajectories, seq_len=10, future_steps=10):
    model.eval()
    predicted = []
    with torch.no_grad():
        for user_path in trajectories:
            history = user_path[-seq_len:]
            future = []
            for _ in range(future_steps):
                inp = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0)
                next_pos = model(inp).squeeze(0).numpy()
                future.append(next_pos)
                history = np.vstack([history, next_pos])
            predicted.append(np.vstack(future))
    return np.array(predicted)  # Shape: (num_users, future_steps, 2)

# ---------- Visualization ----------
def plot_trajectories(original, predicted=None):
    plt.figure(figsize=(10, 8))
    for i, path in enumerate(original):
        plt.plot(path[:, 0], path[:, 1], label=f'User {i+1} - true', linestyle='--')
        if predicted is not None:
            plt.plot(predicted[i][:, 0], predicted[i][:, 1], label=f'User {i+1} - pred')
    plt.title("User Trajectories: True vs Predicted")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()