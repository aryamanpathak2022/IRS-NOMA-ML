
from LSTM.lstm_predictor import *      
from CLUSTERING.kgmm_clustering import *
from DQN.agent import *
from DQN.noma_env import *
from DQN.oma_env import *




# Step 1: Simulate
trajectories = generate_user_trajectories(num_users=10, steps=50)

# Step 2: Train LSTM
lstm_model = train_lstm_model(trajectories, seq_len=10)

# Step 3: Predict future positions
predicted = predict_positions(lstm_model, trajectories, future_steps=10)

# Step 4: Visualize
plot_trajectories(trajectories[:, -10:, :], predicted)

# Use final predicted positions for clustering (shape: num_users x 2)
last_pred_positions = predicted[:, -1, :]  # Last predicted step for each user

# Perform clustering
labels, centers = k_gmm_clustering(last_pred_positions, n_clusters=5)

# Visualize
plot_clusters(last_pred_positions, labels, centers)




# -----------------plot---------------------

# 1. Sum Rate vs Transmission Power with Different Iterations
powers = [5, 10, 20, 30, 40, 50]
iterations = [100, 300, 500, 700, 1000]
avg_sum_rates = {iteration: [] for iteration in iterations}

for iteration in iterations:
    for p in powers:
        env = IRSNOMAEnv(signal_power=p)
        agent = DQNAgent(env.state_dim, env.action_dim)

        state = env.reset()
        total_reward = 0
        for _ in range(iteration):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break

        avg_sum_rates[iteration].append(total_reward / iteration)

# Plotting Sum Rate vs Transmission Power with Different Iterations
plt.figure(figsize=(8, 6))
for iteration in iterations:
    plt.plot(powers, avg_sum_rates[iteration], label=f"{iteration} Iterations", marker='o')
plt.xlabel("Transmit Power (dBm)")
plt.ylabel("Average Sum Rate (bps/Hz)")
plt.title("Figure 3: Sum Rate vs Transmit Power with Different Iterations (IRS-NOMA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Sum Rate vs IRS Elements with Different Powers
elements = [5, 10, 15, 20, 25, 30]
avg_sum_rates = {p: [] for p in powers}

for p in powers:
    for n in elements:
        env = IRSNOMAEnv(num_elements=n, signal_power=p)
        agent = DQNAgent(env.state_dim, env.action_dim)

        state = env.reset()
        total_reward = 0
        for _ in range(300):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break

        avg_sum_rates[p].append(total_reward / 300)

# Plotting Sum Rate vs IRS Elements with Different Powers
plt.figure(figsize=(8, 6))
for p in powers:
    plt.plot(elements, avg_sum_rates[p], label=f"Power {p} dBm", marker='s')
plt.xlabel("Number of IRS Elements")
plt.ylabel("Average Sum Rate (bps/Hz)")
plt.title("Figure 4: Sum Rate vs IRS Elements with Different Powers (IRS-NOMA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Sum Rate vs IRS Elements (NOMA vs OMA) with Different Powers
noma_avg_sum_rates = {p: [] for p in powers}
oma_avg_sum_rates = {p: [] for p in powers}

for p in powers:
    for n in elements:
        # NOMA
        env_noma = IRSNOMAEnv(num_elements=n, signal_power=p, noma=True)
        agent_noma = DQNAgent(env_noma.state_dim, env_noma.action_dim)

        state = env_noma.reset()
        total_reward_noma = 0
        for _ in range(300):
            action = agent_noma.select_action(state)
            next_state, reward, done = env_noma.step(action)
            agent_noma.store(state, action, reward, next_state, done)
            agent_noma.train()
            state = next_state
            total_reward_noma += reward
            if done:
                break
        noma_avg_sum_rates[p].append(total_reward_noma / 300)

        # OMA
        env_oma = IRSNOMAEnv(num_elements=n, signal_power=p, noma=False)
        agent_oma = DQNAgent(env_oma.state_dim, env_oma.action_dim)

        state = env_oma.reset()
        total_reward_oma = 0
        for _ in range(300):
            action = agent_oma.select_action(state)
            next_state, reward, done = env_oma.step(action)
            agent_oma.store(state, action, reward, next_state, done)
            agent_oma.train()
            state = next_state
            total_reward_oma += reward
            if done:
                break
        oma_avg_sum_rates[p].append(total_reward_oma / 300)

# Plotting Sum Rate vs IRS Elements (NOMA vs OMA) with Different Powers
plt.figure(figsize=(8, 6))
for p in powers:
    plt.plot(elements, noma_avg_sum_rates[p], label=f"NOMA Power {p} dBm", marker='o')
    plt.plot(elements, oma_avg_sum_rates[p], label=f"OMA Power {p} dBm", marker='x')
plt.xlabel("Number of IRS Elements")
plt.ylabel("Average Sum Rate (bps/Hz)")
plt.title("Figure 5: Sum Rate vs IRS Elements (NOMA vs OMA) with Different Powers (IRS-NOMA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()