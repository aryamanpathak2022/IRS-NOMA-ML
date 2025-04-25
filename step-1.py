from LSTM.lstm_predictor import *      
from CLUSTERING.kgmm_clustering import *
from DQN.agent_2 import *
from DQN.noma_env_2 import *
import time

import matplotlib.pyplot as plt

# Step 1: Simulate
print("Simulating user trajectories...")
trajectories = generate_user_trajectories(num_users=10, steps=50)
print("User trajectories generated. Shape:", trajectories.shape)

# Step 2: Train LSTM
print("Training LSTM model...")
lstm_model = train_lstm_model(trajectories, seq_len=10)
print("LSTM model trained.")

# Step 3: Predict future positions
print("Predicting future positions using LSTM...")
predicted = predict_positions(lstm_model, trajectories, future_steps=10)
print("Prediction completed. Shape of predicted positions:", predicted.shape)

# Step 4: Visualize
print("Plotting actual vs predicted trajectories...")
plot_trajectories(trajectories[:, -10:, :], predicted)

# Use final predicted positions for clustering (shape: num_users x 2)
last_pred_positions = predicted[:, -1, :]
print("Last predicted positions extracted. Shape:", last_pred_positions.shape)

# Perform clustering
print("Performing K-GMM clustering...")
labels, centers = k_gmm_clustering(last_pred_positions, n_clusters=5)
print("Clustering completed. Labels:", labels)

# Visualize clusters
print("Plotting clusters...")
plot_clusters(last_pred_positions, labels, centers)

# --- Simulation Parameters ---
powers_dbm = [20, 30, 40, 50] # Transmit power in dBm
powers_watts = [10**((p - 30) / 10) for p in powers_dbm] # Convert dBm to Watts for the environment/agent
iterations_list = [500, 1000, 2000, 5000] # Training steps per simulation
num_runs = 10  # Number of independent runs to average over

# --- Environment Fixed Parameters ---
env_params = {
    'num_users': 10,
    'num_clusters': 5,
    'num_elements': 25,
    'resolution_bits': 3 # Lowered bits -> Fewer phase levels (8) - might train faster initially
}
num_clusters = env_params['num_clusters'] # Extract for agent init
phase_levels = 2**env_params['resolution_bits'] # Extract for agent init

# --- Agent Hyperparameters ---
agent_params = {
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon': 1.0,
    'decay': 0.999, # Slower decay for more exploration
    'epsilon_min': 0.05, # Higher minimum epsilon
    'target_update_freq': 200, # Update target less frequently
    'buffer_size': 50000 # Larger buffer
}
agent_batch_size = 64 # Batch size for training agent.train()

# --- Data Structures to Store Results ---
# Stores results for all runs: {iteration_count: [ [run1_pow1, run1_pow2,...], [run2_pow1, run2_pow2,...], ... ]}
all_runs_avg_sum_rates = {iteration: [] for iteration in iterations_list}
# Store losses per run per iteration count (optional)
# all_runs_losses = {iteration: [] for iteration in iterations_list}


print(f"Starting simulation: {num_runs} runs, Iterations: {iterations_list}, Powers (dBm): {powers_dbm}")
overall_start_time = time.time()

# --- Main Simulation Loop (Over Multiple Runs) ---
for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} ---")
    run_start_time = time.time()
    # Stores results for this single run: {iteration_count: [pow1_rate, pow2_rate, ...]}
    run_avg_sum_rates = {iteration: [] for iteration in iterations_list}
    # run_losses = {iteration: [] for iteration in iterations_list} # Store losses for this run

    for iteration_count in iterations_list:
        print(f"  Training iterations: {iteration_count}")
        power_rates_this_iter = [] # Temporarily store rates for this iteration count
        # losses_this_iter = [] # Temporarily store losses

        for p_watts, p_dbm in zip(powers_watts, powers_dbm):
            # print(f"    Simulating with transmit power = {p_dbm} dBm ({p_watts:.4f} W)") # Verbose

            # Initialize environment and agent for EACH power level simulation
            env = IRSNOMAEnv(max_power=p_watts, **env_params)
            agent = DQNAgent(
                state_dim=env.state_dim,
                phase_levels=env.phase_levels,
                num_clusters=env.num_clusters,
                total_power=p_watts, # Pass power in Watts
                **agent_params # Pass hyperparameters
            )

            state = env.reset()
            total_reward_accumulated = 0
            # total_loss_accumulated = 0
            # loss_count = 0

            # --- Training Loop ---
            for i in range(iteration_count):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store(state, action, reward, next_state, done)

                # Train the agent
                loss = agent.train(batch_size=agent_batch_size)
                # if loss is not None:
                #     total_loss_accumulated += loss
                #     loss_count += 1

                state = next_state
                total_reward_accumulated += reward

                # Reset if episode ends (though 'done' is based on time limit here)
                if done:
                    state = env.reset()
                    # print(f"      Episode finished at step {i+1}, resetting env.") # Debug


            # Calculate average reward (sum rate) ACHIEVED DURING TRAINING
            # This reflects learning progress but isn't a pure evaluation of the final policy
            avg_reward_this_run = total_reward_accumulated / iteration_count if iteration_count > 0 else 0
            power_rates_this_iter.append(avg_reward_this_run)

            # avg_loss_this_run = total_loss_accumulated / loss_count if loss_count > 0 else 0
            # losses_this_iter.append(avg_loss_this_run)
            # print(f"      Avg sum rate (Training): {avg_reward_this_run:.4f}") # Verbose
            # print(f"      Avg loss (Training): {avg_loss_this_run:.4f}") # Verbose


            # --- Optional: Evaluate AFTER Training (Recommended) ---
            # Uncomment this section to evaluate the policy learned after training.
            # print(f"    Evaluating policy for power = {p_dbm} dBm...")
            # eval_episodes = 10 # Number of episodes to evaluate
            # eval_max_steps = 50 # Max steps per evaluation episode
            # total_eval_reward = 0
            # total_eval_steps = 0
            #
            # current_epsilon = agent.epsilon # Save epsilon
            # agent.epsilon = 0 # Use greedy policy for evaluation
            #
            # for ep in range(eval_episodes):
            #     eval_state = env.reset()
            #     episode_reward = 0
            #     for step in range(eval_max_steps):
            #         eval_action = agent.select_action(eval_state) # Greedy action
            #         next_eval_state, reward, done = env.step(eval_action)
            #         episode_reward += reward
            #         eval_state = next_eval_state
            #         total_eval_steps += 1
            #         if done:
            #             break # End episode if env signals done
            #     total_eval_reward += episode_reward
            #
            # agent.epsilon = current_epsilon # Restore epsilon
            # avg_eval_sum_rate = total_eval_reward / eval_episodes if eval_episodes > 0 else 0
            # print(f"      Avg EVALUATION sum rate: {avg_eval_sum_rate:.4f}")
            # # If using evaluation, append this rate instead of the training one:
            # # power_rates_this_iter.append(avg_eval_sum_rate) # Overwrite the training average
            # --- End Optional Evaluation ---


        # Store the list of rates (and losses) for this iteration count for the current run
        run_avg_sum_rates[iteration_count] = power_rates_this_iter
        # run_losses[iteration_count] = losses_this_iter


    # Append the results of the entire run to the master list
    for iteration_count in iterations_list:
        all_runs_avg_sum_rates[iteration_count].append(run_avg_sum_rates[iteration_count])
        # all_runs_losses[iteration_count].append(run_losses[iteration_count]) # Store losses across runs

    run_end_time = time.time()
    print(f"--- Run {run + 1} finished in {run_end_time - run_start_time:.2f} seconds ---")


# --- Averaging Results Across Runs ---
print("\n--- Averaging Results Across Runs ---")
final_avg_sum_rates = {iteration: [] for iteration in iterations_list}
final_std_sum_rates = {iteration: [] for iteration in iterations_list} # Standard deviation

for iteration_count in iterations_list:
    # Convert list of lists into a NumPy array for easy averaging
    # Shape will be (num_runs, num_powers)
    rates_array = np.array(all_runs_avg_sum_rates[iteration_count])

    # Calculate mean and std dev across the 'runs' axis (axis=0)
    final_avg_sum_rates[iteration_count] = np.mean(rates_array, axis=0).tolist()
    final_std_sum_rates[iteration_count] = np.std(rates_array, axis=0).tolist()
    # print(f"Iteration {iteration_count}: Avg Rates: {[f'{x:.4f}' for x in final_avg_sum_rates[iteration_count]]}")
    # print(f"Iteration {iteration_count}: Std Devs: {[f'{x:.4f}' for x in final_std_sum_rates[iteration_count]]}")


# --- Plotting Averaged Results ---
print("\nPlotting Averaged Sum Rates...")
plt.figure(figsize=(10, 7))
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X'] # Different markers

for i, iteration_count in enumerate(iterations_list):
    marker = markers[i % len(markers)]
    # Plot the averaged line
    plt.plot(powers_dbm, final_avg_sum_rates[iteration_count],
             label=f"{iteration_count} Iterations", marker=marker, linewidth=2, markersize=6)

    # Optional: Add shaded region for standard deviation
    lower_bound = np.array(final_avg_sum_rates[iteration_count]) - np.array(final_std_sum_rates[iteration_count])
    upper_bound = np.array(final_avg_sum_rates[iteration_count]) + np.array(final_std_sum_rates[iteration_count])
    plt.fill_between(powers_dbm, lower_bound, upper_bound, alpha=0.15) # Adjust alpha for transparency

plt.xlabel("Transmit Power (dBm)", fontsize=12)
plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
plt.title(f"Avg Sum Rate vs Transmit Power (Averaged over {num_runs} runs)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--')
plt.xticks(powers_dbm) # Ensure all power levels are marked
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

overall_end_time = time.time()
print(f"\nTotal simulation time: {overall_end_time - overall_start_time:.2f} seconds")

# --- Optional: Plotting Average Loss (if tracked) ---
# print("\nPlotting Average Training Loss...")
# final_avg_losses = {iteration: [] for iteration in iterations_list}
# for iteration_count in iterations_list:
#     loss_array = np.array(all_runs_losses[iteration_count]) # Shape (num_runs, num_powers)
#     final_avg_losses[iteration_count] = np.mean(loss_array, axis=0).tolist()
#
# plt.figure(figsize=(10, 7))
# for i, iteration_count in enumerate(iterations_list):
#     marker = markers[i % len(markers)]
#     plt.plot(powers_dbm, final_avg_losses[iteration_count],
#              label=f"{iteration_count} Iterations", marker=marker, linewidth=2, markersize=6)
#
# plt.xlabel("Transmit Power (dBm)", fontsize=12)
# plt.ylabel("Average Training Loss", fontsize=12)
# plt.title(f"Avg Training Loss vs Transmit Power (Averaged over {num_runs} runs)", fontsize=14)
# plt.legend(fontsize=10)
# plt.grid(True, linestyle='--')
# plt.xticks(powers_dbm)
# plt.yscale('log') # Loss often varies greatly, log scale can be helpful
# plt.tight_layout()
# plt.show()

# --- Simulation Parameters ---
# Using dBm for definition, converting to Watts for env/agent
powers_dbm = [5, 10, 15, 20, 25, 30]
powers_watts = [10**((p - 30) / 10) for p in powers_dbm]

elements = [5, 10, 15, 20, 25, 30]
num_runs = 10  # Number of independent runs to average over
fixed_iterations = 300 # Fixed number of training steps for plots 2 and 3

# --- Fixed Environment Parameters (Ensure consistency) ---
# These should match the defaults used in your IRSNOMAEnv definition if not passed explicitly
env_defaults = {
    'num_users': 10,
    'num_clusters': 5,
    'resolution_bits': 3,
    # Add any other relevant fixed parameters your env might need
}
# Define noise power explicitly for OMA calculation consistency
# This MUST match the value used inside your actual IRSNOMAEnv's step method
NOISE_POWER_LINEAR = 1e-9

# --- Agent Hyperparameters (Consistent Set) ---
agent_params = {
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon': 1.0,         # Initial epsilon
    'decay': 0.995,         # Epsilon decay rate per step
    'epsilon_min': 0.01,
    'target_update_freq': 100,
    'buffer_size': 10000,
    # Add batch_size if your agent.train() expects it
    'batch_size': 64
}


# ==============================================================================
# 2. Sum Rate vs IRS Elements with Different Powers (Averaged)
# ==============================================================================
print("\nStarting Simulation 2: Sum Rate vs IRS Elements (Averaged)...")
start_time_2 = time.time()

# Structure: {power_dbm: [ [run1_elem1, run1_elem2,...], [run2_elem1, run2_elem2,...], ... ]}
all_runs_avg_sum_rates_elements = {p_dbm: [] for p_dbm in powers_dbm}

for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} --- (Plot 2)")
    # Structure: {power_dbm: [elem1_rate, elem2_rate, ...]}
    run_avg_sum_rates_elements = {p_dbm: [] for p_dbm in powers_dbm}

    for p_dbm, p_watts in zip(powers_dbm, powers_watts):
        print(f"  Power level: {p_dbm} dBm ({p_watts:.4f} W)")
        element_rates_this_power = [] # Temp list for rates at this power level

        for n_elements in elements:
            # print(f"    Number of IRS elements: {n_elements}") # Verbose

            # --- Initialize Environment and Agent ---
            try:
                env = IRSNOMAEnv(num_elements=n_elements,
                                 max_power=p_watts,
                                 **env_defaults)

                agent = DQNAgent(state_dim=env.state_dim,
                                 phase_levels=env.phase_levels,
                                 num_clusters=env.num_clusters,
                                 total_power=p_watts, # Agent uses total power for scaling actions
                                 **agent_params) # Pass other hyperparameters
                batch_size = agent_params.get('batch_size', 64) # Get batch size from params

            except Exception as e:
                print(f"ERROR initializing env/agent: {e}")
                print(f"Skipping combination: Power={p_dbm}dBm, Elements={n_elements}")
                element_rates_this_power.append(np.nan) # Mark as invalid
                continue # Skip to next element count

            # --- Training Loop ---
            state = env.reset()
            total_reward = 0
            steps_in_run = 0

            for i in range(fixed_iterations): # Use fixed iterations
                action = agent.select_action(state)
                try:
                    next_state, reward, done = env.step(action)
                except Exception as e:
                     print(f"ERROR during env.step: {e}")
                     # Decide how to handle: break, mark as NaN, etc.
                     total_reward = np.nan # Mark run as invalid
                     steps_in_run = 1 # Avoid division by zero
                     break

                agent.store(state, action, reward, next_state, done)
                agent.train(batch_size=batch_size) # Pass batch size if needed
                state = next_state
                total_reward += reward
                steps_in_run += 1
                if done:
                    # print(f"    Episode ended early at step {i}, resetting.") # Verbose
                    state = env.reset()

            # Calculate average reward for this specific (element, power) setting in this run
            if steps_in_run > 0 and not np.isnan(total_reward):
                avg_reward_this_run = total_reward / steps_in_run
            else:
                avg_reward_this_run = np.nan # Invalid result

            element_rates_this_power.append(avg_reward_this_run)
            # print(f"      Elements={n_elements}, Avg Rate={avg_reward_this_run:.4f}") # Verbose

        # Store the list of rates for this power level for the current run
        run_avg_sum_rates_elements[p_dbm] = element_rates_this_power

    # Append the results of the entire run to the master list
    for p_dbm in powers_dbm:
        all_runs_avg_sum_rates_elements[p_dbm].append(run_avg_sum_rates_elements[p_dbm])

# --- Averaging Results Across Runs (Plot 2) ---
print("\n--- Averaging Results Across Runs (Plot 2) ---")
final_avg_elements = {p_dbm: [] for p_dbm in powers_dbm}
final_std_elements = {p_dbm: [] for p_dbm in powers_dbm} # Optional: Standard deviation

for p_dbm in powers_dbm:
    # Shape: (num_runs, num_elements_points)
    rates_array = np.array(all_runs_avg_sum_rates_elements[p_dbm], dtype=float)
    # Use nanmean/nanstd to ignore NaN values from potential errors
    final_avg_elements[p_dbm] = np.nanmean(rates_array, axis=0).tolist()
    final_std_elements[p_dbm] = np.nanstd(rates_array, axis=0).tolist()

# --- Plotting Averaged Results (Plot 2) ---
print("\nPlotting Averaged Results (Plot 2: Rate vs Elements)...")
plt.figure(figsize=(10, 7))
markers = ['o', 's', '^', 'd', 'v', '*']
colors_plot2 = plt.cm.plasma(np.linspace(0, 1, len(powers_dbm)))

# Check if any valid data exists before plotting
has_valid_data_p2 = not all(all(np.isnan(val) for val in final_avg_elements[p_dbm]) for p_dbm in powers_dbm)

if has_valid_data_p2:
    for i, p_dbm in enumerate(powers_dbm):
        if not all(np.isnan(final_avg_elements[p_dbm])): # Check if data for this power level exists
            plt.plot(elements, final_avg_elements[p_dbm],
                     label=f"Power {p_dbm} dBm", marker=markers[i % len(markers)],
                     color=colors_plot2[i], linewidth=2, markersize=6)
            # Optional: Add shaded region for standard deviation
            lower_bound = np.array(final_avg_elements[p_dbm]) - np.array(final_std_elements[p_dbm])
            upper_bound = np.array(final_avg_elements[p_dbm]) + np.array(final_std_elements[p_dbm])
            plt.fill_between(elements, lower_bound, upper_bound, alpha=0.15, color=colors_plot2[i])

    plt.xlabel("Number of IRS Elements", fontsize=12)
    plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
    plt.title(f"Avg Sum Rate vs IRS Elements (Averaged over {num_runs} runs, {fixed_iterations} iter.)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--')
    plt.xticks(elements)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping Plot 2: No valid data generated.")


end_time_2 = time.time()
print(f"\nSimulation 2 time: {end_time_2 - start_time_2:.2f} seconds")


# ==============================================================================
# 3. Sum Rate vs IRS Elements: NOMA (RL) vs OMA (Analytical) (Averaged)
# ==============================================================================

print("\nStarting Simulation 3: NOMA (RL) vs OMA (Analytical) Comparison (Averaged)...")
start_time_3 = time.time()

# Data structures for NOMA (from RL) and OMA (analytical) results
# Structure: {power_dbm: [ [run1_elem1,...], [run2_elem1,...], ... ]}
all_runs_noma_rates = {p_dbm: [] for p_dbm in powers_dbm}
all_runs_oma_rates = {p_dbm: [] for p_dbm in powers_dbm} # Stores analytical results per run (will be identical)

for run in range(num_runs):
    print(f"\n--- Run {run + 1} / {num_runs} --- (Plot 3)")
    run_noma_rates = {p_dbm: [] for p_dbm in powers_dbm}
    run_oma_rates = {p_dbm: [] for p_dbm in powers_dbm} # Store analytical OMA per run

    for p_dbm, p_watts in zip(powers_dbm, powers_watts):
        print(f"  Power level: {p_dbm} dBm ({p_watts:.4f} W)")

        element_rates_noma_this_power = []
        element_rates_oma_this_power = [] # Store analytical OMA results

        for n_elements in elements:
            # --- NOMA Simulation (RL based) ---
            # print(f"    NOMA (RL): Elements={n_elements}") # Verbose
            try:
                env_noma = IRSNOMAEnv(num_elements=n_elements,
                                     max_power=p_watts,
                                     **env_defaults)
                agent_noma = DQNAgent(state_dim=env_noma.state_dim,
                                     phase_levels=env_noma.phase_levels,
                                     num_clusters=env_noma.num_clusters,
                                     total_power=p_watts,
                                     **agent_params)
                batch_size = agent_params.get('batch_size', 64)

            except Exception as e:
                print(f"ERROR initializing NOMA env/agent: {e}")
                print(f"Skipping NOMA combination: Power={p_dbm}dBm, Elements={n_elements}")
                element_rates_noma_this_power.append(np.nan) # Mark as invalid
                # Still calculate OMA for comparison if desired
                # element_rates_oma_this_power.append(np.nan)
                # continue # Skip NOMA training loop if init failed

            # --- NOMA Training Loop ---
            if not np.isnan(element_rates_noma_this_power[-1:]): # Proceed if init succeeded
                state = env_noma.reset()
                total_reward_noma = 0
                steps_noma = 0
                for i in range(fixed_iterations):
                    action = agent_noma.select_action(state)
                    try:
                        next_state, reward, done = env_noma.step(action)
                    except Exception as e:
                         print(f"ERROR during NOMA env.step: {e}")
                         total_reward_noma = np.nan
                         steps_noma = 1
                         break

                    agent_noma.store(state, action, reward, next_state, done)
                    agent_noma.train(batch_size=batch_size)
                    state = next_state
                    total_reward_noma += reward
                    steps_noma += 1
                    if done:
                        state = env_noma.reset()

                # Calculate NOMA average rate for this setting
                if steps_noma > 0 and not np.isnan(total_reward_noma):
                     avg_noma = total_reward_noma / steps_noma
                else:
                     avg_noma = np.nan
                element_rates_noma_this_power.append(avg_noma)
            # --- End NOMA Simulation ---


            # --- OMA Analytical Calculation ---
            # print(f"    OMA (Analytical): Elements={n_elements}") # Verbose
            try:
                # Assumptions for Analytical OMA Baseline:
                # 1. Ideal quadratic gain scaling with number of elements (common theoretical assumption)
                # 2. Perfect phase alignment (max possible gain factor = 1)
                # 3. Interference-free (by OMA definition)
                # 4. Uses the *same* noise power as the NOMA environment for fair comparison
                # 5. Rate calculated as if one user gets all power/resources (simple baseline)

                scaled_gain_oma_analytical = n_elements ** 2 # Ideal quadratic gain

                # Calculate the interference-free SNR for a single representative OMA user
                # Add small epsilon to noise to prevent log2(1+inf) if noise is zero
                snr_oma_analytical = (scaled_gain_oma_analytical * p_watts) / (NOISE_POWER_LINEAR + 1e-20)

                # Calculate Analytical OMA Rate = log2(1 + SNR_single_user)
                # Clamp SNR to avoid potential overflow issues with very high gain/power
                max_snr = 1e12 # Set a reasonable upper limit for SNR
                snr_oma_analytical = min(snr_oma_analytical, max_snr)

                avg_oma_analytical = np.log2(1 + snr_oma_analytical)
                element_rates_oma_this_power.append(avg_oma_analytical)

            except Exception as e:
                print(f"ERROR during OMA calculation: {e}")
                element_rates_oma_this_power.append(np.nan)
            # --- End OMA Analytical Calculation ---

        # Store results for this power level
        run_noma_rates[p_dbm] = element_rates_noma_this_power
        run_oma_rates[p_dbm] = element_rates_oma_this_power

    # Append results for the whole run
    for p_dbm in powers_dbm:
        all_runs_noma_rates[p_dbm].append(run_noma_rates[p_dbm])
        all_runs_oma_rates[p_dbm].append(run_oma_rates[p_dbm])

# --- Averaging Results Across Runs (Plot 3)---
# Averaging NOMA (RL results) smooths RL variance.
# Averaging OMA (analytical results) will just return the deterministic analytical value.
print("\n--- Averaging Results Across Runs (Plot 3) ---")
final_avg_noma = {p_dbm: [] for p_dbm in powers_dbm}
final_std_noma = {p_dbm: [] for p_dbm in powers_dbm}
final_avg_oma = {p_dbm: [] for p_dbm in powers_dbm}
final_std_oma = {p_dbm: [] for p_dbm in powers_dbm} # Will be ~zero for analytical OMA

for p_dbm in powers_dbm:
    noma_rates_array = np.array(all_runs_noma_rates[p_dbm], dtype=float)
    final_avg_noma[p_dbm] = np.nanmean(noma_rates_array, axis=0).tolist()
    final_std_noma[p_dbm] = np.nanstd(noma_rates_array, axis=0).tolist()

    oma_rates_array = np.array(all_runs_oma_rates[p_dbm], dtype=float)
    final_avg_oma[p_dbm] = np.nanmean(oma_rates_array, axis=0).tolist()
    final_std_oma[p_dbm] = np.nanstd(oma_rates_array, axis=0).tolist() # Should be near zero

# --- Plotting Averaged Results (Plot 3 - NOMA vs Analytical OMA) ---
print("\nPlotting Averaged Results (Plot 3: NOMA (RL) vs OMA (Analytical))...")
plt.figure(figsize=(10, 7))
markers = ['o', 's', '^', 'd', 'v', '*']
linestyles = ['-', '--', ':', '-.'] # Solid for OMA, Dashed for NOMA
colors_plot3 = plt.cm.viridis(np.linspace(0, 1, len(powers_dbm)))

# Check if any valid data exists
has_valid_noma_p3 = not all(all(np.isnan(val) for val in final_avg_noma[p_dbm]) for p_dbm in powers_dbm)
has_valid_oma_p3 = not all(all(np.isnan(val) for val in final_avg_oma[p_dbm]) for p_dbm in powers_dbm)


if has_valid_noma_p3 or has_valid_oma_p3: # Proceed if at least one data type is valid
    for i, p_dbm in enumerate(powers_dbm):
        color = colors_plot3[i]
        marker = markers[i % len(markers)]

        # Plot NOMA (RL Results) - Dashed lines
        if has_valid_noma_p3 and not all(np.isnan(final_avg_noma[p_dbm])):
            plt.plot(elements, final_avg_noma[p_dbm],
                     label=f"NOMA (RL) P={p_dbm}dBm", marker=marker,
                     linestyle=linestyles[1], color=color, linewidth=2, markersize=6)
            lower_noma = np.array(final_avg_noma[p_dbm]) - np.array(final_std_noma[p_dbm])
            upper_noma = np.array(final_avg_noma[p_dbm]) + np.array(final_std_noma[p_dbm])
            plt.fill_between(elements, lower_noma, upper_noma, alpha=0.15, color=color)

        # Plot OMA (Analytical Results) - Solid lines
        if has_valid_oma_p3 and not all(np.isnan(final_avg_oma[p_dbm])):
             plt.plot(elements, final_avg_oma[p_dbm],
                      label=f"OMA (Analytical) P={p_dbm}dBm", marker=marker,
                      linestyle=linestyles[0], color=color, linewidth=2, markersize=6)
             # No shaded region for analytical OMA needed as std dev should be ~0


    plt.xlabel("Number of IRS Elements", fontsize=12)
    plt.ylabel("Average Sum Rate (bps/Hz)", fontsize=12)
    plt.title(f"Avg Sum Rate vs IRS Elements: NOMA (RL) vs OMA (Analytical) (NOMA Avg over {num_runs} runs)", fontsize=14)
    # Adjust legend font size and position
    plt.legend(fontsize=8, loc='upper left', ncol=2) # Example: smaller font, 2 columns
    plt.grid(True, linestyle='--')
    plt.xticks(elements)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout()
    plt.show()
else:
    print("Skipping Plot 3: No valid NOMA or OMA data generated.")


end_time_3 = time.time()
print(f"\nSimulation 3 time: {end_time_3 - start_time_3:.2f} seconds")