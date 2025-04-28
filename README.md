# IRS-NOMA-ML: Machine Learning for IRS-Assisted NOMA Systems

This repository provides an end-to-end implementation of the paper **"Resource Allocation In IRSs Aided MISO-NOMA
Networks: A Machine Learning Approach"**  
We perform LSTM-based prediction, KGMM clustering, and analyze the system performance through multiple key plots.

---

## Directory Structure

```
aryamanpathak2022-irs-noma-ml/
|
├── main.py                # Master script
├── step-1.py               # Step 1 implementation
├── step-2.py               # Step 2 implementation
├── step-3.py               # Step 3 implementation
|
├── CLUSTERING/
│   └── kgmm_clustering.py  # K-GMM clustering code
|
├── DQN/
│   ├── agent.py            # Deep Q-learning agent
│   ├── agent_2.py
│   ├── noma_env.py         # NOMA environment for agent
│   └── noma_env_2.py
|
├── LSTM/
│   └── lstm_predictor.py   # LSTM prediction model
```

---

## Results and Plots

### 1. LSTM Prediction
- **Code File:** `LSTM/lstm_predictor.py`
- **Description:** Predicts the performance metrics based on time-series data.
- **Plot:**
<img width="530" alt="Screenshot 2025-04-28 at 9 03 22 AM" src="https://github.com/user-attachments/assets/447462ed-3756-4c50-92b9-dbaa4d3a906c" />

---

### 2. KGMM Clustering
- **Code File:** `CLUSTERING/kgmm_clustering.py`
- **Description:** Applies K-GMM (K-means + Gaussian Mixture Model) clustering to analyze user distribution.
- **Plot:**
<img width="579" alt="Screenshot 2025-04-28 at 9 03 40 AM" src="https://github.com/user-attachments/assets/ae853a8a-e52a-4634-a1ed-2f7372f4dcc6" />



---


In the following steps, we have:
- Built a **DQN agent** (`DQN/agent_@.py`) that learns to optimize resource allocation.
- Designed a **NOMA environment** (`DQN/noma_env_2.py`) that simulates the IRS-assisted wireless system.
- The agent interacts with the environment and learns the best actions, which are then used to plot the system's performance.

---


### 3. Average Sum Rate vs Number of IRS Elements
- **Code File:** `step-1.py`
- **Description:** Shows how the number of IRS elements affects the system's average sum rate.
- **Plot:**

<img width="583" alt="Screenshot 2025-04-28 at 9 04 40 AM" src="https://github.com/user-attachments/assets/e5ebf3fe-fa1e-4157-8bc6-dd9620f9f890" />



---

### 4. Average Sum Rate vs Transmission Power
- **Code File:** `step-2.py`
- **Description:** Plots the system's average sum rate variation with respect to different transmission power levels.
- **Plot:**

<img width="579" alt="Screenshot 2025-04-28 at 9 04 10 AM" src="https://github.com/user-attachments/assets/3c21869d-cb66-44b3-a030-19701668cd28" />

---

### 5. Average Sum Rate: OMA
- **Code File:** `step-2.py`
- **Description:** Compares the average sum rate of the IRS-assisted system versus the traditional OMA system.
- **Plot:**
<img width="604" alt="Screenshot 2025-04-28 at 9 05 01 AM" src="https://github.com/user-attachments/assets/c9e0f318-6ee5-44ef-87ad-16fe064adb0c" />



---

## How to Run

```bash
# Clone the repository
git clone https://github.com/aryamanpathak2022/IRS-NOMA-ML

# Navigate to the project directory
cd IRS-NOMA-ML

# Run the scripts
python step-1.py
python step-2.py

```

Make sure to install all the required Python packages using:


---

## Acknowledgment

This implementation is inspired by the research paper **"Resource Allocation In IRSs Aided MISO-NOMA
Networks: A Machine Learning Approach2"**. We thank the original authors for their work in advancing IRS-NOMA communication systems.

---

