# SQL Injection Detection and Reinforcement Learning Agent

This project implements a system to detect SQL Injection (SQLi) attacks using machine learning and reinforcement learning. It includes a trained model for SQLi detection, an environment to simulate SQL queries, and agents to generate and evaluate malicious queries.

---

## Project Overview
This project comprises:
- A **Naive Bayes-based SQLi detection model** trained on a dataset of SQL queries.
- A **Reinforcement Learning (RL) environment** simulating SQL query modifications.
- Two agents:
  - **DQNAgent**: A deep Q-learning agent.
  - **SimpleAgent**: A Q-learning-based agent with an epsilon-greedy approach.
- Tools to generate, evaluate, and analyze malicious queries.

---

## Setup Instructions

### Installation Steps
1. Clone this repository:
git clone <repository_url>
cd <repository_directory>

2. Install the required dependencies:
pip install -r requirement.txt


---

## Usage

### 1. Training the Model
To train the Naive Bayes model for SQLi detection:
python train_model.py

This will save the trained model as `sqli_model.pkl`.

### 2. Running the Environment
To train agents in the RL environment and generate malicious queries:
python main.py

This script:
- Trains both `DQNAgent` and `SimpleAgent`.
- Generates malicious queries and saves them as:
  - `generated_queries_Agent1_dqn.csv`
  - `generated_queries_Agent2_simple.csv`

### 3. Predicting New Queries
To evaluate the generated queries using the trained Naive Bayes model:
python Predict_new_generated_queries.py

This will:
- Predict whether each query is malicious.
- Save results as:
  - `simpleAgent_with_predictions.csv`
  - `dqnAgent_with_predictions.csv`

---

## Agents and Environment Details

### Agents
1. **DQNAgent**:
   - Implements Deep Q-Learning using TensorFlow.
   - Learns optimal actions to modify queries for bypassing detection.

2. **SimpleAgent**:
   - Uses a Q-learning approach with a pre-initialized Q-table.
   - Balances exploration and exploitation using an epsilon-greedy strategy.

### Environment (`SQLInjectionEnv`)
The environment simulates SQL injection scenarios by:
- Modifying queries based on selected actions.
- Rewarding agents for successfully bypassing detection mechanisms.

#### Actions Available in the Environment:
- Add `OR 1=1`, `--`, or `UNION SELECT`.
- Modify quotes or encode queries (e.g., Base64, Hex).
- Inject time-based or error-based payloads.

---

## Results Analysis

To analyze predictions made by both agents:
python Predict_new_generated_queries.py

This script calculates:
- Percentage of queries classified as malicious by each agent.
- Comparison of effectiveness between `DQNAgent` and `SimpleAgent`.
