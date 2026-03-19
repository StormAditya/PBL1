import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# --- 1. CORE HELPERS & ENVIRONMENT ---

def get_actual_processing_time(job_size, job_type, machine_idx):
    is_gpu_node = machine_idx >= 7
    is_fast_cpu = machine_idx < 3

    base_speed = 0.6 if is_fast_cpu else 1.0
    machine_type = 1 if is_gpu_node else 0
    
    if job_type == machine_type:
        return (job_size * base_speed) * 0.5  
    else:
        return (job_size * base_speed) * 3.0  

class AgenticSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # These are defaults, but the evaluation function dynamically overwrites them based on UI
        self.num_jobs = 100        
        self.num_machines = 10 
        self.pool_size = 5 
        
        self.action_space = spaces.Discrete(self.pool_size * self.num_machines)
        self.observation_space = spaces.Box(
            low=0, high=20, 
            shape=((self.pool_size * 2) + self.num_machines,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        all_jobs = [(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(self.num_jobs)]
        
        self.job_pool = all_jobs[:self.pool_size]
        self.remaining_jobs = all_jobs[self.pool_size:]
        self.machine_times = [0.0] * self.num_machines
        return self._get_state(), {}

    def _get_state(self):
        state_features = []
        for i in range(self.pool_size):
            if i < len(self.job_pool):
                state_features.append(self.job_pool[i][0] / 100.0)
                state_features.append(float(self.job_pool[i][1]))
            else:
                state_features.extend([0.0, 0.0])
                
        norm_machines = [m / 1000.0 for m in self.machine_times] 
        state_features.extend(norm_machines)
        return np.array(state_features, dtype=np.float32)

    def step(self, action):
        job_idx_in_pool = action // self.num_machines
        machine_idx = action % self.num_machines
        
        if job_idx_in_pool >= len(self.job_pool):
            return self._get_state(), -10.0, False, False, {}
            
        job_size, job_type = self.job_pool.pop(job_idx_in_pool)
        
        old_makespan = max(self.machine_times)
        old_imbalance = np.std(self.machine_times) 

        actual_time = get_actual_processing_time(job_size, job_type, machine_idx)
        self.machine_times[machine_idx] += actual_time
        
        if self.remaining_jobs:
            self.job_pool.append(self.remaining_jobs.pop(0))
            
        new_makespan = max(self.machine_times)
        new_imbalance = np.std(self.machine_times)
        
        reward = -(new_makespan - old_makespan) - (new_imbalance - old_imbalance)
        terminated = len(self.job_pool) == 0
        return self._get_state(), reward, terminated, False, {}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. TRADITIONAL SCHEDULERS ---

def run_random_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    for job_size, job_type in jobs:
        chosen_machine = random.randint(0, num_machines - 1)
        machines[chosen_machine] += get_actual_processing_time(job_size, job_type, chosen_machine)
    return max(machines)

def run_fcfs_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    for job_size, job_type in jobs:
        chosen_machine = machines.index(min(machines))
        machines[chosen_machine] += get_actual_processing_time(job_size, job_type, chosen_machine)
    return max(machines)

def run_sjf_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    sorted_jobs = sorted(jobs, key=lambda x: x[0]) 
    for job_size, job_type in sorted_jobs:
        chosen_machine = machines.index(min(machines))
        machines[chosen_machine] += get_actual_processing_time(job_size, job_type, chosen_machine)
    return max(machines)

# --- 3. AGENTIC AI EVALUATION ---

def get_valid_actions(env):
    mask = torch.ones(env.action_space.n, dtype=torch.bool)
    for i in range(env.action_space.n):
        job_idx = i // env.num_machines
        if job_idx >= len(env.job_pool):
            mask[i] = False
    return mask

def run_agentic_evaluation(env, policy_net, jobs):
    env.reset()
    env.job_pool = jobs[:env.pool_size].copy()
    env.remaining_jobs = jobs[env.pool_size:].copy()
    env.machine_times = [0.0] * env.num_machines
    
    state = env._get_state()
    terminated = False
    
    while not terminated:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            valid_mask = get_valid_actions(env)
            q_values[0, ~valid_mask] = -float('inf')
            action = q_values.argmax().item()
            
        state, _, terminated, _, _ = env.step(action)
        
    return max(env.machine_times)

# --- 4. STREAMLIT UI & EXECUTION ---

st.set_page_config(page_title="Agentic AI Scheduler", layout="wide")

st.title("🏭 Real-World Factory Scheduling Showdown")
st.markdown("""
Compare traditional scheduling algorithms against a trained Reinforcement Learning (Agentic AI) model.
Adjust the parameters on the left to see how the algorithms handle different workloads.
""")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Simulation Settings")
num_test_scenarios = st.sidebar.slider("Number of Test Scenarios", min_value=1, max_value=50, value=10)
num_jobs = st.sidebar.slider("Total Jobs per Scenario", min_value=10, max_value=500, value=100, step=10)
job_size_range = st.sidebar.slider("Job Size Range", min_value=1, max_value=200, value=(10, 100))

st.sidebar.divider()
st.sidebar.header("🧠 Model Configuration")
model_path = st.sidebar.text_input("Model Weights File", "single_agentic_ddqn.pth")

if st.button("🚀 Run Algorithm Showdown", type="primary", use_container_width=True):
    with st.spinner("Loading environment and model..."):
        env = AgenticSchedulingEnv()
        policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
        
        try:
            policy_net.load_state_dict(torch.load(model_path, weights_only=True))
            policy_net.eval()
        except FileNotFoundError:
            st.error(f"ERROR: '{model_path}' not found. Make sure the file is in the same directory!")
            st.stop()

    with st.spinner(f"Running {num_test_scenarios} Test Scenarios..."):
        # Generate scenarios based on UI custom sliders
        min_size, max_size = job_size_range
        test_scenarios = [
            [(np.random.randint(min_size, max_size + 1), random.choice([0, 1])) for _ in range(num_jobs)] 
            for _ in range(num_test_scenarios)
        ]

        # Keep track of individual run data for the raw table
        raw_data = []

        results = {"Random": 0, "FCFS": 0, "SJF": 0, "Agentic AI": 0}

        for i, jobs in enumerate(test_scenarios):
            r_rand = run_random_scheduler(jobs, env.num_machines)
            r_fcfs = run_fcfs_scheduler(jobs, env.num_machines)
            r_sjf = run_sjf_scheduler(jobs, env.num_machines)
            r_ai = run_agentic_evaluation(env, policy_net, jobs)
            
            results["Random"] += r_rand
            results["FCFS"] += r_fcfs
            results["SJF"] += r_sjf
            results["Agentic AI"] += r_ai
            
            raw_data.append({
                "Scenario": i + 1,
                "Random": round(r_rand, 2),
                "FCFS": round(r_fcfs, 2),
                "SJF": round(r_sjf, 2),
                "Agentic AI": round(r_ai, 2)
            })

        # Average out the results
        for key in results:
            results[key] /= num_test_scenarios

    # --- UI DISPLAY RESULTS ---
    st.divider()
    
    # Calculate performance improvement metric
    improvement = ((results['FCFS'] - results['Agentic AI']) / results['FCFS']) * 100
    
    st.subheader(f"📊 Final Average Makespans (Across {num_test_scenarios} Runs)")
    if improvement > 0:
        st.success(f"**Insight:** Agentic AI was **{improvement:.1f}% faster** than standard FCFS scheduling on average.")
    else:
        st.warning(f"**Insight:** Agentic AI was slower than standard FCFS by **{abs(improvement):.1f}%** on average. (Consider more training!)")
    
    # Display metrics side-by-side
    cols = st.columns(4)
    for idx, (key, value) in enumerate(results.items()):
        cols[idx].metric(label=key, value=f"{value:.1f}s")

    # --- VISUALIZATION ---
    labels = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Highlight the AI bar in a different color
    colors = ['#ff6666', '#bdc3c7', '#9b59b6', '#2ecc71']
    bars = ax.bar(labels, values, color=colors, edgecolor='black')

    ax.set_ylabel("Makespan (Seconds) - Lower is Better", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(bottom=0, top=max(values) * 1.15) 
    plt.tight_layout() 
    
    # Render plot in Streamlit
    st.pyplot(fig)
    
    # --- RAW DATA EXPANDER ---
    with st.expander("🔍 View Raw Scenario Data"):
        st.markdown("Here is the exact makespan (in seconds) for every algorithm across all individual test runs.")
        df = pd.DataFrame(raw_data)
        df.set_index("Scenario", inplace=True)
        st.dataframe(df, use_container_width=True)