import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import os

# --- Model Definition ---
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

# --- Helper Functions ---
def get_actual_processing_time(job_size, job_type, machine_idx):
    is_gpu_node = machine_idx >= 7
    is_fast_cpu = machine_idx < 3
    base_speed = 0.6 if is_fast_cpu else 1.0
    machine_type = 1 if is_gpu_node else 0
    return (job_size * base_speed) * (0.5 if job_type == machine_type else 3.0)

class RealWorldSchedulingEnv(gym.Env):
    def __init__(self, num_jobs=100):
        super().__init__()
        self.num_jobs = num_jobs        
        self.num_machines = 10 
        self.lookahead = 10  
        self.action_space = spaces.Discrete(self.num_machines)
        self.observation_space = spaces.Box(low=0, high=20, shape=(self.num_machines + (self.lookahead * 2),), dtype=np.float32)
        
    def reset_with_jobs(self, jobs):
        self.jobs = jobs
        self.current_job_idx = 0
        self.machine_times = [0.0] * self.num_machines
        return self._get_state()

    def _get_state(self):
        state_jobs = []
        for i in range(self.lookahead):
            idx = self.current_job_idx + i
            if idx < self.num_jobs:
                state_jobs.extend([self.jobs[idx][0] / 100.0, float(self.jobs[idx][1])])
            else:
                state_jobs.extend([0.0, 0.0]) 
        norm_machines = [m / 1000.0 for m in self.machine_times] 
        return np.array(state_jobs + norm_machines, dtype=np.float32)

    def step(self, action):
        job_size, job_type = self.jobs[self.current_job_idx]
        self.machine_times[action] += get_actual_processing_time(job_size, job_type, action)
        self.current_job_idx += 1
        return self._get_state(), 0.0, bool(self.current_job_idx >= self.num_jobs), False, {}

# --- Schedulers ---
def run_baseline(jobs, num_machines, strategy="FCFS"):
    machines = [0.0] * num_machines
    job_list = sorted(jobs, key=lambda x: x[0]) if strategy == "SJF" else jobs
    for job_size, job_type in job_list:
        idx = random.randint(0, num_machines-1) if strategy == "Random" else machines.index(min(machines))
        machines[idx] += get_actual_processing_time(job_size, job_type, idx)
    return max(machines)

def run_ddqn_evaluation(env, policy_net, jobs):
    state = env.reset_with_jobs(jobs)
    terminated = False
    while not terminated:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()
        state, _, terminated, _, _ = env.step(action)
    return max(env.machine_times)

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Scheduling Lab", layout="wide")
st.title("🤖 AI vs. Traditional Scheduling")
st.markdown("Compare a **Deep Q-Network** against standard industry heuristics.")

# Sidebar for Configuration
st.sidebar.header("Simulation Settings")
num_jobs = st.sidebar.slider("Number of Jobs", 50, 500, 100)
num_scenarios = st.sidebar.slider("Test Scenarios", 1, 50, 10)
model_path = "realworld_ddqn.pth"

# Cached Model Loader
@st.cache_resource
def load_model(path, input_dim, output_dim):
    if not os.path.exists(path): return None
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model

env = RealWorldSchedulingEnv(num_jobs=num_jobs)
loaded_policy_net = load_model(model_path, env.observation_space.shape[0], env.action_space.n)

if loaded_policy_net is None:
    st.error(f"Model file '{model_path}' not found! Please ensure it's in the directory.")
else:
    if st.button("🚀 Run Comparison Showdown"):
        results = {"Random": 0, "FCFS": 0, "SJF": 0, "Loaded DDQN": 0}
        
        progress_bar = st.progress(0)
        for i in range(num_scenarios):
            test_jobs = [(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(num_jobs)]
            results["Random"] += run_baseline(test_jobs, env.num_machines, "Random")
            results["FCFS"] += run_baseline(test_jobs, env.num_machines, "FCFS")
            results["SJF"] += run_baseline(test_jobs, env.num_machines, "SJF")
            results["Loaded DDQN"] += run_ddqn_evaluation(env, loaded_policy_net, test_jobs)
            progress_bar.progress((i + 1) / num_scenarios)

        for key in results: results[key] /= num_scenarios

        # --- Visualizations ---
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff6666', '#bdc3c7', '#9b59b6', '#2ecc71']
            bars = ax.bar(results.keys(), results.values(), color=colors)
            ax.set_ylabel("Avg Makespan (Seconds)")
            ax.set_title(f"Performance over {num_scenarios} Scenarios")
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}s", ha='center', fontweight='bold')
            
            st.pyplot(fig)

        with col2:
            st.subheader("Key Metrics")
            ai_perf = results["Loaded DDQN"]
            fcfs_perf = results["FCFS"]
            improvement = ((fcfs_perf - ai_perf) / fcfs_perf) * 100
            
            st.metric("AI Makespan", f"{ai_perf:.1f}s")
            st.metric("Improvement vs FCFS", f"{improvement:.1%}", delta=f"{improvement:.1f}%")
            
            st.table(results)