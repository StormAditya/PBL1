import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import altair as alt
import gymnasium as gym
from gymnasium import spaces

# --- 0. SESSION STATE INITIALIZATION ---
# This forces the app to run on the very first load, but stops it from 
# re-running automatically when sliders change.
if 'first_load_done' not in st.session_state:
    st.session_state.run_sim = True
    st.session_state.first_load_done = True
    st.session_state.results = None
    st.session_state.raw_data = None

# Callback function for the button
def trigger_simulation():
    st.session_state.run_sim = True

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

# Enforce wide layout
st.set_page_config(page_title="Agentic AI Scheduler", layout="wide", initial_sidebar_state="expanded")

st.title("Simulation: Agentic AI vs Traditional Schedulers")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Simulation Settings")
num_test_scenarios = st.sidebar.slider("Test Scenarios", 1, 50, 10)
num_jobs = st.sidebar.slider("Jobs per Scenario", 10, 500, 100, 10)
job_size_range = st.sidebar.slider("Job Size Range", 1, 200, (10, 100))
model_path = st.sidebar.text_input("Model Weights", "single_agentic_ddqn.pth")

st.sidebar.divider()
# The button uses the callback to set run_sim = True
st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True, on_click=trigger_simulation)


# --- EXECUTION LOGIC ---
# Only run the heavy math if the flag is True
if st.session_state.run_sim:
    with st.spinner("Loading environment and running simulations..."):
        env = AgenticSchedulingEnv()
        policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
        
        try:
            policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            policy_net.eval()
        except FileNotFoundError:
            st.error(f"ERROR: '{model_path}' not found. Place it in the same directory!")
            st.stop()

        min_size, max_size = job_size_range
        test_scenarios = [
            [(np.random.randint(min_size, max_size + 1), random.choice([0, 1])) for _ in range(num_jobs)] 
            for _ in range(num_test_scenarios)
        ]

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
            
            raw_data.append({"Scenario": i + 1, "Random": r_rand, "FCFS": r_fcfs, "SJF": r_sjf, "AI": r_ai})

        for key in results:
            results[key] /= num_test_scenarios
            
        # Save results to session memory
        st.session_state.results = results
        st.session_state.raw_data = raw_data
        
        # Turn off the run flag so changing sliders won't re-trigger this block
        st.session_state.run_sim = False


# --- UI DISPLAY RESULTS (COMPACT LAYOUT) ---
# We only display if we have data in memory
if st.session_state.results is not None:
    results = st.session_state.results
    raw_data = st.session_state.raw_data

    tab1, tab2 = st.tabs(["Dashboard View", "Raw Data View"])

    with tab1:
        left_col, right_col = st.columns([1, 2.5])
        
        with left_col:
            improvement = ((results['FCFS'] - results['Agentic AI']) / results['FCFS']) * 100 if results['FCFS'] > 0 else 0
            
            if improvement > 0:
                st.success(f"**AI won!** {improvement:.1f}% faster than FCFS.")
            else:
                st.warning(f"**AI lost.** {abs(improvement):.1f}% slower than FCFS.")
                
            st.markdown("### Average Makespans")
            m1, m2 = st.columns(2)
            m1.metric("Random", f"{results['Random']:.1f}s")
            m2.metric("FCFS", f"{results['FCFS']:.1f}s")
            m1.metric("SJF", f"{results['SJF']:.1f}s")
            m2.metric("Agentic AI", f"{results['Agentic AI']:.1f}s")

        with right_col:
            st.markdown("### Makespan Comparison")
            
            # Structure the data
            chart_df = pd.DataFrame({
                "Algorithm": list(results.keys()),
                "Makespan (Seconds)": list(results.values())
            })
            
            # Define the exact order and colors
            sort_order = ["Random", "FCFS", "SJF", "Agentic AI"]
            
            # Create the Base Bar Chart
            bars = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('Algorithm', sort=sort_order, axis=alt.Axis(labelAngle=0, title="Scheduling Algorithm")),
                y=alt.Y('Makespan (Seconds)', title="Average Makespan (Lower is Better)"),
                color=alt.Color(
                    'Algorithm', 
                    legend=None, 
                    scale=alt.Scale(domain=sort_order, range=['#e74c3c', '#3498db', '#9b59b6', '#2ecc71'])
                ),
                tooltip=['Algorithm', 'Makespan (Seconds)']
            )

            # Create the Text Labels Layer
            text = bars.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,  # Nudges the text slightly above the bar
                fontWeight='bold',
                fontSize=14
            ).encode(
                text=alt.Text('Makespan (Seconds):Q', format='.1f') 
            )

            # Combine bars and text, and set height
            chart = (bars + text).properties(height=500)
            
            # Render the chart
            st.altair_chart(chart, use_container_width=True)

    with tab2:
        st.markdown("Raw makespan data across all individual runs.")
        df = pd.DataFrame(raw_data).set_index("Scenario")
        st.dataframe(df, use_container_width=True, height=400)