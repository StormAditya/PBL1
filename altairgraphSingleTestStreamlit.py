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
        self.schedule_records = []
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

        start_time = self.machine_times[machine_idx]
        actual_time = get_actual_processing_time(job_size, job_type, machine_idx)
        self.machine_times[machine_idx] += actual_time
        finish_time = self.machine_times[machine_idx]
        is_gpu_node = machine_idx >= 7
        affinity_match = int(job_type == (1 if is_gpu_node else 0))
        self.schedule_records.append({
            "start_time": start_time,
            "finish_time": finish_time,
            "processing_time": actual_time,
            "affinity_match": affinity_match,
        })
        
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

def compute_schedule_metrics(machine_times, schedule_records, num_machines):
    makespan = max(machine_times) if machine_times else 0.0
    total_work = sum(machine_times)
    utilization = (total_work / (num_machines * makespan)) if makespan > 0 else 0.0
    waiting_times = [record["start_time"] for record in schedule_records]
    completion_times = [record["finish_time"] for record in schedule_records]
    throughput = (len(schedule_records) / makespan) if makespan > 0 else 0.0
    affinity_rate = (100.0 * np.mean([record["affinity_match"] for record in schedule_records])) if schedule_records else 0.0

    return {
        "makespan": makespan,
        "avg_waiting_time": float(np.mean(waiting_times)) if waiting_times else 0.0,
        "avg_completion_time": float(np.mean(completion_times)) if completion_times else 0.0,
        "throughput": throughput,
        "load_std": float(np.std(machine_times)) if machine_times else 0.0,
        "utilization": utilization,
        "affinity_rate": affinity_rate,
    }

def run_random_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    schedule_records = []
    for job_size, job_type in jobs:
        chosen_machine = random.randint(0, num_machines - 1)
        start_time = machines[chosen_machine]
        actual_time = get_actual_processing_time(job_size, job_type, chosen_machine)
        machines[chosen_machine] += actual_time
        is_gpu_node = chosen_machine >= 7
        schedule_records.append({
            "start_time": start_time,
            "finish_time": machines[chosen_machine],
            "processing_time": actual_time,
            "affinity_match": int(job_type == (1 if is_gpu_node else 0)),
        })
    return compute_schedule_metrics(machines, schedule_records, num_machines)

def run_fcfs_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    schedule_records = []
    for job_size, job_type in jobs:
        chosen_machine = machines.index(min(machines))
        start_time = machines[chosen_machine]
        actual_time = get_actual_processing_time(job_size, job_type, chosen_machine)
        machines[chosen_machine] += actual_time
        is_gpu_node = chosen_machine >= 7
        schedule_records.append({
            "start_time": start_time,
            "finish_time": machines[chosen_machine],
            "processing_time": actual_time,
            "affinity_match": int(job_type == (1 if is_gpu_node else 0)),
        })
    return compute_schedule_metrics(machines, schedule_records, num_machines)

def run_sjf_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    schedule_records = []
    sorted_jobs = sorted(jobs, key=lambda x: x[0]) 
    for job_size, job_type in sorted_jobs:
        chosen_machine = machines.index(min(machines))
        start_time = machines[chosen_machine]
        actual_time = get_actual_processing_time(job_size, job_type, chosen_machine)
        machines[chosen_machine] += actual_time
        is_gpu_node = chosen_machine >= 7
        schedule_records.append({
            "start_time": start_time,
            "finish_time": machines[chosen_machine],
            "processing_time": actual_time,
            "affinity_match": int(job_type == (1 if is_gpu_node else 0)),
        })
    return compute_schedule_metrics(machines, schedule_records, num_machines)

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
    env.schedule_records = []
    
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
        
    return compute_schedule_metrics(env.machine_times, env.schedule_records, env.num_machines)

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
# Only run if the flag is True
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

        metrics_to_report = [
            "makespan",
            "avg_waiting_time",
            "avg_completion_time",
            "throughput",
            "utilization",
            "affinity_rate",
        ]

        metric_display = {
            "makespan": "Makespan (s)",
            "avg_waiting_time": "Avg Waiting Time (s)",
            "avg_completion_time": "Avg Completion Time (s)",
            "throughput": "Throughput (jobs/s)",
            "utilization": "Utilization (%)",
            "affinity_rate": "Affinity Match Rate (%)",
        }

        raw_data = []
        results = {
            "Random": {metric: 0.0 for metric in metrics_to_report},
            "FCFS": {metric: 0.0 for metric in metrics_to_report},
            "SJF": {metric: 0.0 for metric in metrics_to_report},
            "Agentic AI": {metric: 0.0 for metric in metrics_to_report},
        }

        for i, jobs in enumerate(test_scenarios):
            r_rand = run_random_scheduler(jobs, env.num_machines)
            r_fcfs = run_fcfs_scheduler(jobs, env.num_machines)
            r_sjf = run_sjf_scheduler(jobs, env.num_machines)
            r_ai = run_agentic_evaluation(env, policy_net, jobs)
            
            for metric in metrics_to_report:
                results["Random"][metric] += r_rand[metric]
                results["FCFS"][metric] += r_fcfs[metric]
                results["SJF"][metric] += r_sjf[metric]
                results["Agentic AI"][metric] += r_ai[metric]

            for algorithm, metric_values in {
                "Random": r_rand,
                "FCFS": r_fcfs,
                "SJF": r_sjf,
                "Agentic AI": r_ai,
            }.items():
                raw_row = {
                    "Scenario": i + 1,
                    "Algorithm": algorithm,
                    "Makespan (s)": metric_values["makespan"],
                    "Avg Waiting Time (s)": metric_values["avg_waiting_time"],
                    "Avg Completion Time (s)": metric_values["avg_completion_time"],
                    "Throughput (jobs/s)": metric_values["throughput"],
                    "Utilization (%)": metric_values["utilization"] * 100.0,
                    "Affinity Match Rate (%)": metric_values["affinity_rate"],
                }
                raw_data.append(raw_row)

        for key in results:
            for metric in metrics_to_report:
                results[key][metric] /= num_test_scenarios

        average_rows = []
        for algorithm, metric_values in results.items():
            average_rows.append({
                "Algorithm": algorithm,
                metric_display["makespan"]: metric_values["makespan"],
                metric_display["avg_waiting_time"]: metric_values["avg_waiting_time"],
                metric_display["avg_completion_time"]: metric_values["avg_completion_time"],
                metric_display["throughput"]: metric_values["throughput"],
                metric_display["utilization"]: metric_values["utilization"] * 100.0,
                metric_display["affinity_rate"]: metric_values["affinity_rate"],
            })
            
        # Save results to session memory
        st.session_state.results = results
        st.session_state.raw_data = raw_data
        st.session_state.average_metrics = average_rows
        st.session_state.metric_display = metric_display
        st.session_state.metrics_to_report = metrics_to_report
        
        # Turn off the run flag so changing sliders won't re-trigger this block
        st.session_state.run_sim = False


# --- UI DISPLAY RESULTS  ---
# We only display if we have data in memory
if st.session_state.results is not None:
    results = st.session_state.results
    raw_data = st.session_state.raw_data
    average_metrics = st.session_state.average_metrics
    metric_display = st.session_state.metric_display
    metrics_to_report = st.session_state.metrics_to_report

    tab1, tab2, tab3 = st.tabs(["Dashboard View", "Raw Data View", "Average Metrics Table"])

    with tab1:
        left_col, right_col = st.columns([1, 2.5])
        
        with left_col:
            fcfs_makespan = results["FCFS"]["makespan"]
            ai_makespan = results["Agentic AI"]["makespan"]
            improvement = ((fcfs_makespan - ai_makespan) / fcfs_makespan) * 100 if fcfs_makespan > 0 else 0
            
            if improvement > 0:
                st.success(f"**AI won!** {improvement:.1f}% faster than FCFS.")
            else:
                st.warning(f"**AI lost.** {abs(improvement):.1f}% slower than FCFS.")

            selected_metric = st.selectbox(
                "Metric for Comparison Chart",
                options=metrics_to_report,
                format_func=lambda x: metric_display[x],
                index=0,
            )

            st.info(
                """
                **Environment Guide**
                - **Machine Types:**
                  - Machines `0-2`: Fast CPU nodes
                  - Machines `3-6`: Standard CPU nodes
                  - Machines `7-9`: GPU nodes
                - **Job Types:**
                  - `0` = CPU oriented job
                  - `1` = GPU oriented job
                - **Affinity Rule:**
                  - Matching job type to machine type runs much faster
                  - Mismatch runs much slower
                """
            )

        with right_col:
            st.markdown(f"### {metric_display[selected_metric]} Comparison")
            
            # Structure the data
            chart_df = pd.DataFrame({
                "Algorithm": list(results.keys()),
                "Metric Value": [results[algorithm][selected_metric] for algorithm in results.keys()]
            })
            
            # Define the exact order and colors
            sort_order = ["Random", "FCFS", "SJF", "Agentic AI"]
            
            # Create the Base Bar Chart
            bars = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('Algorithm', sort=sort_order, axis=alt.Axis(labelAngle=0, title="Scheduling Algorithm")),
                y=alt.Y('Metric Value', title=metric_display[selected_metric]),
                color=alt.Color(
                    'Algorithm', 
                    legend=None, 
                    scale=alt.Scale(domain=sort_order, range=['#e74c3c', '#3498db', '#9b59b6', '#2ecc71'])
                ),
                tooltip=['Algorithm', 'Metric Value']
            )

            # Create the Text Labels Layer
            text = bars.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,  # Nudges the text slightly above the bar
                fontWeight='bold',
                fontSize=14
            ).encode(
                text=alt.Text('Metric Value:Q', format='.2f') 
            )

            # Combine bars and text, and set height
            chart = (bars + text).properties(height=500)
            
            # Render the chart
            st.altair_chart(chart, use_container_width=True)

    with tab2:
        st.markdown("Raw per-scenario metrics for every algorithm.")
        df = pd.DataFrame(raw_data)
        st.dataframe(df, use_container_width=True, height=400)

    with tab3:
        st.markdown("Average metrics across all test scenarios.")
        avg_df = pd.DataFrame(average_metrics).set_index("Algorithm")
        st.dataframe(avg_df.style.format("{:.2f}"), use_container_width=True, height=420)