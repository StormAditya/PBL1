import torch
import torch.nn as nn
import numpy as np
import random
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

# --- 4. EXECUTION & SHOWDOWN ---

if __name__ == "__main__":
    print("Loading Trained Agentic AI from 'agentic_ddqn.pth'...")
    env = AgenticSchedulingEnv()
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
    
    try:
        policy_net.load_state_dict(torch.load("single_agentic_ddqn.pth", weights_only=True))
        policy_net.eval()
        print("Model loaded successfully!\n")
    except FileNotFoundError:
        print("ERROR: 'agentic_ddqn.pth' not found. Make sure you've run the training script first!")
        exit()

    print("Running Algorithm Showdown (10 Test Scenarios)...")
    
    # Generate 10 identical test scenarios for all algorithms to ensure fairness
    test_scenarios = [[(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(env.num_jobs)] for _ in range(10)]

    results = {
        "Random": 0, 
        "FCFS": 0,
        "SJF (Shortest Job First)": 0,
        "Agentic AI ": 0
    }

    for i, jobs in enumerate(test_scenarios):
        results["Random"] += run_random_scheduler(jobs, env.num_machines)
        results["FCFS"] += run_fcfs_scheduler(jobs, env.num_machines)
        results["SJF (Shortest Job First)"] += run_sjf_scheduler(jobs, env.num_machines)
        results["Agentic AI "] += run_agentic_evaluation(env, policy_net, jobs)

    # Average out the results
    for key in results:
        results[key] /= len(test_scenarios)

    print("\n--- Final Average Makespans (Lower is Better) ---")
    for key, value in results.items():
        print(f"{key}: {value:.1f}s")

    # --- VISUALIZATION ---
    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    colors = ['red', 'gray', 'purple', 'green']
    bars = plt.bar(labels, values, color=colors)

    plt.title("Real-World Factory Scheduling Showdown")
    plt.ylabel("Makespan (Seconds) - Lower is Better")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')

    plt.ylim(bottom=0, top=max(values) * 1.1) 
    plt.tight_layout() 
    plt.show()