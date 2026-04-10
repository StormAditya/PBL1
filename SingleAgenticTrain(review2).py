import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

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
        self.pool_size = 5 # The agent can see and pick from 5 jobs at once
        
        # Action space: Pool Size (Which job to pick) * Num Machines (Where to put it)
        self.action_space = spaces.Discrete(self.pool_size * self.num_machines)
        
        # State: (5 jobs * 2 features) + 10 machine times
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
        
        # Failsafe penalty if it picks an empty slot
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

# --- TRADITIONAL SCHEDULERS ---

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

def get_valid_actions(env):
    """Returns a boolean mask of valid actions (False for picking empty pool slots)"""
    mask = torch.ones(env.action_space.n, dtype=torch.bool)
    for i in range(env.action_space.n):
        job_idx = i // env.num_machines
        if job_idx >= len(env.job_pool):
            mask[i] = False
    return mask

def run_agentic_evaluation(env, policy_net, jobs):
    env.reset()
    # Load the exact test jobs into the agent's environment
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
            
            # Action Masking: Force Q-values of illegal moves to negative infinity
            valid_mask = get_valid_actions(env)
            q_values[0, ~valid_mask] = -float('inf')
            
            action = q_values.argmax().item()
            
        state, _, terminated, _, _ = env.step(action)
        
    return compute_schedule_metrics(env.machine_times, env.schedule_records, env.num_machines)

# --- TRAINING ---
print("Training Agentic AI (Active Picker)")
env = AgenticSchedulingEnv()
policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4) 
memory = deque(maxlen=100000) 
batch_size = 64
gamma = 0.99
tau = 0.005
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.997 
episodes = 3500 
episode_makespans = [] 


for episode in range(episodes):
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        valid_mask = get_valid_actions(env)
        
        if random.random() < epsilon:
            # Sample only from valid actions
            valid_actions = torch.where(valid_mask)[0].tolist()
            action = random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                q_values[0, ~valid_mask] = -float('inf') # Mask invalid actions
                action = q_values.argmax().item()
                
        next_state, reward, terminated, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, terminated))
        state = next_state
        
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states = torch.FloatTensor(np.array([b[0] for b in batch]))
            actions = torch.LongTensor(np.array([b[1] for b in batch])).unsqueeze(1)
            rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1)
            next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
            dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1)
            
            current_q = policy_net(states).gather(1, actions)
            
            with torch.no_grad():
                best_next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                max_next_q = target_net(next_states).gather(1, best_next_actions)
                target_q = rewards + (1 - dones) * gamma * max_next_q
                
            loss = F.smooth_l1_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
        
        if terminated:
            episode_makespans.append(max(env.machine_times))
                
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (episode + 1) % 100 == 0:
        avg_makespan = np.mean(episode_makespans[-100:])
        print(f"Episode {episode + 1:4d}/{episodes} | Epsilon: {epsilon:.3f} | Avg Makespan: {avg_makespan:.1f}s")

print("Training Complete!\n")

torch.save(policy_net.state_dict(), "single_agentic_ddqn.pth")
print("--> Model successfully saved to 'single_agentic_ddqn.pth'!\n")


# --- ALGORITHM SHOWDOWN ---
print("Running Real-World Algorithm Showdown...")

# Generate 10 identical test scenarios for all algorithms
test_scenarios = [[(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(env.num_jobs)] for _ in range(10)]

metrics_to_report = [
    "makespan",
    "avg_waiting_time",
    "avg_completion_time",
    "throughput",
    "load_std",
    "utilization",
    "affinity_rate",
]

results = {
    "Random": {metric: 0.0 for metric in metrics_to_report},
    "FCFS": {metric: 0.0 for metric in metrics_to_report},
    "SJF": {metric: 0.0 for metric in metrics_to_report},
    "Agentic AI (Yours)": {metric: 0.0 for metric in metrics_to_report},
}

for i, jobs in enumerate(test_scenarios):
    random_metrics = run_random_scheduler(jobs, env.num_machines)
    fcfs_metrics = run_fcfs_scheduler(jobs, env.num_machines)
    sjf_metrics = run_sjf_scheduler(jobs, env.num_machines)
    agentic_metrics = run_agentic_evaluation(env, policy_net, jobs)

    for metric in metrics_to_report:
        results["Random"][metric] += random_metrics[metric]
        results["FCFS"][metric] += fcfs_metrics[metric]
        results["SJF"][metric] += sjf_metrics[metric]
        results["Agentic AI (Yours)"][metric] += agentic_metrics[metric]

for key in results:
    for metric in metrics_to_report:
        results[key][metric] /= len(test_scenarios)

print("\n--- Final Average Metrics Across Scenarios ---")
for key, metric_values in results.items():
    print(f"\n{key}")
    print(f"  Makespan (Lower Better): {metric_values['makespan']:.1f}s")
    print(f"  Avg Waiting Time (Lower Better): {metric_values['avg_waiting_time']:.1f}s")
    print(f"  Avg Completion Time (Lower Better): {metric_values['avg_completion_time']:.1f}s")
    print(f"  Throughput (Higher Better): {metric_values['throughput']:.3f} jobs/s")
    print(f"  Load Std Dev (Lower Better): {metric_values['load_std']:.2f}")
    print(f"  Utilization (Higher Better): {metric_values['utilization'] * 100:.1f}%")
    print(f"  Affinity Match Rate (Higher Better): {metric_values['affinity_rate']:.1f}%")

# --- VISUALIZATION ---
labels = list(results.keys())
values = [results[label]["makespan"] for label in labels]

plt.figure(figsize=(10, 6))
colors = ['red', 'gray', 'purple', 'green']
bars = plt.bar(labels, values, color=colors)

plt.title("Real-World Factory (Hardware Affinities) - Showdown")
plt.ylabel("Makespan (Seconds) - Lower is Better")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')

plt.ylim(bottom=0, top=max(values) * 1.1) 
plt.tight_layout() 
plt.show()
