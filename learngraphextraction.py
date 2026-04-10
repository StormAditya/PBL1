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
        self.pool_size = 5 
        
        self.action_space = spaces.Discrete(self.pool_size * self.num_machines)
        self.observation_space = spaces.Box(
            low=0, high=20, 
            shape=((self.pool_size * 2) + self.num_machines,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # CHANGED: Jobs are now larger (50 to 200) to naturally push makespan > 500
        all_jobs = [(np.random.randint(50, 201), random.choice([0, 1])) for _ in range(self.num_jobs)]
        
        self.job_pool = all_jobs[:self.pool_size]
        self.remaining_jobs = all_jobs[self.pool_size:]
        self.machine_times = [0.0] * self.num_machines
        return self._get_state(), {}

    def _get_state(self):
        state_features = []
        for i in range(self.pool_size):
            if i < len(self.job_pool):
                # CHANGED: Normalized by 200.0 instead of 100.0 to match new job sizes
                state_features.append(self.job_pool[i][0] / 200.0)
                state_features.append(float(self.job_pool[i][1]))
            else:
                state_features.extend([0.0, 0.0])
                
        # CHANGED: Normalized by 3000.0 because total machine times will be much higher now
        norm_machines = [m / 3000.0 for m in self.machine_times] 
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

# --- TRADITIONAL SCHEDULERS ---

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

# --- TRAINING ---
print("Training Standard DQN Agent...")
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

# 2000 Episodes as requested
episodes = 2000 
episode_makespans = [] 

for episode in range(episodes):
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        valid_mask = get_valid_actions(env)
        
        if random.random() < epsilon:
            valid_actions = torch.where(valid_mask)[0].tolist()
            action = random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                q_values[0, ~valid_mask] = -float('inf') 
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
            
            # Standard DQN Target Calculation
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
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

torch.save(policy_net.state_dict(), "single_agentic_standard_dqn.pth")
print("--> Model successfully saved to 'single_agentic_standard_dqn.pth'!\n")


# --- ALGORITHM SHOWDOWN ---
print("Running Real-World Algorithm Showdown...")

# CHANGED: Ensure the test scenarios also generate the larger 50-200 jobs
test_scenarios = [[(np.random.randint(50, 201), random.choice([0, 1])) for _ in range(env.num_jobs)] for _ in range(10)]

results = {
    "Random": 0, 
    "FCFS": 0,
    "SJF": 0,
    "Standard DQN": 0
}

for i, jobs in enumerate(test_scenarios):
    results["Random"] += run_random_scheduler(jobs, env.num_machines)
    results["FCFS"] += run_fcfs_scheduler(jobs, env.num_machines)
    results["SJF"] += run_sjf_scheduler(jobs, env.num_machines)
    results["Standard DQN"] += run_agentic_evaluation(env, policy_net, jobs)

for key in results:
    results[key] /= len(test_scenarios)

print("\n--- Final Average Makespans (Lower is Better) ---")
for key, value in results.items():
    print(f"{key}: {value:.1f}s")

# --- VISUALIZATION ---
plt.figure(figsize=(15, 6))

# --- Plot 1: Algorithm Showdown (Bar Chart) ---
plt.subplot(1, 2, 1)
labels = list(results.keys())
values = list(results.values())
colors = ['red', 'gray', 'purple', 'green']
bars = plt.bar(labels, values, color=colors)

plt.title("Real-World Factory Showdown\n(Lower is Better)", fontweight='bold')
plt.ylabel("Makespan (Seconds)")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')

# Graph limits auto-scale from 0 to slightly above the tallest bar
plt.ylim(bottom=0, top=max(values) * 1.1) 

# --- Plot 2: Learning Curve (Raw Data Only) ---
plt.subplot(1, 2, 2)

plt.plot(episode_makespans, label='Raw Episode Makespan', color='blue', linewidth=1.5)

plt.title("Standard DQN Learning Curve Over Time\n(Raw Data Only)", fontweight='bold')
plt.xlabel("Training Episode")
plt.ylabel("Makespan (Seconds)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout() 
plt.show()