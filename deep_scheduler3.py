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

# ==========================================
# 1. The Environment (Vision & Balance Upgrade)
# ==========================================
class MassiveSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jobs = 100        
        self.num_machines = 10 
        self.lookahead = 10  # <-- UPGRADED: AI now sees 10 jobs ahead
        
        self.action_space = spaces.Discrete(self.num_machines)
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.num_machines + self.lookahead,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.job_times = [np.random.randint(1, 51) for _ in range(self.num_jobs)]
        self.current_job_idx = 0
        self.machine_times = [0.0] * self.num_machines
        return self._get_state(), {}

    def _get_state(self):
        upcoming_jobs = self.job_times[self.current_job_idx : self.current_job_idx + self.lookahead]
        
        while len(upcoming_jobs) < self.lookahead:
            upcoming_jobs.append(0.0)
            
        norm_jobs = [float(j) / 50.0 for j in upcoming_jobs]
        norm_machines = [m / 1000.0 for m in self.machine_times] 
        
        state = norm_jobs + norm_machines
        return np.array(state, dtype=np.float32)

    def step(self, action):
        job_time = self.job_times[self.current_job_idx]
        
        # Capture state BEFORE action
        old_makespan = max(self.machine_times)
        old_imbalance = np.std(self.machine_times) # Standard Deviation measures factory balance
        
        # Take action
        self.machine_times[action] += job_time
        self.current_job_idx += 1
        
        # Capture state AFTER action
        new_makespan = max(self.machine_times)
        new_imbalance = np.std(self.machine_times)
        
        # <-- UPGRADED REWARD SHAPING -->
        # Penalize for raising the max time AND penalize for making machines uneven
        reward = -(new_makespan - old_makespan) - (new_imbalance - old_imbalance)
        
        terminated = bool(self.current_job_idx >= self.num_jobs)
        return self._get_state(), reward, terminated, False, {}

# ==========================================
# 2. The Neural Network (256-Neuron Brain)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # <-- UPGRADED: 256 neurons to process 10-job sequences
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 3. Traditional Baseline Algorithms
# ==========================================
def run_random_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    for job in jobs:
        machines[random.randint(0, num_machines - 1)] += job
    return max(machines)

def run_round_robin_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    for i, job in enumerate(jobs):
        machines[i % num_machines] += job
    return max(machines)

def run_fcfs_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    for job in jobs:
        min_machine_idx = machines.index(min(machines))
        machines[min_machine_idx] += job
    return max(machines)

def run_sjf_scheduler(jobs, num_machines):
    machines = [0.0] * num_machines
    sorted_jobs = sorted(jobs)
    for job in sorted_jobs:
        min_machine_idx = machines.index(min(machines))
        machines[min_machine_idx] += job
    return max(machines)

def run_ddqn_evaluation(env, policy_net, jobs):
    env.reset()
    env.job_times = jobs.copy() 
    state = env._get_state()
    terminated = False
    
    while not terminated:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()
        state, _, terminated, _, _ = env.step(action)
        
    return max(env.machine_times)

# ==========================================
# 4. Training the DDQN
# ==========================================
print("Training Advanced 'Balanced' DDQN Agent...")
env = MassiveSchedulingEnv()
policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4) 
memory = deque(maxlen=100000) # Increased buffer
batch_size = 64
gamma = 0.99
tau = 0.005
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.997 # Slower decay for more exploration
episodes = 3000 # Increased training time

for episode in range(episodes):
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()
                
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
                
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if (episode + 1) % 500 == 0:
        print(f"Training Episode {episode + 1}/{episodes} Complete...")

print("Training Complete!\n")
torch.save(policy_net.state_dict(), "advanced_ddqn.pth")

# ==========================================
# 5. The Ultimate Showdown (Evaluation)
# ==========================================
print("Running Algorithm Showdown...")
test_scenarios = [[np.random.randint(1, 51) for _ in range(env.num_jobs)] for _ in range(10)]

results = {
    "Random": 0, 
    "Round-Robin": 0, 
    "FCFS": 0,
    "SJF": 0,
    "DDQN (Yours)": 0
}

for i, jobs in enumerate(test_scenarios):
    results["Random"] += run_random_scheduler(jobs, env.num_machines)
    results["Round-Robin"] += run_round_robin_scheduler(jobs, env.num_machines)
    results["FCFS"] += run_fcfs_scheduler(jobs, env.num_machines)
    results["SJF"] += run_sjf_scheduler(jobs, env.num_machines)
    results["DDQN (Yours)"] += run_ddqn_evaluation(env, policy_net, jobs)

for key in results:
    results[key] /= len(test_scenarios)

print("\n--- Final Average Makespans (Lower is Better) ---")
for key, value in results.items():
    print(f"{key}: {value:.1f}s")

# ==========================================
# 6. Plot the Comparison
# ==========================================
labels = list(results.keys())
values = list(results.values())

plt.figure(figsize=(10, 6))
colors = ['red', 'orange', 'gray', 'purple', 'green']
bars = plt.bar(labels, values, color=colors)

plt.title("Scheduling Algorithms Comparison (Average over 10 Scenarios)")
plt.ylabel("Makespan (Seconds) - Lower is Better")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 3, f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')

# Automatically scale the Y-axis to make differences more visible
plt.ylim(bottom=min(values) - 50, top=max(values) + 50) 
plt.tight_layout() 
plt.show()