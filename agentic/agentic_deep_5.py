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
# 1. Environment with Cooperative Rewards
# ==========================================

def get_actual_processing_time(job_size, job_type, machine_idx):
    is_gpu_node = machine_idx >= 7
    is_fast_cpu = machine_idx < 3
    base_speed = 0.6 if is_fast_cpu else 1.0
    machine_type = 1 if is_gpu_node else 0
    # Penalty of 3.0x if job type doesn't match machine hardware
    return (job_size * base_speed) * (0.5 if job_type == machine_type else 3.0)

class CooperativeSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jobs = 100
        self.num_machines = 10
        self.lookahead = 10
        self.action_space = spaces.Discrete(self.num_machines)
        self.observation_space = spaces.Box(low=0, high=20, shape=(self.num_machines + (self.lookahead * 2),), dtype=np.float32)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.jobs = [(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(self.num_jobs)]
        self.current_job_idx = 0
        self.machine_times = [0.0] * self.num_machines
        return self._get_state(), {}

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

    def step(self, action, agent_id):
        job_size, job_type = self.jobs[self.current_job_idx]
        old_makespan = max(self.machine_times)
        
        # Track cluster-specific balance
        old_std = np.std(self.machine_times[:7]) if agent_id == 0 else np.std(self.machine_times[7:])

        actual_time = get_actual_processing_time(job_size, job_type, action)
        self.machine_times[action] += actual_time
        self.current_job_idx += 1
        
        new_makespan = max(self.machine_times)
        new_std = np.std(self.machine_times[:7]) if agent_id == 0 else np.std(self.machine_times[7:])

        # COOPERATIVE REWARD: -Delta Makespan (Global) - Delta Variance (Local)
        reward = -(new_makespan - old_makespan) * 2.5 - (new_std - old_std)
        
        terminated = bool(self.current_job_idx >= self.num_jobs)
        return self._get_state(), reward, terminated, False, {}

# ==========================================
# 2. Dueling DQN Architecture (The Brains)
# ==========================================

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))

# ==========================================
# 3. Training Loop (2 Specialized Agents)
# ==========================================

env = CooperativeSchedulingEnv()
# Agent 0 handles CPU Jobs (Type 0), Agent 1 handles GPU Jobs (Type 1)
agents = [DuelingDQN(env.observation_space.shape[0], env.action_space.n) for _ in range(2)]
targets = [DuelingDQN(env.observation_space.shape[0], env.action_space.n) for _ in range(2)]
optimizers = [optim.Adam(a.parameters(), lr=1e-4) for a in agents]
memories = [deque(maxlen=100000) for _ in range(2)]

for i in range(2): targets[i].load_state_dict(agents[i].state_dict())

epsilon, episodes = 1.0, 3000
batch_size, gamma, tau = 64, 0.99, 0.005

print("Starting 2-Agent Cooperative Training...")

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        job_type = env.jobs[env.current_job_idx][1]
        active_id = job_type 
        
        # Epsilon-Greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agents[active_id](torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        
        next_state, reward, done, _, _ = env.step(action, active_id)
        memories[active_id].append((state, action, reward, next_state, done))
        state = next_state
        
        # Optimize the agent that just took the action
        if len(memories[active_id]) > batch_size:
            b = random.sample(memories[active_id], batch_size)
            s_j = torch.FloatTensor(np.array([x[0] for x in b]))
            a_j = torch.LongTensor(np.array([x[1] for x in b])).unsqueeze(1)
            r_j = torch.FloatTensor(np.array([x[2] for x in b])).unsqueeze(1)
            ns_j = torch.FloatTensor(np.array([x[3] for x in b]))
            d_j = torch.FloatTensor(np.array([x[4] for x in b])).unsqueeze(1)

            curr_q = agents[active_id](s_j).gather(1, a_j)
            next_q = targets[active_id](ns_j).max(1)[0].unsqueeze(1)
            target_q = r_j + (1 - d_j) * gamma * next_q
            
            loss = F.smooth_l1_loss(curr_q, target_q)
            optimizers[active_id].zero_grad()
            loss.backward()
            optimizers[active_id].step()

            for t, p in zip(targets[active_id].parameters(), agents[active_id].parameters()):
                t.data.copy_(tau * p.data + (1 - tau) * t.data)

    epsilon = max(0.05, epsilon * 0.997)
    if (ep + 1) % 500 == 0:
        print(f"Episode {ep+1}/{episodes} | Epsilon: {epsilon:.2f}")

# ==========================================
# 4. Showdown (MARL vs Traditional Models)
# ==========================================

def run_baselines(jobs):
    # Random
    m_rand = [0.0]*10
    for sz, t in jobs:
        idx = random.randint(0, 9)
        m_rand[idx] += get_actual_processing_time(sz, t, idx)
    
    # FCFS (Min Load)
    m_fcfs = [0.0]*10
    for sz, t in jobs:
        idx = m_fcfs.index(min(m_fcfs))
        m_fcfs[idx] += get_actual_processing_time(sz, t, idx)
        
    # SJF (Shortest Job First)
    m_sjf = [0.0]*10
    sorted_jobs = sorted(jobs, key=lambda x: x[0])
    for sz, t in sorted_jobs:
        idx = m_sjf.index(min(m_sjf))
        m_sjf[idx] += get_actual_processing_time(sz, t, idx)
        
    return max(m_rand), max(m_fcfs), max(m_sjf)

def run_marl(jobs):
    env.reset()
    env.jobs = jobs
    state = env._get_state()
    done = False
    while not done:
        j_type = env.jobs[env.current_job_idx][1]
        with torch.no_grad():
            action = agents[j_type](torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        state, _, done, _, _ = env.step(action, j_type)
    return max(env.machine_times)

results = {"Random": [], "FCFS": [], "SJF": [], "2-Agent MARL": []}

for _ in range(20):
    test_jobs = [(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(100)]
    r, f, s = run_baselines(test_jobs)
    m = run_marl(test_jobs)
    results["Random"].append(r); results["FCFS"].append(f)
    results["SJF"].append(s); results["2-Agent MARL"].append(m)

# Final Plot
avg_res = {k: np.mean(v) for k, v in results.items()}
plt.figure(figsize=(10, 6))
bars = plt.bar(avg_res.keys(), avg_res.values(), color=['red', 'gray', 'orange', 'green'])
plt.ylabel("Avg Makespan (Seconds)")
plt.title("Algorithm Showdown: Traditional vs Multi-Agent RL")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}s", ha='center', fontweight='bold')

plt.show()