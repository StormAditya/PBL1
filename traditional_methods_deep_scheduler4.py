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

class RealWorldSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jobs = 100        
        self.num_machines = 10 
        self.lookahead = 10  
        
        self.action_space = spaces.Discrete(self.num_machines)
        

        self.observation_space = spaces.Box(low=0, high=20, shape=(self.num_machines + (self.lookahead * 2),), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
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
                state_jobs.append(self.jobs[idx][0] / 100.0) 
                state_jobs.append(float(self.jobs[idx][1]))  
            else:
                state_jobs.extend([0.0, 0.0]) 
                
        norm_machines = [m / 1000.0 for m in self.machine_times] 
        return np.array(state_jobs + norm_machines, dtype=np.float32)

    def step(self, action):
        job_size, job_type = self.jobs[self.current_job_idx]
        
        old_makespan = max(self.machine_times)
        old_imbalance = np.std(self.machine_times) 
        

        actual_time = get_actual_processing_time(job_size, job_type, action)
        
        self.machine_times[action] += actual_time
        self.current_job_idx += 1
        
        new_makespan = max(self.machine_times)
        new_imbalance = np.std(self.machine_times)
        
       
        reward = -(new_makespan - old_makespan) - (new_imbalance - old_imbalance)
        
        terminated = bool(self.current_job_idx >= self.num_jobs)
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

def run_ddqn_evaluation(env, policy_net, jobs):
    env.reset()
    env.jobs = jobs.copy() 
    state = env._get_state()
    terminated = False
    
    while not terminated:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()
        state, _, terminated, _, _ = env.step(action)
        
    return max(env.machine_times)


print("Training AI")
env = RealWorldSchedulingEnv()
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
        
        if terminated:
            episode_makespans.append(max(env.machine_times))
                
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (episode + 1) % 100 == 0:
        avg_makespan = np.mean(episode_makespans[-100:])
        print(f"Episode {episode + 1:4d}/{episodes} | Epsilon: {epsilon:.3f} | Avg Makespan: {avg_makespan:.1f}s")

print("Training Complete!\n")


torch.save(policy_net.state_dict(), "realworld_ddqn.pth")
print("--> Model successfully saved to 'realworld_ddqn.pth'!\n")




print("Running Real-World Algorithm Showdown...")

test_scenarios = [[(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(env.num_jobs)] for _ in range(10)]

results = {
    "Random": 0, 
    "FCFS": 0,
    "SJF": 0,
    "DDQN (Yours)": 0
}

for i, jobs in enumerate(test_scenarios):
    results["Random"] += run_random_scheduler(jobs, env.num_machines)
    results["FCFS"] += run_fcfs_scheduler(jobs, env.num_machines)
    results["SJF"] += run_sjf_scheduler(jobs, env.num_machines)
    results["DDQN (Yours)"] += run_ddqn_evaluation(env, policy_net, jobs)

for key in results:
    results[key] /= len(test_scenarios)

print("\n--- Final Average Makespans (Lower is Better) ---")
for key, value in results.items():
    print(f"{key}: {value:.1f}s")


labels = list(results.keys())
values = list(results.values())

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
