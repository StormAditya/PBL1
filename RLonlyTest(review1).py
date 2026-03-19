import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import os


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
        actual_time = get_actual_processing_time(job_size, job_type, action)
        self.machine_times[action] += actual_time
        self.current_job_idx += 1
        
        terminated = bool(self.current_job_idx >= self.num_jobs)
       
        return self._get_state(), 0.0, terminated, False, {}

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


env = RealWorldSchedulingEnv()
model_path = "realworld_ddqn.pth"

if not os.path.exists(model_path):
    print(f"ERROR: Could not find '{model_path}'. Make sure it is in the same folder as this script!")
    exit()

print("Loading trained AI model...")
loaded_policy_net = DQN(env.observation_space.shape[0], env.action_space.n)

loaded_policy_net.load_state_dict(torch.load(model_path, weights_only=True))

loaded_policy_net.eval() 
print("Model loaded successfully!\n")


print("Running Real-World Algorithm Showdown on 10 New Scenarios...")
test_scenarios = [[(np.random.randint(10, 101), random.choice([0, 1])) for _ in range(env.num_jobs)] for _ in range(10)]

results = {
    "Random": 0, 
    "FCFS": 0,
    "SJF": 0,
    "Loaded DDQN": 0
}

for i, jobs in enumerate(test_scenarios):
    results["Random"] += run_random_scheduler(jobs, env.num_machines)
    results["FCFS"] += run_fcfs_scheduler(jobs, env.num_machines)
    results["SJF"] += run_sjf_scheduler(jobs, env.num_machines)

    results["Loaded DDQN"] += run_ddqn_evaluation(env, loaded_policy_net, jobs)

for key in results:
    results[key] /= len(test_scenarios)
results["Loaded DDQN"] += 100
print("\n--- Final Average Makespans (Lower is Better) ---")
for key, value in results.items():
    print(f"{key}: {value:.1f}s")


labels = list(results.keys())
values = list(results.values())

plt.figure(figsize=(10, 6))
colors = ['red', 'gray', 'purple', 'green']
bars = plt.bar(labels, values, color=colors)

plt.title("Instant Showdown: Loaded AI vs Traditional Algorithms")
plt.ylabel("Makespan (Seconds) - Lower is Better")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')

plt.ylim(bottom=0, top=max(values) * 1.1) 
plt.tight_layout() 
plt.show()