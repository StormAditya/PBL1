import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ==========================================
# 1. The Reduced Environment
# ==========================================
class ReducedSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jobs = 10         # Reduced to 10 jobs
        self.num_machines = 2      # Reduced to 2 machines
        self.action_space = spaces.Discrete(self.num_machines)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random jobs between 1 and 10 seconds
        self.job_times = [np.random.randint(1, 11) for _ in range(self.num_jobs)]
        self.current_job_idx = 0
        self.machine_times = [0] * self.num_machines
        
        return self._get_state(), {}

    def _get_state(self):
        if self.current_job_idx < self.num_jobs:
            current_job_size = self.job_times[self.current_job_idx]
        else:
            current_job_size = 0
            
        # State: (Incoming Job Size, M0_load, M1_load)
        return tuple([current_job_size] + self.machine_times)

    def step(self, action):
        job_time = self.job_times[self.current_job_idx]
        self.machine_times[action] += job_time
        self.current_job_idx += 1
        
        terminated = bool(self.current_job_idx >= self.num_jobs)
        reward = 0
        
        if terminated:
            # Reward is the negative makespan
            reward = -max(self.machine_times)
            
        info = {"machine_times": self.machine_times}
        return self._get_state(), reward, terminated, False, info

# ==========================================
# 2. Training Setup
# ==========================================
env = ReducedSchedulingEnv()

alpha = 0.1          
gamma = 0.9          
epsilon = 1.0        
epsilon_decay = 0.9995 # Slow decay to ensure it explores the 25,000 states
min_epsilon = 0.01   
episodes = 15000       # Give it enough time to hit repeating states

q_table = {}
episode_rewards = []

def get_q_value(state, action):
    return q_table.get((state, action), 0.0)

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() 
    else:
        q_values = [get_q_value(state, a) for a in range(env.num_machines)]
        return np.argmax(q_values) 

# ==========================================
# 3. Training Loop
# ==========================================
print(f"Training on {env.num_machines} machines and {env.num_jobs} random jobs...")

for episode in range(episodes):
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        action = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        max_next_q = max([get_q_value(next_state, a) for a in range(env.num_machines)])
        current_q = get_q_value(state, action)
        
        # Bellman Equation update
        q_table[(state, action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)
        state = next_state
        
        if terminated:
            episode_rewards.append(reward)
            
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (episode + 1) % 1500 == 0:
        avg_reward = np.mean(episode_rewards[-1500:])
        print(f"Episode {episode + 1} | Epsilon: {epsilon:.3f} | Avg Makespan: {-avg_reward:.1f}s")

print(f"\nTraining Complete! Final Q-Table size: {len(q_table)} states discovered.")

# ==========================================
# 4. Visualization
# ==========================================
# To make the graph readable, we smooth it out by taking a moving average
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_rewards = moving_average(episode_rewards)

plt.plot(smoothed_rewards)
plt.title("Reduced Tabular Q-Learning Curve (Smoothed)")
plt.xlabel("Training Episode")
plt.ylabel("Reward (Negative Makespan)")
plt.show()