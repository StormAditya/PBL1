import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ==========================================
# 1. The Massive Dynamic Environment
# ==========================================
class MassiveSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jobs = 100        # Increased to 100 jobs
        self.num_machines = 10     # Increased to 10 machines
        self.action_space = spaces.Discrete(self.num_machines)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize jobs heavily between 1 and 50 seconds for EVERY episode
        self.job_times = [np.random.randint(1, 51) for _ in range(self.num_jobs)]
        self.current_job_idx = 0
        self.machine_times = [0] * self.num_machines
        
        return self._get_state(), {}

    def _get_state(self):
        """State is now: (Incoming Job Size, M0_load, M1_load, ..., M9_load)"""
        if self.current_job_idx < self.num_jobs:
            current_job_size = self.job_times[self.current_job_idx]
        else:
            current_job_size = 0
            
        # We use a tuple so it can be hashed as a dictionary key in our Q-table
        return tuple([current_job_size] + self.machine_times)

    def step(self, action):
        job_time = self.job_times[self.current_job_idx]
        self.machine_times[action] += job_time
        self.current_job_idx += 1
        
        terminated = bool(self.current_job_idx >= self.num_jobs)
        reward = 0
        
        if terminated:
            # Reward is still the negative makespan
            reward = -max(self.machine_times)
            
        info = {"machine_times": self.machine_times}
        return self._get_state(), reward, terminated, False, info

# ==========================================
# 2. Training Setup
# ==========================================
env = MassiveSchedulingEnv()

alpha = 0.1          
gamma = 0.9          
epsilon = 1.0        
epsilon_decay = 0.999 # Slower decay because there are way more states to explore
min_epsilon = 0.05   
episodes = 10000      # Increased episodes 

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
    
    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(episode_rewards[-1000:])
        print(f"Episode {episode + 1} | Epsilon: {epsilon:.3f} | Avg Reward (Makespan): {avg_reward:.1f}")

print("\nTraining Complete! Q-Table size:", len(q_table))

# ==========================================
# 4. Visualization
# ==========================================
plt.plot(episode_rewards)
plt.title("Massive AI Scheduler Learning Curve")
plt.xlabel("Training Episode")
plt.ylabel("Reward (Negative Makespan)")
plt.show()