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
# 1. The Environment (Normalized & Shaped)
# ==========================================
class MassiveSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_jobs = 100        
        self.num_machines = 10     
        self.action_space = spaces.Discrete(self.num_machines)
        # The observation space bounds can now be smaller since we are normalizing to roughly 0.0 - 5.0
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.num_machines + 1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.job_times = [np.random.randint(1, 51) for _ in range(self.num_jobs)]
        self.current_job_idx = 0
        self.machine_times = [0.0] * self.num_machines
        self.current_makespan = 0.0 # Track makespan for reward shaping
        return self._get_state(), {}

    def _get_state(self):
        if self.current_job_idx < self.num_jobs:
            current_job_size = float(self.job_times[self.current_job_idx])
        else:
            current_job_size = 0.0
            
        # FIX 1: NORMALIZE THE STATE
        # Neural networks need small numbers. We divide by realistic maximums.
        norm_job = current_job_size / 50.0  # Max job size is 50
        norm_machines = [m / 1000.0 for m in self.machine_times] 
        
        state = [norm_job] + norm_machines
        return np.array(state, dtype=np.float32)

    def step(self, action):
        job_time = self.job_times[self.current_job_idx]
        self.machine_times[action] += job_time
        self.current_job_idx += 1
        
        # REWARD SHAPING: Calculate how much this specific action increased the total makespan
        new_makespan = max(self.machine_times)
        reward = -(new_makespan - self.current_makespan) 
        self.current_makespan = new_makespan
        
        terminated = bool(self.current_job_idx >= self.num_jobs)
            
        return self._get_state(), reward, terminated, False, {}

# ==========================================
# 2. The Neural Network
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 3. Hyperparameters & Setup
# ==========================================
env = MassiveSchedulingEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n            

# Initialize Policy AND Target Networks
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # Target net doesn't train

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4) 

memory = deque(maxlen=50000)
batch_size = 64
gamma = 0.99
tau = 0.005 # Soft update rate for target network
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05
episodes = 2000

episode_makespans = []

# ==========================================
# 4. Training Loop (Double DQN)
# ==========================================
print("Starting Double DQN (DDQN) Training...")

for episode in range(episodes):
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            
        next_state, reward, terminated, _, _ = env.step(action)
        
        memory.append((state, action, reward, next_state, terminated))
        state = next_state
        
        # --- THE DEEP LEARNING STEP ---
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            
            states = torch.FloatTensor(np.array([b[0] for b in batch]))
            actions = torch.LongTensor(np.array([b[1] for b in batch])).unsqueeze(1)
            rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1)
            next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
            dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1)
            
            # Predict current Q-values
            current_q = policy_net(states).gather(1, actions)
            
            # FIX 2: DOUBLE DQN (DDQN) LOGIC
            with torch.no_grad():
                # Policy Net selects the best action for the next state
                best_next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                
                # Target Net evaluates that specific action
                max_next_q = target_net(next_states).gather(1, best_next_actions)
                
                target_q = rewards + (1 - dones) * gamma * max_next_q
                
            # Huber Loss & Gradient Clipping for stability
            loss = F.smooth_l1_loss(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Soft update the target network
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
                
        if terminated:
            episode_makespans.append(-max(env.machine_times))
            
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (episode + 1) % 100 == 0:
        avg_makespan = np.mean(episode_makespans[-100:])
        print(f"Episode {episode + 1} | Epsilon: {epsilon:.3f} | Avg Makespan: {-avg_makespan:.1f}s")

print("\nTraining Complete!")
torch.save(policy_net.state_dict(), "ddqn_scheduler.pth")
print("Model saved to ddqn_scheduler.pth!")
# ==========================================
# 5. Visualization
# ==========================================
plt.plot(episode_makespans)
plt.title("Double DQN (DDQN) Performance")
plt.xlabel("Training Episode")
plt.ylabel("Reward (Negative Final Makespan)")
plt.show()
