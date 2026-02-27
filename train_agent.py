import numpy as np
import random
import matplotlib.pyplot as plt # [NEW] Import matplotlib
from scheduler_env import JobSchedulingEnv

# 1. Initialize Environment
env = JobSchedulingEnv()

# 2. Hyperparameters
alpha = 0.1          
gamma = 0.9          
epsilon = 1.0        
epsilon_decay = 0.99 
min_epsilon = 0.01   
episodes = 5000      

q_table = {}

def get_q_value(state, action):
    return q_table.get((state, action), 0.0)

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() 
    else:
        q_values = [get_q_value(state, a) for a in range(env.num_machines)]
        return np.argmax(q_values) 

# 3. Training Loop
print("Starting Training...")

# [NEW] List to keep track of the final reward for each episode
episode_rewards = [] 

for episode in range(episodes):
    current_job, _ = env.reset()
    state = (current_job, 0, 0) 
    
    terminated = False
    
    while not terminated:
        action = choose_action(state)
        next_job, reward, terminated, truncated, info = env.step(action)
        next_state = (next_job, info["machine_times"][0], info["machine_times"][1])
        
        max_next_q = max([get_q_value(next_state, a) for a in range(env.num_machines)])
        current_q = get_q_value(state, action)
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        q_table[(state, action)] = new_q
        
        state = next_state
        
        # [NEW] Save the reward when the episode finishes
        if terminated:
            episode_rewards.append(reward)
            
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training Complete!\n")

# [NEW] Generate and display the learning curve graph
plt.plot(episode_rewards)
plt.title("AI Scheduler Learning Curve")
plt.xlabel("Training Episode")
plt.ylabel("Reward (Negative Makespan)")
plt.show()

# 4. Test the Trained Agent
print("Testing the Trained Agent...")
current_job, _ = env.reset()
state = (current_job, 0, 0)
terminated = False

while not terminated:
    q_values = [get_q_value(state, a) for a in range(env.num_machines)]
    action = np.argmax(q_values)
    
    print(f"Assigning Job {current_job} (Time: {env.job_times[current_job]}s) to Machine {action}")
    
    next_job, reward, terminated, truncated, info = env.step(action)
    state = (next_job, info["machine_times"][0], info["machine_times"][1])
    current_job = next_job 

print(f"\nFinal State Reached!")
print(f"Machine Workloads: {info['machine_times']}")
print(f"Final Reward (Negative Makespan): {reward}")