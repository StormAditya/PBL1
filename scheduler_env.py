import gymnasium as gym
from gymnasium import spaces
import numpy as np

class JobSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # The 5 jobs and their execution times (in seconds)
        self.job_times = [2, 4, 1, 5, 3]
        self.num_jobs = len(self.job_times)
        self.num_machines = 2
        
        # Action Space: The agent can choose Machine 0 or Machine 1
        self.action_space = spaces.Discrete(self.num_machines)
        
        # Observation (State) Space: Which job is currently being assigned?
        # States are 0, 1, 2, 3, 4 (the jobs), and 5 (terminal state/done)
        self.observation_space = spaces.Discrete(self.num_jobs + 1)
        
        # Initialize variables
        self.current_job = 0
        self.machine_times = [0, 0]

    def reset(self, seed=None, options=None):
        """Resets the environment for a new training episode."""
        super().reset(seed=seed)
        self.current_job = 0
        self.machine_times = [0, 0]
        
        # Return the initial state and an empty info dictionary
        return self.current_job, {}

    def step(self, action):
        """Takes an action (assigns a job) and returns the new state and reward."""
        # 1. Execute the action: Assign the current job to the chosen machine
        job_time = self.job_times[self.current_job]
        self.machine_times[action] += job_time
        
        # 2. Move to the next job
        self.current_job += 1
        
        # 3. Check if the episode is over
        terminated = bool(self.current_job >= self.num_jobs)
        truncated = False # Not used for this simple episodic task
        
        # 4. Calculate the Reward
        reward = 0
        if terminated:
            # The makespan is the time when the last machine finishes
            makespan = max(self.machine_times)
            
            # We use a negative reward because Q-Learning maximizes the score.
            # Minimizing the makespan means maximizing a negative makespan.
            reward = -makespan
            
        info = {"machine_times": self.machine_times}
        
        return self.current_job, reward, terminated, truncated, info
    
if __name__ == "__main__":
    env = JobSchedulingEnv()
    state, _ = env.reset()
    
    print("Starting random assignment test...")
    terminated = False
    
    while not terminated:
        # Pick a random machine (0 or 1)
        action = env.action_space.sample() 
        
        print(f"Assigning Job {state} (Time: {env.job_times[state]}s) to Machine {action}")
        
        # Take the step
        state, reward, terminated, truncated, info = env.step(action)
        
    print(f"\nFinal State Reached!")
    print(f"Machine Workloads: {info['machine_times']}")
    print(f"Final Reward (Negative Makespan): {reward}")