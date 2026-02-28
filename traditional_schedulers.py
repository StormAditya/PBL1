import numpy as np

def round_robin_scheduling(jobs, num_machines):
    """Assigns jobs strictly sequentially: M0, M1, M2..."""
    machine_times = [0] * num_machines
    for i, job in enumerate(jobs):
        machine_idx = i % num_machines
        machine_times[machine_idx] += job
    return machine_times, max(machine_times)

def fcfs_scheduling(jobs, num_machines):
    """Assigns the next job in the queue to the currently lightest machine."""
    machine_times = [0] * num_machines
    for job in jobs:
        # Find the machine with the current minimum load
        lightest_machine = np.argmin(machine_times)
        machine_times[lightest_machine] += job
    return machine_times, max(machine_times)

def lpt_scheduling(jobs, num_machines):
    """Sorts jobs from largest to smallest, then assigns to the lightest machine."""
    machine_times = [0] * num_machines
    # Sort jobs in descending order
    sorted_jobs = sorted(jobs, reverse=True)
    
    for job in sorted_jobs:
        lightest_machine = np.argmin(machine_times)
        machine_times[lightest_machine] += job
    return machine_times, max(machine_times)

if __name__ == "__main__":
    # Setup a rigorous test: 10 machines, 100 heavily randomized jobs
    NUM_MACHINES = 10
    NUM_JOBS = 100
    
    # Generate the exact same random job queue for all algorithms to ensure a fair test
    test_jobs = [np.random.randint(1, 51) for _ in range(NUM_JOBS)]
    
    print(f"--- Scheduling Test: {NUM_JOBS} Jobs on {NUM_MACHINES} Machines ---")
    print(f"Total Workload to distribute: {sum(test_jobs)} seconds")
    print(f"Theoretical Absolute Best Makespan: {sum(test_jobs) / NUM_MACHINES:.1f} seconds\n")

    # 1. Round Robin
    rr_times, rr_makespan = round_robin_scheduling(test_jobs, NUM_MACHINES)
    print("1. Round Robin")
    print(f"Makespan: {rr_makespan}")
    print(f"Workloads: {rr_times}\n")

    # 2. First Come, First Served (FCFS)
    fcfs_times, fcfs_makespan = fcfs_scheduling(test_jobs, NUM_MACHINES)
    print("2. First Come, First Served (FCFS)")
    print(f"Makespan: {fcfs_makespan}")
    print(f"Workloads: {fcfs_times}\n")

    # 3. Longest Processing Time (LPT)
    lpt_times, lpt_makespan = lpt_scheduling(test_jobs, NUM_MACHINES)
    print("3. Longest Processing Time First (LPT)")
    print(f"Makespan: {lpt_makespan}")
    print(f"Workloads: {lpt_times}\n")