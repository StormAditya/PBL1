# Simulation-Based Agentic AI for Dynamic Job Scheduling

## 🚀 Overview
This project implements an **AI-driven job scheduling system** using **Reinforcement Learning (RL)** to optimize resource allocation in dynamic cloud environments.

Traditional scheduling algorithms like **FCFS** and **SJF** rely on static rules and fail to adapt to real-time workloads.  
Our approach introduces an **Agentic AI scheduler** using **Double Deep Q Networks (DDQN)** that learns optimal scheduling policies through interaction with the environment.

---

## 🎯 Key Objectives
- Design a **dynamic job scheduling system**
- Compare traditional vs AI-based scheduling
- Implement and evaluate:
  - Q-Learning
  - DQN
  - DDQN
- Optimize **makespan (total completion time)**
- Improve **resource utilization (CPU/GPU)**

---

## 🧠 Core Idea
The scheduler is modeled as an **intelligent agent** that:
1. Observes the system state (job pool + resources)
2. Selects the best job dynamically
3. Receives reward based on execution efficiency
4. Learns optimal decisions over time

---

## ⚙️ System Architecture
Job Generator → Job Pool → RL Agent → Resource Allocation → Execution → Reward → Learning

- **State:** Job pool (CPU/GPU), machine availability  
- **Action:** Select a job to execute  
- **Reward:** Based on execution efficiency (fast vs slow mapping)  

---

## 🤖 Algorithms Implemented

### 1. Q-Learning ❌
- Tabular approach  
- Failed due to large state space  

### 2. Deep Q Network (DQN) ⚠️
- Uses neural network  
- Improved performance but unstable  

### 3. Double DQN (DDQN) ✅
- Reduces overestimation bias  
- Stable learning  
- Best performance  

---

## 🧪 Experimental Setup

- **Episodes:** 3500  
- **Jobs per Episode:** 100  

### Job Types:
- Type 0 → CPU jobs  
- Type 1 → GPU jobs  

### Machines:
- Fast CPU VM  
- Slow CPU VM  
- GPU VM  

### Action Space:
- Select 1 job from a pool of 5  

---

## 🎯 Reward Function

- ✅ Correct job-machine mapping → Faster execution (positive reward)  
- ❌ Incorrect mapping → Slower execution (penalty)  

> Reward is **time-based**, directly optimizing system performance.

---

## 📊 Results (example)

| Algorithm             | Makespan (Lower is Better) |
|-----------------------|----------------------------|
| Random                | 1602.3 s                   |
| FCFS                  | 929.8 s                    | 
| SJF                   | 928.1 s                    |
| **DDQN (Agentic AI)** | **399.6 s**                |

### 🔥 Key Insights:
- DDQN achieved **>2× improvement** over traditional algorithms  
- Learns optimal **job-resource mapping**  
- Reduces inefficient scheduling decisions  

---

## 📈 Learning Behavior
- Initial phase: random decisions  
- Gradual improvement through reward feedback  
- Converges to optimal scheduling policy  

---

## 💻 Tech Stack

- **Language:** Python  
- **Libraries:**
  - PyTorch  
  - NumPy  
  - Matplotlib  
- **Frontend (Visualization):** Streamlit  

---

## 🖥️ Demo
Interactive simulation using Streamlit:
- Compare scheduling algorithms  
- Visualize makespan  
- Observe agent learning  

---

## 🌍 Real-World Impact
- Improves **cloud resource efficiency**
- Reduces **energy consumption**
- Lowers **infrastructure cost**
- Enables **adaptive scheduling systems**

---

## 🔗 SDG Alignment
- SDG 9 – Industry, Innovation & Infrastructure  
- SDG 7 – Affordable & Clean Energy  
- SDG 12 – Responsible Consumption  

---

## ⚠️ Limitations
- Simulation-based (not deployed on real cloud)
- Limited job pool size
- Training time required for convergence  

---

## 🔮 Future Work
- Multi-agent scheduling systems  
- Real cloud deployment (AWS/GCP)  
- Advanced RL algorithms (PPO, Actor-Critic)  
- Scalable distributed scheduling  

---

## 👨‍💻 Contributors
- Aditya Desai  
- Anoushka Mathew  
- Anshuman Gahlot
- Akanksha Nandy    

---

## 📚 References

1. Y. Sanjalawe, S. Al-E’mari, S. Fraihat, and S. Makhadmeh,  
   “AI-driven job scheduling in cloud computing: a comprehensive review,”  
   *Artificial Intelligence Review*, vol. 58, no. 7, art. no. 197, Apr. 2025.  
   doi: 10.1007/s10462-025-11208-8  

2. G. Zhou, W. Tian, R. Buyya, R. Xue, and L. Song,  
   “Deep reinforcement learning-based methods for resource scheduling in cloud computing: a review and future directions,”  
   *Artificial Intelligence Review*, vol. 57, pp. 124–165, Apr. 2024.  
   doi: 10.1007/s10462-024-10756-9  

3. Alzoubi, Yehia; Mishra, Alok; Topcu, Ahmet,  
   “Research trends in deep learning and machine learning for cloud computing security,”  
   *Artificial Intelligence Review*, vol. 57, 2024.  
   doi: 10.1007/s10462-024-10776-5  

4. Li, Pochun; Xiao, Yuyang; Yan, Jinghua; Li, Xuan; Wang, Xiaoye,  
   “Reinforcement Learning for Adaptive Resource Scheduling in Complex System Environments,”  
   arXiv:2411.05346, 2024.  

5. Y. Yang, F. Ren, and M. Zhang,  
   “A BDI Agent-Based Task Scheduling Framework for Cloud Computing,”  
   arXiv preprint arXiv:2401.02223, Jan. 2024.  

6. Radhika, S.; Swain, S. K.; Adinarayana, S.; Babu, B. R.,  
   “Efficient task scheduling in cloud using double deep Q Network,”  
   *International Journal of Computing and Digital Systems*, vol. 16, no. 1, pp. 1–11, 2024.  

7. Gu, Y.; Liu, Z.; Dai, S.; Liu, C.; Wang, Y.; Wang, S.; Cheng, L.,  
   “Deep reinforcement learning for job scheduling and resource management in cloud computing: An algorithm-level review,”  
   arXiv:2501.01007, 2025.  

8. Zhang, Y.; Liu, B.; Gong, Y.; Huang, J.; Xu, J.; Wan, W.,  
   “Application of machine learning optimization in cloud computing resource scheduling and management,”  
   In *Proceedings of the 5th International Conference on Computer Information and Big Data Applications*, pp. 171–175, Apr. 2024.  

9. Sun, X.; Duan, Y.; Deng, Y.; Guo, F.; Cai, G.; Peng, Y.,  
   “Dynamic operating system scheduling using double DQN: A reinforcement learning approach to task optimization,”  
   In *2025 8th International Conference on Advanced Algorithms and Control Engineering (ICAACE)*, pp. 1492–1497, IEEE, Mar. 2025.  

10. Shen, W.; Lin, W.; Wu, W.; Wu, H.; Li, K.,  
    “Reinforcement learning-based task scheduling for heterogeneous computing in end-edge-cloud environment,”  
    *Cluster Computing*, vol. 28, no. 3, 2025.
    
---

## ⭐ Final Note
This project demonstrates how **Agentic AI + Reinforcement Learning** can outperform traditional scheduling by learning from data and adapting to dynamic environments.
