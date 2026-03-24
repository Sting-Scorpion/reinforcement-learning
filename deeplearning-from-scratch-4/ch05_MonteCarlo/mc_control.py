import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.gridworld import GridWorld

from collections import defaultdict
import numpy as np

def greedy_probs(Q, state, action_size=4):
    """贪婪策略，100%利用"""
    qs = [Q[(state, action)] for action in range(action_size)] # 获得当前状态下所有动作的动作价值函数
    max_action = np.argmax(qs) # 动作价值最大的动作

    action_probs = {action: 0.0 for action in range(action_size)} # 动作选择的概率，先全置0
    action_probs[max_action] = 1.0 # 将动作价值最大的动作的概率置1
    return action_probs

def epsilon_greedy_probs(Q, state, epsilon=0, action_size=4):
    """epsilon-贪婪策略，兼顾利用与探索"""
    qs = [Q[(state, action)] for action in range(action_size)] # 获得当前状态下所有动作的动作价值函数
    max_action = np.argmax(qs) # 动作价值最大的动作

    base_prob = epsilon / action_size # 有epsilon的概率不选动作价值最大的
    action_probs = {action: base_prob for action in range(action_size)} # 动作选择的概率，先将概率均分到每个动作
    action_probs[max_action] += 1 - epsilon # 剩下的概率作为动作价值最大的动作
    return action_probs

class McAgent:
    def __init__(self, gamma=0.9, action_size=4):
        """按照蒙特卡洛方法进行贪心策略迭代的agent"""
        self.gamma = gamma
        self.action_size = action_size # 行动个数

        random_action = {}
        for i in range(self.action_size):
            random_action[i] = 1 / self.action_size
        self.pi = defaultdict(lambda: random_action) # 策略
        self.Q = defaultdict(lambda: 0) # 动作价值函数
        self.cnts = defaultdict(lambda: 0) # 访问次数
        self.memory = [] # 采样轨迹

    def get_action(self, state): 
        """按概率分布对action进行采样"""
        action_probs = self.pi[state] 
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        """在轨迹中添加 (状态, 行动, 奖励) 的三元组"""
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = reward + self.gamma * G
            key = (state, action) # 动作价值函数需要(状态, 动作)
            self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) / self.cnts[key] # 增量式更新样本均值，每个样本权重相同

            self.pi[state] = greedy_probs(self.Q, state)

class ModifiedMcAgent:
    def __init__(self, gamma=0.9, action_size=4, alpha=0.1, epsilon=0.1):
        """按照蒙特卡洛方法进行ε策略迭代的agent"""
        self.gamma = gamma
        self.action_size = action_size # 行动个数
        self.alpha = alpha # 更新Q值时的固定权重
        self.epsilon = epsilon # epsilon-greedy的epsilon值

        random_action = {}
        for i in range(self.action_size):
            random_action[i] = 1 / self.action_size
        self.pi = defaultdict(lambda: random_action) # 策略
        self.Q = defaultdict(lambda: 0) # 动作价值函数
        self.memory = [] # 采样轨迹

    def get_action(self, state): 
        """按概率分布对action进行采样"""
        action_probs = self.pi[state] 
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        """在轨迹中添加 (状态, 行动, 奖励) 的三元组"""
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = reward + self.gamma * G
            key = (state, action) # 动作价值函数需要(状态, 动作)
            self.Q[key] += (G - self.Q[key]) * self.alpha # 更新样本指数移动平均，越新的数据权重越大

            self.pi[state] = epsilon_greedy_probs(self.Q, state, epsilon=self.epsilon)

if __name__ == "__main__":
    np.random.seed(42) # 使结果可复现
    env = GridWorld()
    # agent = McAgent(0.9, 4)
    agent = ModifiedMcAgent(0.9, 4, 0.1, 0.1)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update()
                break

            state = next_state

    env.render_q(agent.Q)