import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict, deque

import numpy as np

from common.gridworld import GridWorld
from common.utils import greedy_probs

class SarsaAgent:
    def __init__(self, gamma=0.9, alpha=0.8, epsilon=0.1, action_size=4):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_size = action_size

        random_action = {}
        for i in range(self.action_size):
            random_action[i] = 1 / self.action_size
        self.pi = defaultdict(lambda: random_action)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2) # 使用deque，保留两个最近的经验数据

    def get_action(self, state): 
        """按概率分布对action进行采样"""
        action_probs = self.pi[state] # 同策略型SARSA
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        """同策略SARSA的策略迭代"""
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        # 下一个Q函数
        next_q = 0 if done else self.Q[next_state, next_action]
        # TD方法更新
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += td_error * self.alpha
        # 策略迭代
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

if __name__ == "__main__":
    np.random.seed(42) # 使结果可复现
    env = GridWorld()
    agent = SarsaAgent(gamma=0.9, alpha=0.01, epsilon=0.1, action_size=4)

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, done) # 每次都要调用

            if done:
                # 到达目标时也要调用
                agent.update(next_state, None, None, None) 
                break
            state = next_state

    env.render_q(agent.Q)