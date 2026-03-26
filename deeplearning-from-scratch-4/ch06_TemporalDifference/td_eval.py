import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict

import numpy as np

from common.gridworld import GridWorld


class TdAgent:
    """时序差分算法的评估策略"""
    def __init__(self, gamma=0.9, alpha=0.01, action_size=4):
        self.gamma = gamma
        self.alpha = alpha
        self.action_size = action_size

        random_action = {}
        for i in range(self.action_size):
            random_action[i] = 1 / self.action_size
        self.pi = defaultdict(lambda: random_action)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state): 
        """按概率分布对action进行采样"""
        action_probs = self.pi[state] 
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def eval(self, state, reward, next_state, done):
        """执行时序差分方法进行策略评估"""
        next_V = 0 if done else self.V[next_state]
        td_target = reward + self.gamma * next_V

        td_error = td_target - self.V[state]
        self.V[state] += td_error * self.alpha

if __name__ == "__main__":
    env = GridWorld()
    agent = TdAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.eval(state, reward, next_state, done)
            if done:
                break
            state = next_state

    env.render_v(agent.V)