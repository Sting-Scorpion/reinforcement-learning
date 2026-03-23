import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict

import numpy as np

from common.gridworld import GridWorld

class RandomAgent:
    """按照随机策略采取行动的agent"""
    def __init__(self, gamma=0.9, action_size=4):
        self.gamma = gamma
        self.action_size = action_size # 行动个数

        random_action = {}
        for i in range(self.action_size):
            random_action[i] = 1 / self.action_size
        self.pi = defaultdict(lambda: random_action) # 策略
        self.V = defaultdict(lambda: 0) # 价值函数
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
    
    def eval(self):
        """执行蒙特卡洛方法进行策略评估"""
        G = 0 # 初始化收益，即在最终状态下长期收益为0（已经是最后一步）
        for data in reversed(self.memory): # 反向遍历
            state, action, reward = data
            G = self.gamma * G + reward # G_t = r_t + \gamma * G_{t+1}
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state] # 增量式计算价值函数采样的均值

if __name__ == "__main__":
    np.random.seed(42) # 使结果可复现
    env = GridWorld()
    agent = RandomAgent(0.9, 4)

    eposides = 1000
    for eposide in range(eposides):
        # 重置初始状态
        state = env.reset()
        agent.reset()

        # 在到达终止位置前一直行动
        while True:
            action = agent.get_action(state) # 按照（随机）策略采样动作
            next_state, reward, done = env.step(action) # 在环境中执行动作

            agent.add(state, action, reward) # 记录采样轨迹
            if done:
                agent.eval()
                break

            state = next_state # 更新状态

    env.render_v(agent.V) # 画图