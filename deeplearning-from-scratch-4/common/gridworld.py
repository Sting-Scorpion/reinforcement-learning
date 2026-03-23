import numpy as np
import common.gridworld_render as render_helper

from collections import defaultdict

class GridWorld:
    """网格世界环境类，初始化为 3x4 的网格世界"""
    def __init__(self):
        # 上下左右动作
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        # 初始化地图，左上角为(0, 0) （与数组下标一致）
        self.reward_map = np.array(
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state = (0, 3) # 终止位置
        self.wall_state = (1, 1) # 不可进入区域
        self.start_state = (2, 0) # 起始位置
        self.agent_state = self.start_state # agent的初始位置设置为起点

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        """根据s_t和a_t获得s_{t+1}"""
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 判断下一时刻的状态是非法位置（超出边界或在不可进入区域）
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        """计算即时奖励"""
        # 为了和奖励值 = r(s_t, a_t, s_{t+1})对应，入参设计成这样。
        # 但是这是确定性奖励，只会用到下一时刻的状态
        return self.reward_map[next_state]

    def reset(self):
        """重置环境"""
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        """让agent以action进行移动"""
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)

if __name__ == "__main__":
    env = GridWorld() 
    V = {}
    # 字典元素的初始化
    # 确定性策略
    # for state in env.states():
    #     V[state] = 0
    V = defaultdict(lambda: 0)
    
    state = (1, 2)
    print(V[state]) # 输出状态 (1,2) 的价值函数

    # 随机性策略
    V = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    
    state = (1, 2)
    print(V[state]) # 输出状态 (1,2) 的价值函数