from task_allocation import *
import numpy as np
import pygame
from stable_baselines3 import PPO, SAC, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import spaces, Env
from collections import Counter, deque
import os
import matplotlib.pyplot as plt
import time
import random
import argparse
import json
from datetime import datetime
import sys


# 设置确定性种子
def set_deterministic(seed=42):
    """设置所有随机源为确定性模式"""
    # 设置NumPy随机种子
    np.random.seed(seed)
    # 设置Python随机种子
    random.seed(seed)
    # 设置PyGame随机种子（如果有的话）
    # 设置环境变量以确保确定性（对于某些CUDA操作）
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # 对于PyTorch（如果使用GPU）
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，用于处理 numpy 数据类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class UAVCombatGymEnv(Env):
    """将自定义环境包装成Gymnasium接口格式，包含历史序列信息"""

    def __init__(self, render_mode=False, history_length=1, algorithm_name="PPO", seed=None, input_data=None):
        super(UAVCombatGymEnv, self).__init__()
        self.render_mode = render_mode
        self.history_length = history_length  # 保留的历史状态数量
        self.seed = seed
        self.input_data = input_data  # 存储输入JSON数据
        self.algorithm_name = algorithm_name
        self.steps = None
        self.count_step = 0

        # 解析输入JSON数据
        if input_data:
            # 提取红色方无人机配置
            red_uavs = Counter(input_data.get('red_uavs', {
                'interceptor': 40,
                'recon': 10,
                'escort': 0
            }))

            # 提取红色方机场位置
            red_airport = input_data.get('red_airport', {
                'x': 807.1191291159989,
                'y': 540.0075557531011
            })
            self.steps = input_data.get('steps')
            ground_attack_num, recon_num, escort_num = 0, 0, 0
            for step in range(len(self.steps)):
                ground_attack_num += self.steps[step]['blue_uavs']['ground_attack']
                recon_num += self.steps[step]['blue_uavs']['recon']
                escort_num += self.steps[step]['blue_uavs']['escort']


            # 创建自定义环境
            self.custom_env = UAVCombatEnv(
                initial_red_uav_counts=red_uavs,
                initial_blue_uav_counts=Counter({'ground_attack': ground_attack_num, 'recon': recon_num, 'escort': escort_num}),
                red_base_rect=pygame.Rect(50, 400, 200, 300),
                blue_base_rect=pygame.Rect(950, 400, 200, 300),
                render_mode=render_mode
            )
        else:
            # 使用默认配置
            self.custom_env = UAVCombatEnv(
                initial_red_uav_counts=Counter({'interceptor': 40, 'recon': 10, 'escort': 0}),
                initial_blue_uav_counts=Counter({'ground_attack': 10, 'recon': 10, 'escort': 30}),
                red_base_rect=pygame.Rect(50, 400, 200, 300),
                blue_base_rect=pygame.Rect(950, 400, 200, 300),
                render_mode=render_mode
            )

        # 定义观察空间 - 包含当前状态和历史状态
        # 每个时间步的状态: 红方剩余(3) + 蓝方派出(3)
        state_dim = 3 + 3
        self.observation_space = spaces.Box(
            low=0, high=50, shape=(history_length, state_dim), dtype=np.float32
        )

        # 设置算法类型
        self.set_algorithm_type(algorithm_name)

        # 初始化随机数生成器
        self.rng = np.random.RandomState(seed)

        # 初始化输出数据结构
        self.output_data = {
            "episode_info": {
                "start_time": datetime.now().isoformat(),
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT
            },
            "steps": []
        }
        self.uav_counter = 0  # 全局无人机编号计数器
        self.uav_ids = {
            'interceptor': [],
            'recon': [],
            'escort': []
        }

    def set_algorithm_type(self, algorithm_name):
        """根据算法类型设置动作空间"""
        self.algorithm_type = algorithm_name

        if algorithm_name == "DQN":
            # DQN需要离散动作空间
            # 计算总动作数: 拦截机(41) * 侦察机(11) * 护卫机(1) = 451
            self.action_space = spaces.Discrete(41 * 11 * 1)
        elif algorithm_name == "SAC":
            # SAC需要连续动作空间
            self.action_space = spaces.Box(
                low=np.array([0, 0, 0]),
                high=np.array([40, 10, 0]),
                dtype=np.float32
            )
        else:
            # 其他算法使用MultiDiscrete
            self.action_space = spaces.MultiDiscrete([41, 11, 1])

    def reset(self, seed=None, options=None):
        # 处理seed和options参数以符合Gymnasium接口
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)

        # 重置自定义环境
        state = self.custom_env.reset()

        # 重置输出数据结构
        self.output_data = {
            "episode_info": {
                "start_time": datetime.now().isoformat(),
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT
            },
            "steps": []
        }

        # 初始化无人机编号
        self.initialize_uav_ids()

        # 重置历史状态
        self.state_history = deque(maxlen=self.history_length)

        # 创建初始观察
        enemy_state = {"current_enemy_formation_remaining": self.custom_env.blue_uavs}
        enemy_formation = self.generate_enemy_formation(enemy_state)
        self.count_step = 0
        initial_obs = self._create_observation(state, enemy_formation)

        # 填充历史缓冲区
        for _ in range(self.history_length):
            self.state_history.append(initial_obs)

        info = {}
        return self._get_current_observation(), info

    def step(self, action):
        # 根据算法类型转换动作
        if self.algorithm_type == "DQN":
            # 将离散动作转换为MultiDiscrete
            action = self._discrete_to_multidiscrete(action)
        elif self.algorithm_type == "SAC":
            # 将连续动作转换为整数
            action = self._continuous_to_discrete(action)

        if self.custom_env.red_uavs['recon'] > 1:
            action[1] = max(action[1], 1)

        # 将离散动作转换为字典格式
        action_dict = {
            'interceptor': min(action[0], self.custom_env.red_uavs['interceptor']),
            'recon': min(action[1], self.custom_env.red_uavs['recon']),
            'escort': 0
        }

        # 获取分配的无人机ID
        assigned_ids = self.get_assigned_uav_ids(action_dict)

        # 使用新逻辑生成敌方配置
        enemy_state = {"current_enemy_formation_remaining": self.custom_env.blue_uavs}
        enemy_formation = self.generate_enemy_formation(enemy_state, step=self.count_step)
        self.count_step += 1
        battlefield_coords = self.generate_battlefield_coords()

        # 执行环境步进
        next_state, reward, done, info = self.custom_env.step(
            action=action_dict,
            enemy_formation=enemy_formation,
            battlefield_coords=battlefield_coords
        )
        if self.count_step >= len(self.steps):
            done = True

        # 记录步骤信息到JSON
        step_info = {
            "step": len(self.output_data["steps"]),
            "battlefield_coords": {
                "x": float(battlefield_coords[0]),
                "y": float(battlefield_coords[1])
            },
            "action": {
                uav_type: {
                    "count": count,
                    "uav_ids": assigned_ids[uav_type]
                } for uav_type, count in action_dict.items()
            },
            "reward": float(reward),
            "cumulative_reward": float(info.get('cumulative_reward', 0)),
            "friendly_remaining": dict(self.custom_env.red_uavs),
            "enemy_remaining": dict(self.custom_env.blue_uavs)
        }

        self.output_data["steps"].append(step_info)

        # 创建新观察
        new_obs = self._create_observation(next_state, enemy_formation)

        # 添加到历史
        self.state_history.append(new_obs)

        # 转换为Gymnasium格式的返回值
        terminated = done
        truncated = False  # 没有时间限制，所以总是False

        # 更新累计奖励信息
        info['cumulative_reward'] = info.get('cumulative_reward', 0) + reward

        return self._get_current_observation(), reward, terminated, truncated, info

    def generate_battlefield_coords(self):
        """
        生成确定性的战场坐标

        Returns:
            tuple: 战场坐标 (x, y)
        """
        # 使用固定的随机数生成器确保确定性
        return (
            self.rng.uniform(SCREEN_WIDTH * 0.3, SCREEN_WIDTH * 0.7),
            self.rng.uniform(SCREEN_HEIGHT * 0.3, SCREEN_HEIGHT * 0.7)
        )

    def _discrete_to_multidiscrete(self, action):
        """将离散动作转换为MultiDiscrete格式"""
        # 计算动作索引
        interceptor = action // (11 * 1)
        recon = (action % (11 * 1)) // 1
        escort = action % 1

        return [interceptor, recon, escort]

    def _continuous_to_discrete(self, action):
        """将连续动作转换为离散动作"""
        # 将连续动作转换为整数
        if np.isnan(action).any():
            action = np.nan_to_num(action, nan=0.0)
        interceptor = int(np.clip(action[0], 0, 40))
        recon = int(np.clip(action[1], 0, 10))
        escort = int(np.clip(action[2], 0, 0))  # 护卫机固定为0

        return [interceptor, recon, escort]

    def _create_observation(self, state, enemy_formation=None):
        """
        将环境状态转换为observation向量
        """
        friendly = state['friendly_remaining']

        # 创建observation向量: [我方拦截机, 我方侦察机, 我方护航机, 敌方对地攻击机, 敌方侦察机, 敌方护航机]
        obs = np.array([
            friendly['interceptor'],
            friendly['recon'],
            friendly['escort'],
            enemy_formation['ground_attack'] if enemy_formation else 0,
            enemy_formation['recon'] if enemy_formation else 0,
            enemy_formation['escort'] if enemy_formation else 0
        ], dtype=np.float32)

        return obs

    def _get_current_observation(self):
        """获取当前观察 - 包含历史序列"""
        # 将历史状态堆叠为矩阵
        return np.array(self.state_history, dtype=np.float32)

    def generate_enemy_formation(self, state, step=0):
        """生成确定性的敌方编队"""
        enemy_remaining = state.get('current_enemy_formation_remaining', {})

        ground_attack_remaining = enemy_remaining.get('ground_attack', 10)
        recon_remaining = enemy_remaining.get('recon', 10)
        escort_remaining = enemy_remaining.get('escort', 30)

        # 使用固定的随机数生成器确保确定性
        # ground_attack = min(self.rng.randint(1, max(2, ground_attack_remaining + 1)), ground_attack_remaining)
        # recon = min(self.rng.randint(1, max(2, recon_remaining + 1)), recon_remaining)
        # escort = min(self.rng.randint(0, max(1, escort_remaining + 1)), escort_remaining)
        ground_attack = self.steps[step]['blue_uavs']['ground_attack']
        recon = self.steps[step]['blue_uavs']['recon']
        escort = self.steps[step]['blue_uavs']['escort']

        # ground_attack = max(ground_attack, 1) if ground_attack_remaining > 0 else 0
        # recon = max(recon, 1) if recon_remaining > 0 else 0

        return {
            'ground_attack': ground_attack,
            'recon': recon,
            'escort': escort
        }

    def initialize_uav_ids(self):
        """初始化无人机编号"""
        self.uav_counter = 0
        self.uav_ids = {
            'interceptor': list(range(self.uav_counter, self.uav_counter + self.custom_env.red_uavs['interceptor'])),
            'recon': list(range(self.uav_counter + self.custom_env.red_uavs['interceptor'],
                                self.uav_counter + self.custom_env.red_uavs['interceptor'] + self.custom_env.red_uavs[
                                    'recon'])),
            'escort': list(
                range(self.uav_counter + self.custom_env.red_uavs['interceptor'] + self.custom_env.red_uavs['recon'],
                      self.uav_counter + self.custom_env.red_uavs['interceptor'] + self.custom_env.red_uavs['recon'] +
                      self.custom_env.red_uavs['escort']))
        }
        self.uav_counter += sum(self.custom_env.red_uavs.values())

    def get_assigned_uav_ids(self, action_dict):
        """获取分配给每个机型的无人机ID"""
        assigned_ids = {}

        for uav_type, count in action_dict.items():
            if count > 0 and self.uav_ids[uav_type]:
                # 取前count个ID
                assigned_ids[uav_type] = self.uav_ids[uav_type][:count]
                # 从可用列表中移除这些ID
                self.uav_ids[uav_type] = self.uav_ids[uav_type][count:]
            else:
                assigned_ids[uav_type] = []

        return assigned_ids

    def render(self):
        if self.render_mode:
            self.custom_env.render()

    def close(self):
        self.custom_env.close()

    def save_output_json(self):
        """保存JSON输出到文件"""
        # 完成episode后添加总结信息
        self.output_data["episode_info"]["end_time"] = datetime.now().isoformat()
        self.output_data["episode_info"]["total_steps"] = len(self.output_data["steps"])
        self.output_data["episode_info"]["final_friendly_remaining"] = dict(self.custom_env.red_uavs)
        self.output_data["episode_info"]["final_enemy_remaining"] = dict(self.custom_env.blue_uavs)

        # 使用默认函数处理 numpy 类型
        def default_converter(o):
            if isinstance(o, (np.integer, np.int64)):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        filename = f"{self.algorithm_name}_episode_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(f"outputs_json/{self.algorithm_name}", exist_ok=True)
        with open(f"outputs_json/{self.algorithm_name}/{filename}", 'w', encoding='utf-8') as f:
            json.dump(self.output_data, f, ensure_ascii=False, indent=2, default=default_converter)
        print(f"Output saved to {filename}")

def run_inference(algorithm_name, model_path, num_episodes=5, render=True, seed=42, input_json=None):
    """
    加载训练好的模型并进行确定性推理
    """
    # 设置确定性模式
    set_deterministic(seed)

    # 解析输入JSON（如果有）
    input_data = None
    if input_json:
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"Loaded input configuration from {input_json}")
        except Exception as e:
            print(f"Error loading input JSON: {e}")
            input_data = None

    # 创建推理环境（传入种子和输入数据）
    env = UAVCombatGymEnv(
        render_mode=render,
        history_length=3,
        algorithm_name=algorithm_name,
        seed=seed,
        input_data=input_data
    )

    # 根据算法类型加载模型
    if algorithm_name == "PPO":
        model = PPO.load(model_path)
    elif algorithm_name == "SAC":
        model = SAC.load(model_path)
    elif algorithm_name == "A2C":
        model = A2C.load(model_path)
    elif algorithm_name == "DQN":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"不支持的算法类型: {algorithm_name}")

    # 设置模型为评估模式（对于PyTorch模型）
    try:
        model.policy.set_training_mode(False)
    except:
        pass

    print(f"开始使用 {algorithm_name} 算法进行确定性推理 (种子={seed})...")

    # 运行多个episode
    start_time = time.time()
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)  # 为每个episode设置不同的种子
        done = False
        total_reward = 0
        step_count = 0

        print(f"\nEpisode {episode + 1}:")
        print("-" * 30)

        while not done:
            # 使用模型进行确定性预测
            action, _states = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            # 打印每一步的信息
            if step_count % 5 == 0:  # 每5步打印一次
                print(f"Step {step_count}: 奖励={reward:.2f}, 累计奖励={total_reward:.2f}")

        print(f"Episode {episode + 1} 完成: 总步数={step_count}, 总奖励={total_reward:.2f}")

        # 打印战斗结果摘要
        if hasattr(env.custom_env, 'red_uavs') and hasattr(env.custom_env, 'blue_uavs'):
            print("红方剩余:", dict(env.custom_env.red_uavs))
            print("蓝方剩余:", dict(env.custom_env.blue_uavs))

        # 保存输出JSON
        env.save_output_json()
    end_time = time.time()
    print(f"规划时间为:{end_time - start_time}")
    env.close()
    print(f"\n{algorithm_name} 算法确定性推理完成!")


class UAVCombatGymEnvTrain(Env):
    """将自定义环境包装成Gymnasium接口格式，包含历史序列信息"""

    def __init__(self, render_mode=False, history_length=1, algorithm_name="PPO"):
        super(UAVCombatGymEnvTrain, self).__init__()
        self.render_mode = render_mode
        self.history_length = history_length  # 保留的历史状态数量

        # 自定义环境配置
        self.custom_env = UAVCombatEnv(
            initial_red_uav_counts=Counter({'interceptor': 40, 'recon': 10, 'escort': 0}),
            initial_blue_uav_counts=Counter({'ground_attack': 10, 'recon': 10, 'escort': 30}),
            red_base_rect=pygame.Rect(50, 400, 200, 300),
            blue_base_rect=pygame.Rect(950, 400, 200, 300),
            render_mode=render_mode
        )

        # 定义观察空间 - 包含当前状态和历史状态
        # 每个时间步的状态: 红方剩余(3) + 蓝方派出(3)
        state_dim = 3 + 3
        self.observation_space = spaces.Box(
            low=0, high=50, shape=(history_length, state_dim), dtype=np.float32
        )

        # 设置算法类型
        self.set_algorithm_type(algorithm_name)

    def set_algorithm_type(self, algorithm_name):
        """根据算法类型设置动作空间"""
        self.algorithm_type = algorithm_name

        if algorithm_name == "DQN":
            # DQN需要离散动作空间
            # 计算总动作数: 拦截机(41) * 侦察机(11) * 护卫机(1) = 451
            self.action_space = spaces.Discrete(41 * 11 * 1)
        elif algorithm_name == "SAC":
            # SAC需要连续动作空间
            self.action_space = spaces.Box(
                low=np.array([0, 0, 0]),
                high=np.array([40, 10, 0]),
                dtype=np.float32
            )
        else:
            # 其他算法使用MultiDiscrete
            self.action_space = spaces.MultiDiscrete([41, 11, 1])

    def reset(self, seed=None, options=None):
        # 处理seed和options参数以符合Gymnasium接口
        if seed is not None:
            np.random.seed(seed)

        # 重置自定义环境
        state = self.custom_env.reset()

        # 重置历史状态
        self.state_history = deque(maxlen=self.history_length)


        # 创建初始观察
        enemy_state = {"current_enemy_formation_remaining": self.custom_env.blue_uavs}
        enemy_formation = self.generate_enemy_formation(enemy_state)
        initial_obs = self._create_observation(state, enemy_formation)

        # 填充历史缓冲区
        for _ in range(self.history_length):
            self.state_history.append(initial_obs)

        info = {}
        return self._get_current_observation(), info

    def step(self, action):
        # 根据算法类型转换动作
        if self.algorithm_type == "DQN":
            # 将离散动作转换为MultiDiscrete
            action = self._discrete_to_multidiscrete(action)
        elif self.algorithm_type == "SAC":
            # 将连续动作转换为整数
            action = self._continuous_to_discrete(action)
        else:
            # 保存上一次行动
            self.last_action = np.array(action, dtype=np.int32)

        # 将离散动作转换为字典格式
        action_dict = {
            'interceptor': action[0],
            'recon': action[1],
            'escort': action[2]
        }

        # 使用新逻辑生成敌方配置
        enemy_state = {"current_enemy_formation_remaining": self.custom_env.blue_uavs}
        enemy_formation = self.generate_enemy_formation(enemy_state)
        battlefield_coords = self.generate_battlefield_coords()

        # 执行环境步进
        next_state, reward, done, info = self.custom_env.step(
            action=action_dict,
            enemy_formation=enemy_formation,
            battlefield_coords=battlefield_coords
        )

        # 创建新观察
        new_obs = self._create_observation(next_state, enemy_formation)

        # 添加到历史
        self.state_history.append(new_obs)

        # 转换为Gymnasium格式的返回值
        terminated = done
        truncated = False  # 没有时间限制，所以总是False

        return self._get_current_observation(), reward, terminated, truncated, info

    def generate_battlefield_coords(self):
        """
        生成随机的战场坐标

        Returns:
            tuple: 战场坐标 (x, y)
        """
        # 随机战场坐标（在屏幕中央区域）
        return (
            np.random.uniform(SCREEN_WIDTH * 0.3, SCREEN_WIDTH * 0.7),
            np.random.uniform(SCREEN_HEIGHT * 0.3, SCREEN_HEIGHT * 0.7)
        )

    def _discrete_to_multidiscrete(self, action):
        """将离散动作转换为MultiDiscrete格式"""
        # 计算动作索引
        interceptor = action // (11 * 1)
        recon = (action % (11 * 1)) // 1
        escort = action % 1

        # 保存上一次行动
        self.last_action = np.array([interceptor, recon, escort], dtype=np.int32)

        return [interceptor, recon, escort]

    def _continuous_to_discrete(self, action):
        """将连续动作转换为离散动作"""
        # 将连续动作转换为整数
        if np.isnan(action).any():
            action = np.nan_to_num(action, nan=0.0)
        interceptor = int(np.clip(action[0], 0, 40))
        recon = int(np.clip(action[1], 0, 10))
        escort = int(np.clip(action[2], 0, 0))  # 护卫机固定为0

        return [interceptor, recon, escort]

    def _create_observation(self, state, enemy_formation=None):
        """
        将环境状态转换为observation向量

        Args:
            state: 环境状态字典
            enemy_formation: 敌方编队信息（可选）

        Returns:
            np.array: observation向量 [我方拦截机, 我方侦察机, 我方护航机, 敌方对地攻击机, 敌方侦察机, 敌方护航机]
        """
        friendly = state['friendly_remaining']

        # 创建observation向量: [我方拦截机, 我方侦察机, 我方护航机, 敌方对地攻击机, 敌方侦察机, 敌方护航机]
        obs = np.array([
            friendly['interceptor'],
            friendly['recon'],
            friendly['escort'],
            enemy_formation['ground_attack'] if enemy_formation else 0,
            enemy_formation['recon'] if enemy_formation else 0,
            enemy_formation['escort'] if enemy_formation else 0
        ], dtype=np.float32)

        return obs

    def _get_current_observation(self):
        """获取当前观察 - 包含历史序列"""
        # 将历史状态堆叠为矩阵
        return np.array(self.state_history, dtype=np.float32)

    def generate_enemy_formation(self, state):
        """生成敌方编队"""
        enemy_remaining = state.get('current_enemy_formation_remaining', {})

        ground_attack_remaining = enemy_remaining.get('ground_attack', 10)
        recon_remaining = enemy_remaining.get('recon', 10)
        escort_remaining = enemy_remaining.get('escort', 30)

        ground_attack = min(np.random.randint(1, max(2, ground_attack_remaining + 1)), ground_attack_remaining)
        recon = min(np.random.randint(1, max(2, recon_remaining + 1)), recon_remaining)
        escort = min(np.random.randint(0, max(1, escort_remaining + 1)), escort_remaining)

        ground_attack = max(ground_attack, 1) if ground_attack_remaining > 0 else 0
        recon = max(recon, 1) if recon_remaining > 0 else 0

        return {
            'ground_attack': ground_attack,
            'recon': recon,
            'escort': escort
        }

    def render(self):
        if self.render_mode:
            self.custom_env.render()

    def close(self):
        self.custom_env.close()


def train_and_evaluate_algorithm(algorithm_class, algorithm_name, total_timesteps=250000):
    """训练并评估单个算法"""
    print(f"\n{'=' * 50}")
    print(f"开始训练 {algorithm_name} 算法")
    print(f"{'=' * 50}")

    # 创建训练环境和评估环境
    def make_env():
        return UAVCombatGymEnvTrain(render_mode=False, history_length=3, algorithm_name=algorithm_name)

    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # 设置算法特定的日志目录
    log_dir = f"logs/{algorithm_name}/"
    os.makedirs(log_dir, exist_ok=True)
    best_model_path = os.path.join(log_dir, "best_model")

    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,  # 每5000步评估一次
        n_eval_episodes=10,  # 每次评估运行10个episode
        deterministic=True,
        render=False,
        verbose=1
    )

    # 创建模型
    model_params = {
        "policy": "MlpPolicy",
        "env": train_env,
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": "auto",
        "policy_kwargs": {
            "net_arch": [256, 256]  # 增加网络容量以处理历史信息
        }
    }

    # 添加算法特定的超参数
    if algorithm_name == "PPO":
        model_params.update({
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01
        })
    elif algorithm_name == "SAC":
        # SAC通常用于连续动作空间，但可以尝试用于离散动作
        model_params.update({
            "learning_rate": 0.0003,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": "auto"
        })
    elif algorithm_name == "A2C":
        model_params.update({
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01
        })
    elif algorithm_name == "DQN":
        # 对于DQN，使用离散动作空间
        model_params.update({
            "learning_rate": 0.0001,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 100,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05
        })

    # 创建模型
    model = algorithm_class(**model_params)

    # 训练模型
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name=f"{algorithm_name}_uav_combat"
    )
    training_time = time.time() - start_time

    # 保存最终模型
    model.save(os.path.join(log_dir, f"{algorithm_name}_uav_final"))


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer different online ppo algs.')
    # 添加所有可能的命令行参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default="test",
                        help='choose train or test')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--alg_name', type=str, default="PPO", help='infer alg names')
    parser.add_argument('--render', action='store_true', help='render params')
    parser.add_argument('--model_path', type=str, default="save_models/PPO_best_model.zip",
                        help="the directory of model")
    parser.add_argument('--input_json', type=str, default="test.json", help='Path to input JSON file with configuration')
    args = parser.parse_args()
    seed = args.seed

    # 运行推理
    if args.mode == "test":
        run_inference(
            args.alg_name,
            args.model_path,
            num_episodes=args.num_episodes,
            render=args.render,
            seed=seed,
            input_json=args.input_json
        )
    else:
        if args.alg_name == 'PPO':
            train_and_evaluate_algorithm(PPO, args.alg_name)
        elif args.alg_name == 'DQN':
            train_and_evaluate_algorithm(DQN, args.alg_name)
        elif args.alg_name == 'A2C':
            train_and_evaluate_algorithm(A2C, args.alg_name)