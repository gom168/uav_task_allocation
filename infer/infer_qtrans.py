import numpy as np
import torch
import pygame
import json
import argparse
from multi_agent import MultiBattlefieldEnv
from collections import deque
import time
from datetime import datetime
import os


class QNetwork(torch.nn.Module):
    """Q网络：输入状态，输出3个动作的Q值"""

    def __init__(self, state_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 41 * 11 * 1)  # 输出40*10个动作的Q值
        )

    def forward(self, state):
        return self.net(state)


class VNetwork(torch.nn.Module):
    """V网络：输入状态，输出状态值"""

    def __init__(self, state_dim, hidden_dim=128):
        super(VNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


class QTRANBase(torch.nn.Module):
    """QTRAN基础网络：将各个智能体的Q值转换为总Q值"""

    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=128):
        super(QTRANBase, self).__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim

        # 转换网络
        self.transformation_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + num_agents * action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs, states, actions):
        # agent_qs: [batch_size, num_agents, 1]
        # states: [batch_size, state_dim]
        # actions: [batch_size, num_agents, action_dim]

        batch_size = agent_qs.size(0)

        # 计算联合Q值
        joint_qs = agent_qs.sum(dim=1)  # [batch_size, 1]

        # 准备转换网络的输入
        actions_flat = actions.view(batch_size, -1)  # [batch_size, num_agents * action_dim]
        transformation_input = torch.cat([states, actions_flat],
                                         dim=1)  # [batch_size, state_dim + num_agents * action_dim]

        # 计算转换后的Q值
        transformed_q = self.transformation_net(transformation_input)  # [batch_size, 1]

        return joint_qs, transformed_q


class QTRANInference:
    """QTRAN推理类，用于加载训练好的模型并进行推理"""

    def __init__(self, model_path, num_battlefields=3, device='cpu'):
        self.device = device
        self.num_agents = num_battlefields
        self.state_dim = 7 * num_battlefields
        self.model_path = model_path

        # 创建网络
        self.q_networks = torch.nn.ModuleList([
            QNetwork(self.state_dim).to(device) for _ in range(self.num_agents)
        ])
        self.v_network = VNetwork(self.state_dim).to(device)
        self.qtran_net = QTRANBase(self.num_agents, self.state_dim, 1).to(device)

        # 加载模型
        self.load_model(model_path)

        # 设置为评估模式
        for network in self.q_networks:
            network.eval()
        self.v_network.eval()
        self.qtran_net.eval()

    def load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载Q网络参数
        for i in range(self.num_agents):
            self.q_networks[i].load_state_dict(checkpoint['q_networks'][i])

        # 加载V网络参数
        self.v_network.load_state_dict(checkpoint['v_network'])

        # 加载QTRAN网络参数
        self.qtran_net.load_state_dict(checkpoint['qtran_net'])

        print(f"Model loaded successfully from {model_path}")

    def preprocess_state(self, state):
        """将环境状态转换为神经网络输入"""
        state_vector = []
        for battlefield in state:
            # 战场ID (0, 1, 2)
            state_vector.append(battlefield['battlefield_id'])
            # 友方剩余兵力
            state_vector.append(battlefield['friendly_remaining']['interceptor'])
            state_vector.append(battlefield['friendly_remaining']['recon'])
            state_vector.append(battlefield['friendly_remaining']['escort'])
            # 敌方剩余兵力
            state_vector.append(battlefield['enemy_remaining']['ground_attack'])
            state_vector.append(battlefield['enemy_remaining']['recon'])
            state_vector.append(battlefield['enemy_remaining']['escort'])

        return np.array(state_vector, dtype=np.float32)

    def _get_available_actions(self, agent_state):
        """获取当前状态下可用的动作数量"""
        max_interceptor = int(agent_state[1])  # 可用拦截机数量
        max_recon = int(agent_state[2])  # 可用侦察机数量
        max_escort = int(agent_state[3])  # 可用护航机数量

        return max_interceptor, max_recon, max_escort

    def _q_values_to_action(self, q_values):
        """将Q值转换为离散动作"""
        action_mask = q_values.clone()
        # 选择最佳动作类型
        best_action = torch.argmax(action_mask).item()
        return best_action

    def get_actions(self, state):
        """根据状态选择动作（推理模式）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions = []

        with torch.no_grad():
            for i in range(self.num_agents):
                agent_state = state[i * 7:(i + 1) * 7]  # 每个智能体的状态

                # 使用网络预测动作
                q_values = self.q_networks[i](state_tensor)
                action = self._q_values_to_action(q_values.squeeze(0))
                actions.append(np.array(action))

        return actions

    def action_to_env_format(self, actions):
        """将神经网络输出的动作转换为环境所需的整数格式"""
        env_actions = []
        for best_action in actions:
            # 对预测的值进行四舍五入并确保非负
            interceptor = int(best_action // (11 * 1))
            recon = int((best_action % (11 * 1)) // 1)
            escort = int(best_action % 1)

            env_actions.append({
                'interceptor': interceptor,
                'recon': recon,
                'escort': escort
            })
        return env_actions


class QTRANInferenceRenderer:
    """QTRAN推理渲染器，支持JSON输入和输出"""

    def __init__(self, model_path, config_path, device='cpu'):
        self.device = device
        self.config_path = config_path
        self.red_uavs = None
        self.blue_uavs = None
        self.steps = None
        self.config = self.load_config(config_path)

        # 根据配置中的steps数量确定战场数量
        self.num_battlefields = 3
        self.agent = QTRANInference(model_path, self.num_battlefields, device)

        # 初始化无人机编号系统
        self.uav_counter = 0
        self.uav_ids = {
            'interceptor': [],
            'recon': [],
            'escort': []
        }

        # 初始化JSON输出结构
        self.output_data = {
            "episode_info": {
                "start_time": datetime.now().isoformat(),
                "num_battlefields": self.num_battlefields,
                "model_path": model_path,
                "config_path": config_path,
                "algorithm": "QTRAN"
            },
            "initial_conditions": self.config,
            "steps": []
        }

    def load_config(self, config_path):
        """加载JSON配置文件"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.red_uavs = config["red_uavs"]
        self.steps = config["steps"]
        ground_attack_num, recon_num, escort_num = 0, 0, 0
        for step in range(len(self.steps)):
            ground_attack_num += self.steps[step]['blue_uavs']['ground_attack']
            recon_num += self.steps[step]['blue_uavs']['recon']
            escort_num += self.steps[step]['blue_uavs']['escort']
        self.blue_uavs = {
            'ground_attack': ground_attack_num,
            'recon': recon_num,
            'escort': escort_num
        }
        return config

    def initialize_environment(self, env):
        """根据配置文件初始化环境状态"""
        # 设置红方初始兵力
        env.red_uavs = self.config['red_uavs']
        env.red_airport = self.config['red_airport']

    def initialize_uav_ids(self, env):
        """初始化无人机编号系统"""
        self.uav_counter = 0
        self.uav_ids = {
            'interceptor': list(range(self.uav_counter, self.uav_counter + env.red_uavs['interceptor'])),
            'recon': list(range(self.uav_counter + env.red_uavs['interceptor'],
                                self.uav_counter + env.red_uavs['interceptor'] + env.red_uavs['recon'])),
            'escort': list(range(self.uav_counter + env.red_uavs['interceptor'] + env.red_uavs['recon'],
                                 self.uav_counter + env.red_uavs['interceptor'] + env.red_uavs['recon'] + env.red_uavs[
                                     'escort']))
        }
        self.uav_counter += sum(env.red_uavs.values())

        # 记录初始无人机分配
        self.output_data["initial_uav_allocation"] = {
            uav_type: len(ids) for uav_type, ids in self.uav_ids.items()
        }

    def get_assigned_uav_ids(self, actions):
        """获取分配给每个战场和机型的无人机ID"""
        assigned_ids = []

        for battlefield_idx, action in enumerate(actions):
            battlefield_assignment = {}

            for uav_type, count in action.items():
                if count > 0 and self.uav_ids[uav_type]:
                    # 取前count个ID
                    battlefield_assignment[uav_type] = {
                        "count": count,
                        "uav_ids": self.uav_ids[uav_type][:count]
                    }
                    # 从可用列表中移除这些ID
                    self.uav_ids[uav_type] = self.uav_ids[uav_type][count:]
                else:
                    battlefield_assignment[uav_type] = {
                        "count": count,
                        "uav_ids": []
                    }

            assigned_ids.append(battlefield_assignment)

        return assigned_ids

    def save_output_json(self):
        """保存JSON输出到文件"""
        folder_path = 'outputs_json/QTRAN'
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{folder_path}/qtran_inference_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.output_data, f, ensure_ascii=False, indent=2)
        print(f"Output saved to {filename}")

    def run_inference(self, render=False, speed=1.0):
        """运行推理并生成JSON输出"""

        # 图像配置
        image_config = {
            'background_image': 'figure/背景.png',
            'red_images': {
                'interceptor': 'figure/红.png',
                'recon': 'figure/红2.png',
                'escort': 'figure/红3.png'
            },
            'blue_images': {
                'ground_attack': 'figure/蓝.png',
                'recon': 'figure/蓝2.png',
                'escort': 'figure/蓝3.png'
            }
        }

        # 创建环境
        env = MultiBattlefieldEnv(
            num_battlefields=self.num_battlefields,
            blue_uavs=self.blue_uavs,
            red_uavs=self.red_uavs,
            render_mode=render,
            **image_config
        )

        # 根据配置文件初始化环境
        self.initialize_environment(env)

        # 初始化无人机编号系统
        self.initialize_uav_ids(env)

        # 重置环境
        state = env.reset()
        processed_state = self.agent.preprocess_state(state)

        total_reward = 0
        step = 0

        # 处理每个战场步骤
        steps = self.config['steps']
        batch_size = self.num_battlefields
        for i in range(0, len(steps), batch_size):
            battlefield_all_coords = []
            all_blue_uavs = []
            batch = steps[i:i + batch_size]
            if len(batch) < batch_size:
                batch = batch + [None] * (batch_size - len(batch))
            for step_idx, step_info in enumerate(batch):
                if step_info is not None:
                    battlefield_coords = step_info['battlefield_coords']
                    battlefield_coords_tep = [battlefield_coords["x"], battlefield_coords["y"]]
                    battlefield_all_coords.append(battlefield_coords_tep)
                    all_blue_uavs.append(step_info['blue_uavs'])
                else:
                    battlefield_all_coords.append([0, 0])
                    all_blue_uavs.append({
                        "ground_attack": 0,
                        "recon": 0,
                        "escort": 0
                    })

            for i in range(self.num_battlefields):
                ground_attack = all_blue_uavs[i]['ground_attack']
                recon = all_blue_uavs[i]['recon']
                escort = all_blue_uavs[i]['escort']
                processed_state[i * 7 + 4] = ground_attack
                processed_state[i * 7 + 5] = recon
                processed_state[i * 7 + 6] = escort

            # 获取动作
            actions = self.agent.get_actions(processed_state)
            env_actions = self.agent.action_to_env_format(actions)

            # 获取分配的无人机ID
            assigned_ids = self.get_assigned_uav_ids(env_actions)

            # 显示动作决策
            print(f"Actions = {env_actions}")

            # 执行动作
            next_state, rewards, done, info = env.step(env_actions, battlefield_all_coords)
            next_processed_state = self.agent.preprocess_state(next_state)

            total_reward += np.mean(rewards)

            # 记录步骤信息到JSON
            step_data = {
                "step": i,
                "battlefield_coords": battlefield_coords,
                "actions": [
                    {
                        "battlefield_id": j,
                        "action": env_actions[j],
                        "assigned_uavs": assigned_ids[j]
                    } for j in range(self.num_battlefields)
                ],
                "rewards": [float(r) for r in rewards],
                "average_reward": float(np.mean(rewards)),
                "cumulative_reward": float(total_reward),
                "battlefield_states": []
            }

            # 添加每个战场的信息
            for i in range(self.num_battlefields):
                battlefield_state = {
                    "battlefield_id": i,
                    "friendly_remaining": dict(next_state[i]['friendly_remaining']),
                    "enemy_remaining": dict(next_state[i]['enemy_remaining'])
                }
                step_data["battlefield_states"].append(battlefield_state)

            self.output_data["steps"].append(step_data)

            step += 1

            # 更新状态
            processed_state = next_processed_state

            # 渲染环境
            if render:
                env.render(env._get_state())
                time.sleep(0.5 / speed)  # 控制渲染速度

            # 如果环境结束，跳出循环
            if done:
                break

        # 完成推理后添加总结信息
        self.output_data["episode_info"]["end_time"] = datetime.now().isoformat()
        self.output_data["episode_info"]["total_steps"] = step
        self.output_data["episode_info"]["total_reward"] = float(total_reward)
        self.output_data["episode_info"]["remaining_uavs"] = {
            uav_type: len(ids) for uav_type, ids in self.uav_ids.items()
        }

        print(f"Inference completed:")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {total_reward:.3f}")

        # 保存JSON输出
        self.save_output_json()

        env.close()


def main():
    parser = argparse.ArgumentParser(description='QTRAN Model Inference with JSON Input/Output')
    parser.add_argument('--model', type=str, default="save_models/qtran_agent.pth", help='Path to the trained model')
    parser.add_argument('--config', type=str, default="test.json", help='Path to the JSON configuration file')
    parser.add_argument('--render', action='store_true', help='Enable rendering (default: False)')
    parser.add_argument('--speed', type=float, default=1.0, help='Rendering speed multiplier')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for inference')

    args = parser.parse_args()

    # 检查模型和配置文件是否存在
    try:
        renderer = QTRANInferenceRenderer(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )

        renderer.run_inference(
            render=args.render,
            speed=args.speed
        )

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please provide valid model and config paths.")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()