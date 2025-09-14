import os
import torch
import numpy as np
from collections import Counter
import pygame
import argparse
import json
from datetime import datetime
import sys

sys.path.append('../')
from task_allocation import UAVCombatEnv, SCREEN_WIDTH, SCREEN_HEIGHT

# 导入您的模型定义
from train.dt_uav import DecisionTransformer, TrainConfig  # 替换为您的模型文件名


def generate_battlefield_coords():
    """生成随机的战场坐标"""
    return (
        np.random.uniform(SCREEN_WIDTH * 0.3, SCREEN_WIDTH * 0.7),
        np.random.uniform(SCREEN_HEIGHT * 0.3, SCREEN_HEIGHT * 0.7)
    )


def generate_enemy_formation(state):
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


def state_to_observation(state, enemy_formation=None):
    """将环境状态转换为observation向量"""
    friendly = state['friendly_remaining']

    obs = np.array([
        friendly['interceptor'],
        friendly['recon'],
        friendly['escort'],
        enemy_formation['ground_attack'] if enemy_formation else 0,
        enemy_formation['recon'] if enemy_formation else 0,
        enemy_formation['escort'] if enemy_formation else 0
    ], dtype=np.float32)

    return obs


class DTInferenceRenderer:
    def __init__(self, checkpoint_path, input_json=None, render=True, device="cpu"):
        self.device = device
        self.render = render
        self.input_json = input_json
        self.load_model(checkpoint_path)
        self.env = self.create_environment()
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

    def load_model(self, checkpoint_path):
        """加载训练好的模型"""
        print(f"Loading model from {checkpoint_path}")

        # 加载配置
        config = TrainConfig()
        config.state_dim = 6
        config.action_dim = 3
        config.seq_len = 20
        config.episode_len = 100
        config.embedding_dim = 128
        config.num_layers = 3
        config.num_heads = 1

        # 创建模型
        self.model = DecisionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            embedding_dim=config.embedding_dim,
            seq_len=config.seq_len,
            episode_len=config.episode_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_action=config.max_action,
        ).to(self.device)

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # 加载统计信息
        self.state_mean = checkpoint.get("state_mean", np.zeros((1, 6)))
        self.state_std = checkpoint.get("state_std", np.ones((1, 6)))

        print("Model loaded successfully!")

    def parse_input_json(self):
        """解析输入的JSON数据"""
        if self.input_json is None:
            return None

        try:
            with open(self.input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取红色方无人机配置
            red_uavs = data.get('red_uavs', {
                'interceptor': 40,
                'recon': 10,
                'escort': 0
            })

            # 提取红色方机场位置
            red_airport = data.get('red_airport', {
                'x': 807.1191291159989,
                'y': 540.0075557531011
            })

            # 提取步骤信息（如果有）
            steps = data.get('steps', [])
            return {
                'red_uavs': red_uavs,
                'red_airport': red_airport,
                'steps': steps
            }
        except Exception as e:
            print(f"Error parsing input JSON: {e}")
            return None

    def create_environment(self):
        """创建仿真环境"""
        # 解析输入JSON
        input_data = self.parse_input_json()

        if input_data:
            # 使用JSON中的配置
            red_uavs = Counter(input_data['red_uavs'])
            red_airport = (input_data['red_airport']['x'], input_data['red_airport']['y'])
        else:
            # 使用默认配置
            red_uavs = Counter({'interceptor': 40, 'recon': 10, 'escort': 0})
            red_airport = (807.1191291159989, 540.0075557531011)

        return UAVCombatEnv(
            initial_red_uav_counts=red_uavs,
            initial_blue_uav_counts=Counter({'ground_attack': 10, 'recon': 10, 'escort': 30}),
            render_mode=self.render,
            background_image_path='figure/背景_new.png',
            red_interceptor_image_path='figure/红.png',
            red_recon_image_path='figure/红2.png',
            red_escort_image_path='figure/红3.png',
            blue_ground_attack_image_path='figure/蓝.png',
            blue_recon_image_path='figure/蓝2.png',
            blue_escort_image_path='figure/蓝3.png',
        )

    def initialize_uav_ids(self):
        """初始化无人机编号"""
        self.uav_counter = 0
        self.uav_ids = {
            'interceptor': list(range(self.uav_counter, self.uav_counter + self.env.red_uavs['interceptor'])),
            'recon': list(range(self.uav_counter + self.env.red_uavs['interceptor'],
                                self.uav_counter + self.env.red_uavs['interceptor'] + self.env.red_uavs['recon'])),
            'escort': list(range(self.uav_counter + self.env.red_uavs['interceptor'] + self.env.red_uavs['recon'],
                                 self.uav_counter + self.env.red_uavs['interceptor'] + self.env.red_uavs['recon'] +
                                 self.env.red_uavs['escort']))
        }
        self.uav_counter += sum(self.env.red_uavs.values())

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

    def prepare_initial_buffer(self, target_return=80.0):
        """准备初始缓冲区"""
        states = torch.zeros(1, self.model.seq_len, self.model.state_dim,
                             dtype=torch.float, device=self.device)
        actions = torch.zeros(1, self.model.seq_len, self.model.action_dim,
                              dtype=torch.float, device=self.device)
        returns = torch.zeros(1, self.model.seq_len, dtype=torch.float, device=self.device)
        time_steps = torch.arange(self.model.seq_len, dtype=torch.long,
                                  device=self.device).view(1, -1)

        # 设置初始return-to-go
        returns[0, -1] = target_return

        return states, actions, returns, time_steps

    def run_episode(self, target_return=80.0):
        """运行一个完整的推理回合"""
        # 重置环境
        state = self.env.reset()
        self.initialize_uav_ids()

        enemy_state = {"current_enemy_formation_remaining": self.env.blue_uavs}
        enemy_formation = generate_enemy_formation(enemy_state)
        battlefield_coords = generate_battlefield_coords()

        # 准备缓冲区
        states, actions, returns, time_steps = self.prepare_initial_buffer(target_return)

        # 转换初始状态
        current_obs = state_to_observation(state, enemy_formation)
        states[0, -1] = torch.as_tensor(current_obs, device=self.device)

        episode_return = 0.0
        done = False
        step_count = 0

        print("Starting inference episode...")
        print(f"Initial state: {state}")

        while not done and step_count < self.model.episode_len:
            # 渲染当前状态（如果启用渲染）
            if self.render:
                self.env.render()

            # 模型推理
            with torch.no_grad():
                predicted_actions = self.model(
                    states=states,
                    actions=actions,
                    returns_to_go=returns,
                    time_steps=time_steps,
                )

            predicted_action = predicted_actions[0, -1].cpu().numpy()

            # 处理动作（确保在合理范围内）
            remaining_friendly = self.env.red_uavs
            action_dict = {
                'interceptor': min(max(0, int(predicted_action[0])), remaining_friendly['interceptor']),
                'recon': min(max(0, int(predicted_action[1])), remaining_friendly['recon']),
                'escort': 0
            }

            # 获取分配的无人机ID
            assigned_ids = self.get_assigned_uav_ids(action_dict)

            print(f"Step {step_count}: Action = {action_dict}")
            print(f"Assigned UAV IDs: {assigned_ids}")

            # 执行动作
            next_state, reward, done, info = self.env.step(
                action=action_dict,
                enemy_formation=enemy_formation,
                battlefield_coords=battlefield_coords
            )

            # 记录步骤信息到JSON
            step_info = {
                "step": step_count,
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
                "cumulative_reward": float(episode_return + reward),
                "friendly_remaining": dict(self.env.red_uavs),
                "enemy_remaining": dict(self.env.blue_uavs)
            }

            self.output_data["steps"].append(step_info)

            # 更新敌方编队和战场坐标
            enemy_state = {"current_enemy_formation_remaining": self.env.blue_uavs}
            enemy_formation = generate_enemy_formation(enemy_state)
            battlefield_coords = generate_battlefield_coords()

            # 转换下一个状态
            next_obs = state_to_observation(next_state, enemy_formation)

            # 更新缓冲区
            states = torch.roll(states, shifts=-1, dims=1)
            actions = torch.roll(actions, shifts=-1, dims=1)
            returns = torch.roll(returns, shifts=-1, dims=1)

            states[0, -1] = torch.as_tensor(next_obs, device=self.device)
            actions[0, -1] = torch.as_tensor(predicted_action, device=self.device)
            returns[0, -1] = torch.as_tensor(returns[0, -2] - reward, device=self.device)

            episode_return += reward
            step_count += 1

            # 添加延迟以便观察（如果启用渲染）
            if self.render:
                pygame.time.delay(100)

            # 显示当前状态信息
            print(f"Step {step_count}: Reward = {reward:.2f}, Total Return = {episode_return:.2f}")
            print(f"Red UAVs: {self.env.red_uavs}")
            print(f"Blue UAVs: {self.env.blue_uavs}")
            print("-" * 50)

        # 完成episode后添加总结信息
        self.output_data["episode_info"]["end_time"] = datetime.now().isoformat()
        self.output_data["episode_info"]["total_steps"] = step_count
        self.output_data["episode_info"]["total_return"] = float(episode_return)
        self.output_data["episode_info"]["final_friendly_remaining"] = dict(self.env.red_uavs)
        self.output_data["episode_info"]["final_enemy_remaining"] = dict(self.env.blue_uavs)

        print(f"Episode finished! Total return: {episode_return:.2f}, Steps: {step_count}")

        # 保存JSON输出
        self.save_output_json()

        return episode_return, step_count

    def save_output_json(self):
        """保存JSON输出到文件"""
        filename = f"dt_episode_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("outputs_json/dt", exist_ok=True)
        with open(f"outputs_json/dt/{filename}", 'w', encoding='utf-8') as f:
            json.dump(self.output_data, f, ensure_ascii=False, indent=2)
        print(f"Output saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='DT UAV Inference Renderer')
    parser.add_argument('--checkpoint', type=str,
                        default="save_models/dt_checkpoint.pt",
                        help='Path to the trained model checkpoint')
    parser.add_argument('--input_json', type=str, default=None,
                        help='Path to input JSON file with configuration')
    parser.add_argument('--target_return', type=float, default=80.0,
                        help='Target return for the episode')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'], help='Device to run inference on')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Enable rendering (default: True)')
    parser.add_argument('--no-render', dest='render', action='store_false',
                        help='Disable rendering')

    args = parser.parse_args()

    # 创建推理渲染器
    renderer = DTInferenceRenderer(
        checkpoint_path=args.checkpoint,
        input_json=args.input_json,
        render=args.render,
        device=args.device
    )

    try:
        # 运行推理
        total_return, total_steps = renderer.run_episode(args.target_return)

        print("\n" + "=" * 60)
        print(f"Inference completed!")
        print(f"Final Return: {total_return:.2f}")
        print(f"Total Steps: {total_steps}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    finally:
        if args.render:
            pygame.quit()


if __name__ == "__main__":
    main()