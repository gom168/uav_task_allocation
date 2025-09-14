# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import h5py
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange  # noqa
import argparse

# 导入您的UAV环境
from task_allocation import *

@dataclass
class TrainConfig:
    def __init__(self, **kwargs):
        # 设置默认值
        defaults = {
            # tensorboard params
            'log_dir': "runs",
            # model params
            'embedding_dim': 128,
            'num_layers': 3,
            'num_heads': 1,
            'seq_len': 20,
            'episode_len': 100,
            'attention_dropout': 0.1,
            'residual_dropout': 0.1,
            'embedding_dropout': 0.1,
            'max_action': 1.0,
            # training params
            'dataset_path': "datasets/uav_combat_dataset.hdf5",
            'learning_rate': 1e-4,
            'betas': (0.9, 0.999),
            'weight_decay': 1e-4,
            'clip_grad': 0.25,
            'batch_size': 64,
            'update_steps': 50,
            'warmup_steps': 50,
            'reward_scale': 1,
            'num_workers': 4,
            # evaluation params
            'target_returns': (80.0,),
            'eval_episodes': 100,
            'eval_every': 2000,
            # general params
            'checkpoints_path': "save_models",
            'deterministic_torch': False,
            'train_seed': 10,
            'eval_seed': 42,
            'device': "cpu",
            'name': None  # 新增name参数
        }

        # 更新默认值
        defaults.update(kwargs)

        # 设置属性
        for key, value in defaults.items():
            setattr(self, key, value)

        # 如果没有提供name，则生成UUID
        if self.name is None:
            self.name = f"DT-UAV-{str(uuid.uuid4())[:8]}"

        # 设置checkpoints路径
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

# general utils
def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


# UAV环境特定的工具函数
def load_uav_trajectories(
        dataset_path: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    """从HDF5文件加载UAV轨迹数据"""
    with h5py.File(dataset_path, 'r') as f:
        observations = np.array(f['observations'][:])
        actions = np.array(f['actions'][:])
        rewards = np.array(f['rewards'][:])
        terminals = np.array(f['terminals'][:])

    traj, traj_len = [], []
    data_ = defaultdict(list)

    current_episode = 0
    for i in trange(len(observations), desc="Processing UAV trajectories"):
        data_["observations"].append(observations[i])
        data_["actions"].append(actions[i])
        data_["rewards"].append(rewards[i])

        if terminals[i] or i == len(observations) - 1:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # 计算return-to-go
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])

            # 重置轨迹缓冲区
            data_ = defaultdict(list)
            current_episode += 1

    # UAV环境的统计信息
    info = {
        "obs_mean": observations.mean(0, keepdims=True),
        "obs_std": observations.std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
        "action_mean": actions.mean(0, keepdims=True),
        "action_std": actions.std(0, keepdims=True) + 1e-6,
    }
    return traj, info


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def pad_along_axis(
        arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class UAVSequenceDataset(IterableDataset):
    def __init__(self, dataset_path: str, seq_len: int = 10, reward_scale: float = 1.0):
        self.dataset, info = load_uav_trajectories(dataset_path, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # 提取序列
        states = traj["observations"][start_idx: start_idx + self.seq_len]
        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        # 归一化
        # states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale

        # 填充到seq_len
        mask = np.hstack([
            np.ones(states.shape[0]),
            np.zeros(self.seq_len - states.shape[0])
        ])

        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["observations"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


# Decision Transformer implementation (保持不变)
class TransformerBlock(nn.Module):
    def __init__(
            self,
            seq_len: int,
            embedding_dim: int,
            num_heads: int,
            attention_dropout: float,
            residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    def forward(
            self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            seq_len: int = 10,
            episode_len: int = 1000,
            embedding_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            attention_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.0,
            max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), )
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            returns_to_go: torch.Tensor,
            time_steps: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]

        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )

        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                    .permute(0, 2, 1)
                    .reshape(batch_size, 3 * seq_len)
            )

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # out = self.out_norm(out)
        out = self.action_head(out[:, 1::3])
        return out


def generate_battlefield_coords():
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


def state_to_observation(state, enemy_formation=None):
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


# UAV环境评估函数
@torch.no_grad()
def eval_uav_rollout(
        model: DecisionTransformer,
        env: UAVCombatEnv,
        target_return: float,
        device: str = "cpu",
) -> Tuple[float, float]:
    """在UAV环境中进行评估"""
    env.reset()

    # 初始化缓冲区
    states = torch.zeros(1, model.seq_len, model.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(1, model.seq_len, model.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, model.seq_len, dtype=torch.float, device=device)
    time_steps = torch.arange(model.seq_len, dtype=torch.long, device=device).view(1, -1)

    # 获取初始状态
    current_state = env.reset()
    enemy_state = {"current_enemy_formation_remaining": env.blue_uavs}
    enemy_formation = generate_enemy_formation(enemy_state)
    battlefield_coords = generate_battlefield_coords()
    current_state = state_to_observation(current_state, enemy_formation)
    states[0, -1] = torch.as_tensor(current_state, device=device)
    returns[0, -1] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len = 0.0, 0
    done = False

    while not done and episode_len < model.episode_len:
        # 预测动作
        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
        )

        predicted_action = predicted_actions[0, -1].cpu().numpy()

        remaining_friendly = env.red_uavs
        # 转换为动作字典格式
        action_dict = {
            'interceptor': min(max(0, int(predicted_action[0])), remaining_friendly['interceptor']),
            'recon': min(max(0, int(predicted_action[1])), remaining_friendly['recon']),
            'escort': 0
        }
        # 生成敌方编队和战场坐标

        # 执行一步
        next_state, reward, done, info = env.step(
            action=action_dict,
            enemy_formation=enemy_formation,
            battlefield_coords=battlefield_coords
        )

        # 更新缓冲区
        states = torch.roll(states, shifts=-1, dims=1)
        actions = torch.roll(actions, shifts=-1, dims=1)
        returns = torch.roll(returns, shifts=-1, dims=1)

        enemy_state = {"current_enemy_formation_remaining": env.blue_uavs}
        enemy_formation = generate_enemy_formation(enemy_state)
        battlefield_coords = generate_battlefield_coords()
        next_state = state_to_observation(next_state, enemy_formation)

        states[0, -1] = torch.as_tensor(next_state, device=device)
        actions[0, -1] = torch.as_tensor(predicted_action, device=device)
        returns[0, -1] = torch.as_tensor(returns[0, -2] - reward, device=device)

        episode_return += reward
        episode_len += 1

    return episode_return, episode_len


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


def train(config):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)

    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.log_dir, config.name))

    # 数据加载
    dataset = UAVSequenceDataset(
        config.dataset_path, seq_len=config.seq_len, reward_scale=config.reward_scale
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    # 创建UAV评估环境
    eval_env = UAVCombatEnv(
        initial_red_uav_counts=Counter({'interceptor': 40, 'recon': 10, 'escort': 0}),
        initial_blue_uav_counts=Counter({'ground_attack': 10, 'recon': 10, 'escort': 30}),
        render_mode=False  # 禁用渲染以加快采样速度
    )

    # 模型设置
    config.state_dim = 6  # UAV环境的observation维度
    config.action_dim = 3  # UAV环境的action维度
    config.max_action = 1.0

    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)

    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        # 使用TensorBoard记录训练指标
        if step % 100 == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)

        # UAV环境评估
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_uav_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    eval_returns.append(eval_return / config.reward_scale)

                # 使用TensorBoard记录评估指标
                mean_return = np.mean(eval_returns)
                std_return = np.std(eval_returns)
                writer.add_scalar(f"eval/{target_return}_return_mean", mean_return, step)
                writer.add_scalar(f"eval/{target_return}_return_std", std_return, step)
                writer.add_scalar(f"eval/{target_return}_episode_len_mean", eval_len, step)

                print(f"Step {step}, Target {target_return}: Mean Return = {mean_return:.2f} ± {std_return:.2f}")
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train DT-UAV model')

    # 添加所有可能的命令行参数
    parser.add_argument('--log_dir', type=str, default="runs", help='Tensorboard log directory')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--episode_len', type=int, default=100, help='Episode length')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--residual_dropout', type=float, default=0.1, help='Residual dropout rate')
    parser.add_argument('--embedding_dropout', type=float, default=0.1, help='Embedding dropout rate')
    parser.add_argument('--max_action', type=float, default=1.0, help='Maximum action value')
    parser.add_argument('--dataset', '--dataset_path', type=str, default="datasets/uav_combat_dataset.hdf5",
                        help='Dataset path')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=0.25, help='Gradient clipping value')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--update_steps', type=int, default=50000, help='Number of update steps')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps')
    parser.add_argument('--reward_scale', type=float, default=1, help='Reward scaling factor')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--eval_every', type=int, default=2000, help='Evaluate every N steps')
    parser.add_argument('--checkpoints_path', type=str, default="save_models", help='Checkpoints directory')
    parser.add_argument('--train_seed', type=int, default=10, help='Training seed')
    parser.add_argument('--eval_seed', type=int, default=42, help='Evaluation seed')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use (cpu/cuda)')

    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 将参数转换为字典，注意处理特殊的参数名
    config = TrainConfig(
        dataset_path=args.dataset,
    )
    train(config)