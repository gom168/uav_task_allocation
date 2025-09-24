import numpy as np
import matplotlib.pyplot as plt
import json
import time
from scipy.spatial import distance
import argparse
import os


# -------------------------- 基础函数 --------------------------
def load_config(json_path):
    """加载配置文件（包含红蓝双方部署信息）"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"配置文件{json_path}未找到，使用默认配置")
        return get_default_config()

def get_default_config():
    """默认配置（无JSON文件时使用）"""
    return {
        "problem": {
            "K": 5,  # 蓝方集群数量
            "positions": {
                "red_airports": {
                    "Fuzhou Airport": (26.0990, 119.2959),
                    "Xiamen Airport": (24.5479, 118.1367),
                    "Quanzhou Airport": (24.8801, 118.5896)
                },
                "blue_clusters": {
                    "Cluster1": (25.0, 119.0),
                    "Cluster2": (25.5, 118.5),
                    "Cluster3": (24.8, 119.2),
                    "Cluster4": (25.2, 118.8),
                    "Cluster5": (24.9, 118.6)
                }
            },
            "blue_uavs": {
                "attack_list": [8, 5, 3, 6, 4],  # 各集群攻击机数量
                "escort_list": [3, 5, 7, 4, 6],  # 各集群护航机数量
                "recon_list": [2, 3, 1, 2, 3]    # 各集群侦察机数量
            }
        }
    }

def generate_fixed_drones(red_airports, drone_config):
    """生成固定数量无人机：侦察机10架、拦截机40架（含唯一ID）"""
    scouts = []
    for airport, ids in drone_config['scouts'].items():
        if airport not in red_airports:
            continue
        x, y = red_airports[airport]
        for drone_id in ids:
            scouts.append({
                "id": drone_id, "type": "scout", "airport": airport, 
                "coords": (x, y), "available": True
            })
    
    interceptors = []
    for airport, ids in drone_config['interceptors'].items():
        if airport not in red_airports:
            continue
        x, y = red_airports[airport]
        for drone_id in ids:
            interceptors.append({
                "id": drone_id, "type": "interceptor", "airport": airport, 
                "coords": (x, y), "available": True
            })
    
    assert len(scouts) == 10 and len(interceptors) == 40, "初始双机型数量错误（需10侦察机+40拦截机）"
    print(f" 初始无人机数量校验通过：侦察机10架，拦截机40架")
    print(f" 侦察机ID列表：{[d['id'] for d in scouts]}")
    print(f" 拦截机ID列表：{[d['id'] for d in interceptors]}")
    return scouts, interceptors

def generate_blue_uav_with_ids(blue_uav_count_list, uav_type):
    """为蓝方飞机生成带唯一ID的列表（攻击机/护航机）"""
    blue_uavs = []
    global_id = 1
    for cluster_idx, count in enumerate(blue_uav_count_list):
        for _ in range(count):
            uav_id = f"{uav_type}_00{global_id}" if global_id < 10 else f"{uav_type}_0{global_id}"
            blue_uavs.append({
                "id": uav_id,
                "type": uav_type,
                "cluster_idx": cluster_idx,  # 所属集群索引（0-4对应Cluster1-5）
                "assigned": True  # 默认为已部署
            })
            global_id += 1
    return blue_uavs

def damage_expectation(s, l, a, e, r, p_le=0.8, p_la=0.7, eps=0.5, S=10, epsilon=0.2, R=11, jam_factor=1.0):
    """损伤期望计算（双机型强关联：侦察机效能影响拦截机毁伤）"""
    rho = e / (a + e) if (a + e) > 0 else 0.5  # 护航机比例
    tau = eps * s / S * jam_factor  # 侦察机效能（含干扰）
    sigma = epsilon * r / R  # 蓝方侦察干扰
    l_e = l * rho  # 对抗护航机的拦截机数量
    l_a = l * (1 - rho)  # 对抗攻击机的拦截机数量
    
    # 有效杀伤概率
    p_le_eff = max(0, min(1, p_le * (1 + tau) * (1 - sigma)))
    p_la_eff = max(0, min(1, p_la * (1 + tau) * (1 - sigma)))
    
    # 毁伤期望
    e_e = e * (1 - np.exp(-p_le_eff * l_e)) if l_e > 0 else 0
    e_a = a * (1 - np.exp(-p_la_eff * l_a)) if l_a > 0 else 0
    return e_e + e_a


# -------------------------- 核心优化函数 --------------------------
class TimeoutError(Exception):
    """重规划超时异常（≤4.5秒）"""
    pass

def lagrangian_relaxation(K, S, L, a_list, e_list, r_list, p_le=0.8, p_la=0.7, 
                         eps=0.5, epsilon=0.2, max_iter=1000, tol=1e-5):
    """初始拉格朗日优化（双机型同步求解）"""
    if S < K or L < K:
        raise ValueError(f"资源不足：需侦察机S≥K（{S}≥{K}）且拦截机L≥K（{L}≥{K}）")
    
    s_total = S - K  # 侦察机可分配余量（每集群至少1架）
    l_total = L - K  # 拦截机可分配余量
    lambda_s, lambda_l = 0.0, 0.0  # 拉格朗日乘子初始化
    history = []
    best_primal = {'s': None, 'l': None, 'value': -np.inf}
    
    step_size = 0.8
    step_decay = 0.9
    min_step = 0.001
    R_total = sum(r_list)
    
    for iteration in range(max_iter):
        s_prime, l_prime, total_value = [], [], 0
        
        # 逐个集群优化双机型分配
        for k in range(K):
            a, e, r = a_list[k], e_list[k], r_list[k]
            best_s, best_l, best_val = 1, 1, -np.inf
            
            # 遍历所有可能的s和l组合
            for s in range(1, S + 1):
                for l in range(1, L + 1):
                    damage = damage_expectation(s, l, a, e, r, p_le, p_la, eps, S, epsilon, R_total)
                    val = damage - lambda_s * (s - 1) - lambda_l * (l - 1)
                    if val > best_val:
                        best_val = val
                        best_s, best_l = s, l
            
            s_prime.append(best_s - 1)
            l_prime.append(best_l - 1)
            total_value += best_val
        
        sum_s, sum_l = sum(s_prime), sum(l_prime)
        dual_value = total_value + lambda_s * (s_total - sum_s) + lambda_l * (l_total - sum_l)
        primal_feasible = (abs(sum_s - s_total) < tol and abs(sum_l - l_total) < tol)
        
        # 记录迭代历史
        history.append({
            'iter': iteration, 'lambda_s': lambda_s, 'lambda_l': lambda_l,
            'sum_s': sum_s, 'sum_l': sum_l, 'dual_value': dual_value,
            'primal_feasible': primal_feasible
        })
        
        # 可行解校验
        if primal_feasible:
            s_actual = [sp + 1 for sp in s_prime]
            l_actual = [lp + 1 for lp in l_prime]
            assert sum(s_actual) == S and sum(l_actual) == L, "分配总和错误"
            
            primal_value = sum(
                damage_expectation(s_actual[k], l_actual[k], a_list[k], e_list[k], r_list[k],
                                p_le, p_la, eps, S, epsilon, R_total) 
                for k in range(K)
            )
            best_primal = {
                's': s_actual, 'l': l_actual, 'value': primal_value,
                'total_damage': primal_value, 'dual_gap': dual_value - primal_value,
                'lambda_s': lambda_s, 'lambda_l': lambda_l
            }
            break
        
        # 更新拉格朗日乘子
        grad_s, grad_l = sum_s - s_total, sum_l - l_total
        grad_norm = np.sqrt(grad_s**2 + grad_l**2) if (grad_s != 0 or grad_l != 0) else 1e-12
        if grad_norm > 0:
            step_size = max(min_step, step_size * step_decay)
            lambda_s = max(0, lambda_s + step_size * grad_s / grad_norm)
            lambda_l = max(0, lambda_l + step_size * grad_l / grad_norm)
    
    # 兜底解（迭代未收敛时）
    if best_primal['s'] is None:
        s_actual = [sp + 1 for sp in s_prime]
        l_actual = [lp + 1 for lp in l_prime]
        
        # 调整侦察机数量
        if sum(s_actual) != S:
            diff = S - sum(s_actual)
            adjust_idx = np.argmax(s_actual) if diff > 0 else np.argmin(s_actual)
            s_actual[adjust_idx] += diff
        
        # 调整拦截机数量
        if sum(l_actual) != L:
            diff = L - sum(l_actual)
            adjust_idx = np.argmax(l_actual) if diff > 0 else np.argmin(l_actual)
            l_actual[adjust_idx] += diff
        
        assert sum(s_actual) == S and sum(l_actual) == L, "兜底解分配错误"
        primal_value = sum(
            damage_expectation(s_actual[k], l_actual[k], a_list[k], e_list[k], r_list[k],
                            p_le, p_la, eps, S, epsilon, R_total) 
            for k in range(K)
        )
        best_primal = {
            's': s_actual, 'l': l_actual, 'value': primal_value,
            'total_damage': primal_value, 'dual_gap': history[-1]['dual_value'] - primal_value if history else np.inf,
            'lambda_s': lambda_s, 'lambda_l': lambda_l
        }
    
    best_primal['history'] = history  
    return best_primal

def incremental_lagrangian(
    K, S, L, a_list, e_list, r_list, init_s, init_l, init_lambda_s, init_lambda_l,
    p_le=0.8, p_la=0.7, eps=0.5, epsilon=0.2, jam_factor=1.0,
    max_iter=800, tol=1e-5, timeout=4.5, start_time=None
):
    """增量优化（基于历史结果加速，支持干扰系数）"""
    if S < K or L < K:
        raise ValueError(f"故障后资源不足：需S≥K（{S}≥{K}）且L≥K（{L}≥{K}）")
    
    s_total = S - K
    l_total = L - K
    lambda_s, lambda_l = init_lambda_s, init_lambda_l
    history = []
    best_primal = {'s': init_s.copy(), 'l': init_l.copy(), 'value': -np.inf}
    R_total = sum(r_list)

    step_size = 0.8
    step_decay = 0.9
    min_step = 0.0001
    
    # 动态搜索范围（资源变化大则扩大范围）
    resource_change_ratio = abs(S - sum(init_s)) / sum(init_s) if sum(init_s) > 0 else 0
    search_range = 4 if resource_change_ratio > 0.05 else 2

    for iteration in range(max_iter):
        # 超时判断
        if start_time and time.time() - start_time > timeout:
            raise TimeoutError(f"重规划超时（{timeout}s）")
        
        s_prime, l_prime, total_value = [], [], 0
        
        for k in range(K):
            a, e, r = a_list[k], e_list[k], r_list[k]
            # 限定搜索范围（基于历史分配）
            min_s = max(1, init_s[k] - search_range)
            max_s = min(S, init_s[k] + search_range)
            min_l = max(1, init_l[k] - search_range)
            max_l = min(L, init_l[k] + search_range)
            
            best_s, best_l, best_val = init_s[k], init_l[k], -np.inf
            
            # 计算当前配置下的毁伤
            for s in range(min_s, max_s + 1):
                for l in range(min_l, max_l + 1):
                    damage = damage_expectation(
                        s, l, a, e, r, p_le, p_la, eps, S, epsilon, R_total, jam_factor
                    )
                    val = damage - lambda_s * (s - 1) - lambda_l * (l - 1)
                    if val > best_val:
                        best_val = val
                        best_s, best_l = s, l
            
            s_prime.append(best_s - 1)
            l_prime.append(best_l - 1)
            total_value += best_val

        sum_s, sum_l = sum(s_prime), sum(l_prime)
        dual_value = total_value + lambda_s * (s_total - sum_s) + lambda_l * (l_total - sum_l)
        primal_feasible = (abs(sum_s - s_total) < tol and abs(sum_l - l_total) < tol)
        
        history.append({
            'iter': iteration, 'lambda_s': lambda_s, 'lambda_l': lambda_l,
            'sum_s': sum_s, 'sum_l': sum_l, 'dual_value': dual_value,
            'primal_feasible': primal_feasible
        })

        # 可行解更新
        if primal_feasible and iteration > max_iter * 0.3:
            s_actual = [sp + 1 for sp in s_prime]
            l_actual = [lp + 1 for lp in l_prime]
            assert sum(s_actual) == S and sum(l_actual) == L, "重规划后总和错误"
            
            primal_value = sum(
                damage_expectation(s_actual[k], l_actual[k], a_list[k], e_list[k], r_list[k],
                                p_le, p_la, eps, S, epsilon, R_total, jam_factor) 
                for k in range(K)
            )
            if primal_value > best_primal['value']:
                best_primal = {
                    's': s_actual, 'l': l_actual, 'value': primal_value,
                    'total_damage': primal_value, 'dual_gap': dual_value - primal_value
                }
            break

        # 更新乘子
        grad_s, grad_l = sum_s - s_total, sum_l - l_total
        grad_norm = np.sqrt(grad_s**2 + grad_l**2) if (grad_s != 0 or grad_l != 0) else 1e-12
        if grad_norm > 1e-6:
            step_size = max(min_step, step_size * step_decay)
            lambda_s = max(0, lambda_s + step_size * grad_s / grad_norm)
            lambda_l = max(0, lambda_l + step_size * grad_l / grad_norm)

    # 兜底解（未找到更优解或资源变化未适配）
    if best_primal['value'] == -np.inf or (best_primal['s'] == init_s and resource_change_ratio > 0.05):
        s_actual = best_primal['s'].copy() if best_primal['s'] is not None else init_s.copy()
        l_actual = best_primal['l'].copy() if best_primal['l'] is not None else init_l.copy()
        
        # 调整侦察机
        if sum(s_actual) != S:
            diff = S - sum(s_actual)
            impact_scores = []
            for i in range(K):
                temp_s = s_actual.copy()
                temp_s[i] = max(1, temp_s[i] - 1)
                impact = damage_expectation(
                    temp_s[i], l_actual[i], a_list[i], e_list[i], r_list[i],
                    p_le, p_la, eps, S, epsilon, R_total, jam_factor
                )
                original = damage_expectation(
                    s_actual[i], l_actual[i], a_list[i], e_list[i], r_list[i],
                    p_le, p_la, eps, S, epsilon, R_total, jam_factor
                )
                impact_scores.append(original - impact)
            sorted_indices = np.argsort(impact_scores)
            idx = 0
            while diff != 0 and idx < len(sorted_indices):
                i = sorted_indices[idx]
                if diff < 0 and s_actual[i] > 1:
                    s_actual[i] -= 1
                    diff += 1
                elif diff > 0:
                    s_actual[i] += 1
                    diff -= 1
                idx += 1
        
        # 调整拦截机
        if sum(l_actual) != L:
            diff = L - sum(l_actual)
            sorted_indices = np.argsort([a_list[i] for i in range(K)])[::-1]  # 攻击机多的集群优先
            idx = 0
            while diff != 0 and idx < len(sorted_indices):
                i = sorted_indices[idx]
                if diff < 0 and l_actual[i] > 1:
                    l_actual[i] -= 1
                    diff += 1
                elif diff > 0:
                    l_actual[i] += 1
                    diff -= 1
                idx += 1
        
        assert sum(s_actual) == S and sum(l_actual) == L, "兜底解分配错误"
        primal_value = sum(
            damage_expectation(s_actual[k], l_actual[k], a_list[k], e_list[k], r_list[k],
                            p_le, p_la, eps, S, epsilon, R_total, jam_factor) 
            for k in range(K)
        )
        last_dual = history[-1]['dual_value'] if history else 0
        best_primal = {
            's': s_actual, 'l': l_actual, 'value': primal_value,
            'total_damage': primal_value, 'dual_gap': last_dual - primal_value
        }
    
    best_primal['history'] = history
    best_primal['algorithm'] = 'incremental_lagrangian'
    best_primal['scout_replanned'] = best_primal['s'] != init_s
    best_primal['interceptor_replanned'] = best_primal['l'] != init_l
    return best_primal


# -------------------------- 场景触发函数（新增ID记录） --------------------------
def trigger_drone_failure(scouts, interceptors, assigned_scouts, assigned_interceptors,
                         fail_type="scout", fail_airport="Fuzhou Airport", fail_count=1):
    """无人机故障：支持单次单类型故障，返回故障ID"""
    # 深拷贝避免修改原始数据
    scouts = [dict(d) for d in scouts]
    interceptors = [dict(d) for d in interceptors]
    assigned_scouts = {k: [dict(d) for d in v] for k, v in assigned_scouts.items()}
    assigned_interceptors = {k: [dict(d) for d in v] for k, v in assigned_interceptors.items()}
    
    failed_ids = []  # 记录故障无人机ID
    
    if fail_type == "scout":
        # 侦察机故障
        fail_candidates = [d for d in scouts if d['airport'] == fail_airport and d['available']]
        if len(fail_candidates) < fail_count:
            raise ValueError(f"{fail_airport}可用侦察机不足{fail_count}架（当前{len(fail_candidates)}架）")
        
        failed_drones = np.random.choice(fail_candidates, fail_count, replace=False)
        for d in failed_drones:
            d['available'] = False
            failed_ids.append(d['id'])  # 记录故障ID
            # 移除已故障无人机的分配记录
            for cluster in assigned_scouts:
                assigned_scouts[cluster] = [uav for uav in assigned_scouts[cluster] 
                                           if not (uav['id'] == d['id'] and uav['type'] == 'scout')]
        
        new_S = sum(1 for d in scouts if d['available'])
        new_L = sum(1 for d in interceptors if d['available'])
        print(f" 侦察机故障：{fail_airport}故障{fail_count}架，ID：{failed_ids} → 剩余{new_S}架")
        return new_S, new_L, scouts, interceptors, assigned_scouts, assigned_interceptors, failed_ids

    else:
        # 拦截机故障
        fail_candidates = [d for d in interceptors if d['airport'] == fail_airport and d['available']]
        if len(fail_candidates) < fail_count:
            raise ValueError(f"{fail_airport}可用拦截机不足{fail_count}架（当前{len(fail_candidates)}架）")
        
        failed_drones = np.random.choice(fail_candidates, fail_count, replace=False)
        for d in failed_drones:
            d['available'] = False
            failed_ids.append(d['id'])  # 记录故障ID
            for cluster in assigned_interceptors:
                assigned_interceptors[cluster] = [uav for uav in assigned_interceptors[cluster] 
                                                if not (uav['id'] == d['id'] and uav['type'] == 'interceptor')]
        
        new_S = sum(1 for d in scouts if d['available'])
        new_L = sum(1 for d in interceptors if d['available'])
        print(f" 拦截机故障：{fail_airport}故障{fail_count}架，ID：{failed_ids} → 剩余{new_L}架")
        return new_S, new_L, scouts, interceptors, assigned_scouts, assigned_interceptors, failed_ids


def trigger_target_change(e_list, blue_escorts, from_cluster_idx=0, to_cluster_idx=1, move_count=3):
    """护航机转移：返回转移的飞机ID"""
    init_e_total = sum(e_list)
    if e_list[from_cluster_idx] < move_count:
        raise ValueError(f"集群{from_cluster_idx+1}护航机不足{move_count}架（当前{e_list[from_cluster_idx]}架）")
    
    # 筛选源集群的护航机
    from_escorts = [uav for uav in blue_escorts if uav['cluster_idx'] == from_cluster_idx and uav['assigned']]
    if len(from_escorts) < move_count:
        raise ValueError(f"集群{from_cluster_idx+1}可用护航机不足{move_count}架（当前{len(from_escorts)}架）")
    
    # 选择转移的护航机
    moved_escorts = np.random.choice(from_escorts, move_count, replace=False)
    moved_ids = [uav['id'] for uav in moved_escorts]  # 记录转移ID
    
    # 更新护航机所属集群
    for uav in moved_escorts:
        uav['cluster_idx'] = to_cluster_idx
    
    # 更新数量列表
    new_e_list = e_list.copy()
    new_e_list[from_cluster_idx] -= move_count
    new_e_list[to_cluster_idx] += move_count
    
    assert sum(new_e_list) == init_e_total, "护航机转移后总数变化"
    print(f" 蓝方护航机转移：集群{from_cluster_idx+1}→集群{to_cluster_idx+1}（{move_count}架），ID：{moved_ids}")
    return new_e_list, blue_escorts, moved_ids

def trigger_attack_transfer(a_list, blue_attacks, from_cluster_idx=4, to_cluster_idx=3, move_count=2):
    """攻击机转移：返回转移的飞机ID"""
    init_a_total = sum(a_list)
    if a_list[from_cluster_idx] < move_count:
        raise ValueError(f"集群{from_cluster_idx+1}攻击机不足{move_count}架（当前{a_list[from_cluster_idx]}架）")
    
    # 筛选源集群的攻击机
    from_attacks = [uav for uav in blue_attacks if uav['cluster_idx'] == from_cluster_idx and uav['assigned']]
    if len(from_attacks) < move_count:
        raise ValueError(f"集群{from_cluster_idx+1}可用攻击机不足{move_count}架（当前{len(from_attacks)}架）")
    
    # 选择转移的攻击机
    moved_attacks = np.random.choice(from_attacks, move_count, replace=False)
    moved_ids = [uav['id'] for uav in moved_attacks]  # 记录转移ID
    
    # 更新攻击机所属集群
    for uav in moved_attacks:
        uav['cluster_idx'] = to_cluster_idx
    
    # 更新数量列表
    new_a_list = a_list.copy()
    new_a_list[from_cluster_idx] -= move_count
    new_a_list[to_cluster_idx] += move_count
    
    assert sum(new_a_list) == init_a_total, "攻击机转移后总数变化"
    print(f" 蓝方攻击机转移：集群{from_cluster_idx+1}→集群{to_cluster_idx+1}（{move_count}架），ID：{moved_ids}")
    return new_a_list, blue_attacks, moved_ids


def trigger_comm_jam(jam_airport="Xiamen Airport"):
    """通信干扰：厦门机场侦察机效能降低50%"""
    jam_factor = 0.5
    print(f" 通信干扰：{jam_airport}侦察机效能→50%（jam_factor={jam_factor}）")
    return jam_factor, jam_airport


# -------------------------- 双机型分配函数 --------------------------
def assign_fixed_drones(drones, target_coords, need):
    """单机型基础分配（按距离排序，距离近优先）"""
    available_drones = [d for d in drones if d['available']]
    if need > len(available_drones):
        raise ValueError(f"基础分配错误：需{need}架，可用仅{len(available_drones)}架")
    
    # 计算无人机到目标集群的距离
    drone_distances = [
        (d, distance.euclidean(d['coords'], target_coords), d['id'])
        for d in available_drones
    ]
    # 按距离升序、ID升序排序
    drone_distances.sort(key=lambda x: (x[1], x[2]))
    
    assigned = [d[0] for d in drone_distances[:need]]
    remaining = [d[0] for d in drone_distances[need:]]
    # 移除基础分配过程记录
    return assigned, remaining

def assign_fixed_drones_incremental(drones, target_coords, need, assigned_old, remaining_old):
    """单机型增量分配（优先保留历史分配）"""
    available_total = sum(1 for d in drones if d['available'])
    if need > available_total:
        raise ValueError(f"增量分配错误：需{need}架，可用仅{available_total}架")
    
    # 优先使用历史分配的可用无人机
    available_old = [d for d in assigned_old if d['available']]
    if len(available_old) >= need:
        assigned = available_old[:need]
        remaining = [d for d in drones if d['available'] and d not in assigned]
        # 移除增量分配过程记录
        return assigned, remaining
    
    # 历史分配不足，从剩余无人机中补充
    supplement_count = need - len(available_old)
    supplement, _ = assign_fixed_drones([d for d in remaining_old if d['available']], target_coords, supplement_count)
    assigned = available_old + supplement
    remaining = [d for d in drones if d['available'] and d not in assigned]
    
    # 移除增量分配过程记录
    assert len(assigned) == need, f"增量分配结果错误：需{need}架，实际{len(assigned)}架"
    return assigned, remaining

def assign_both_drones(
    scouts, interceptors, blue_clusters, cluster_names, 
    s_need_list, l_need_list, assigned_scouts_old, assigned_interceptors_old
):
    """双机型同步分配（侦察机+拦截机）"""
    assigned_scouts = {name: [] for name in cluster_names}
    assigned_interceptors = {name: [] for name in cluster_names}
    remaining_scouts = scouts.copy()
    remaining_interceptors = interceptors.copy()
    
    # 校验总需求与可用资源匹配
    total_s_needed = sum(s_need_list)
    total_s_available = sum(1 for d in scouts if d['available'])
    total_l_needed = sum(l_need_list)
    total_l_available = sum(1 for d in interceptors if d['available'])
    assert total_s_needed == total_s_available, \
        f"侦察机需求不匹配：需{total_s_needed}架，可用{total_s_available}架"
    assert total_l_needed == total_l_available, \
        f"拦截机需求不匹配：需{total_l_needed}架，可用{total_l_available}架"
    
    # 逐个集群分配双机型（移除集群分配提示）
    for i, name in enumerate(cluster_names):
        target_coords = blue_clusters[name]
        s_need = s_need_list[i]
        l_need = l_need_list[i]
        
        # 分配侦察机
        if s_need > 0:
            assigned_s, remaining_scouts = assign_fixed_drones_incremental(
                scouts, target_coords, s_need, assigned_scouts_old[name], remaining_scouts
            )
            assigned_scouts[name] = assigned_s
        
        # 分配拦截机
        if l_need > 0:
            assigned_l, remaining_interceptors = assign_fixed_drones_incremental(
                interceptors, target_coords, l_need, assigned_interceptors_old[name], remaining_interceptors
            )
            assigned_interceptors[name] = assigned_l
    
    # 最终校验
    total_s_assigned = sum(len(v) for v in assigned_scouts.values())
    total_l_assigned = sum(len(v) for v in assigned_interceptors.values())
    assert total_s_assigned == total_s_available and total_l_assigned == total_l_available, "分配总和错误"
    
    # 输出分配明细（保留汇总，移除过程记录）
    print(f"\n 双机型同步分配完成：")
    for name in cluster_names:
        s_ids = [d['id'] for d in assigned_scouts[name]]
        l_ids = [d['id'] for d in assigned_interceptors[name]]
        print(f"    {name}：侦察机{s_ids}（{len(s_ids)}架），拦截机{l_ids}（{len(l_ids)}架）")
    return assigned_scouts, assigned_interceptors


# -------------------------- 结果可视化与JSON输出（补充ID信息） --------------------------
def plot_results(result, S, L, K, scenario_name="Initial", subtitle=""):
    """结果可视化：调整图片尺寸+添加子图小标题+无总标题"""
    # 1. 调整图片尺寸：从(16,12)改为(14,10)，更紧凑适配显示
    plt.figure(figsize=(14, 10))
    # 彻底移除总标题（不调用plt.suptitle）
    
   # 1. 拉格朗日乘子收敛
    plt.subplot(2, 2, 1)
    iters = [h['iter'] for h in result['history']]
    plt.plot(iters, [h['lambda_s'] for h in result['history']], 'b-', linewidth=2, label='λ_s')
    plt.plot(iters, [h['lambda_l'] for h in result['history']], 'g-', linewidth=2, label='λ_l')
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Lagrangian Multipliers', fontsize=11)
    plt.title('Dual Multipliers Convergence', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. 侦察机分配收敛
    plt.subplot(2, 2, 2)
    plt.plot(iters, [h['sum_s'] for h in result['history']], 'b-', linewidth=2, label='Actual Allocation')
    plt.axhline(y=S-K, color='r', linestyle='--', linewidth=2, label=f'Target ')
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Scouts (Transformed)', fontsize=11)
    plt.title('Scout Allocation Convergence', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 3. 拦截机分配收敛
    plt.subplot(2, 2, 3)
    plt.plot(iters, [h['sum_l'] for h in result['history']], 'g-', linewidth=2, label='Actual Allocation')
    plt.axhline(y=L-K, color='r', linestyle='--', linewidth=2, label=f'Target')
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Interceptors (Transformed)', fontsize=11)
    plt.title('Interceptor Allocation Convergence', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 4. 目标函数收敛
    plt.subplot(2, 2, 4)
    plt.plot(iters, [h['dual_value'] for h in result['history']], 'k-', linewidth=2, label='Dual Value')
    if result['dual_gap'] < np.inf:
        plt.axhline(y=result['total_damage'], color='darkred', linestyle=':', linewidth=2, 
                   label=f'Primal Value ({result["total_damage"]:.4f})')
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Objective Function Value', fontsize=11)
    plt.title('Objective Function', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 微调子图间距，避免标题遮挡
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 预留更多顶部空间给子图标题
    plt.show()

def generate_drone_allocation_json(
    cluster_names, assigned_scouts, assigned_interceptors,
    output_path, scenario, S, L, details,
    failed_ids=None, moved_escort_ids=None, moved_attack_ids=None,
    algorithm_result=None # <--- 新增参数，用于传入算法结果
):
    """生成分配结果JSON文件（含故障/转移ID 和 算法结果）"""
    drone_allocation = []
    total_assigned_scouts = 0
    total_assigned_interceptors = 0

    for name in cluster_names:
        # 提取当前集群的分配信息
        scout_ids = [d['id'] for d in assigned_scouts[name] if d['available']]
        interceptor_ids = [d['id'] for d in assigned_interceptors[name] if d['available']]
        total_assigned_scouts += len(scout_ids)
        total_assigned_interceptors += len(interceptor_ids)

        cluster_detail = {
            "cluster_name": name,
            "scout_allocation": {
                "ids": scout_ids,
                "count": len(scout_ids),
                "available_total": S,
            },
            "interceptor_allocation": {
                "ids": interceptor_ids,
                "count": len(interceptor_ids),
                "available_total": L,
            },
            # "scenario": scenario, # 这个在顶层已经记录了
            "replan_details": details,
            # "algorithm": "incremental_lagrangian" # 这个也在顶层或result中了
        }

        # 补充故障/转移ID（按需添加）
        if failed_ids and scenario == "双机场故障场景":
            cluster_detail["failed_uav_ids"] = failed_ids
        if moved_escort_ids and scenario == "双集群转移场景":
            cluster_detail["moved_escort_ids"] = moved_escort_ids
        if moved_attack_ids and scenario == "双集群转移场景":
            cluster_detail["moved_attack_ids"] = moved_attack_ids

        drone_allocation.append(cluster_detail)

    # 校验分配总和
    assert total_assigned_scouts == S and total_assigned_interceptors == L, "JSON生成：分配数量错误"

    # 保存JSON（顶层补充全局故障/转移信息 和 算法结果）
    output_data = {
        "scenario_overview": scenario,
        "red_resource_summary": {
            "available_scouts": S,
            "available_interceptors": L,
        },
        "event_details": {},  # 全局事件信息
        "cluster_allocation_details": drone_allocation
        # <--- 新增：将算法结果直接添加到顶层
    }
    if algorithm_result:
        output_data["algorithm_result"] = algorithm_result

    if failed_ids:
        output_data["event_details"]["failed_uav_ids"] = failed_ids
        output_data["event_details"]["failed_description"] = "红方双机场无人机故障（福州拦截机+厦门侦察机）"
    if moved_escort_ids or moved_attack_ids:
        output_data["event_details"]["moved_escort_ids"] = moved_escort_ids if moved_escort_ids else []
        output_data["event_details"]["moved_attack_ids"] = moved_attack_ids if moved_attack_ids else []
        output_data["event_details"]["moved_description"] = "蓝方双集群飞机转移（护航机+攻击机）"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f" {scenario}场景JSON保存完成：{output_path}")

# -------------------------- 主程序（执行所有场景，输出ID信息） --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer deterministic alg in here.')
    parser.add_argument('--input_json', type=str, default="new_config.json",
                        help='Path to input JSON file with configuration')
    parser.add_argument('--output_json', type=str, default="init_allocation.json", help="Path of the output json.")
    parser.add_argument('--output_scene1', type=str, default='scene1_failure_allocation.json')
    parser.add_argument('--output_scene2', type=str, default='scene2_transfer_allocation.json')
    parser.add_argument('--output_scene3', type=str, default='scene2_transfer_allocation.json')
    args = parser.parse_args()
    try:
        # 1. 初始化配置与参数
        config = load_config(args.input_json)  # 无该文件时自动使用默认配置
        problem = config['problem']
        blue_clusters = problem['positions']['blue_clusters']  # 蓝方集群坐标
        cluster_names = list(blue_clusters.keys())
        K = problem['K']  # 蓝方集群数量（默认5）
        assert K <= 10 and K <= 40, f"集群数K={K}超出双机型资源上限（侦察机10架/拦截机40架）"
        
        # 蓝方初始配置（攻击机、护航机、侦察机）
        blue_uavs = problem['blue_uavs']
        init_a_list = blue_uavs['attack_list']  # 攻击机数量列表
        init_e_list = blue_uavs['escort_list']  # 护航机数量列表
        init_r_list = blue_uavs['recon_list']    # 侦察机数量列表
        
        # 为蓝方攻击机/护航机生成带ID的列表（关键新增）
        blue_attacks = generate_blue_uav_with_ids(init_a_list, "attack")  # 攻击机ID：attack_001~attack_26
        blue_escorts = generate_blue_uav_with_ids(init_e_list, "escort")  # 护航机ID：escort_001~escort_25
        print(f"初始蓝方集群配置：")
        for i, name in enumerate(cluster_names):
            # 统计各集群的飞机ID
            cluster_attack_ids = [uav['id'] for uav in blue_attacks if uav['cluster_idx'] == i]
            cluster_escort_ids = [uav['id'] for uav in blue_escorts if uav['cluster_idx'] == i]
            print(f"  {name}：攻击机{init_a_list[i]}架（ID：{cluster_attack_ids}），"
                  f"护航机{init_e_list[i]}架（ID：{cluster_escort_ids}），侦察机{init_r_list[i]}架")
        
        # 红方机场与无人机配置
        red_airports = problem['positions']['red_airports']
        red_drones_config = {  # 固定无人机ID分配
            "scouts": {
                "Fuzhou Airport": [40, 41, 42, 43],  
                "Xiamen Airport": [44, 45, 46],
                "Quanzhou Airport": [47, 48, 49]
            },
            "interceptors": {
                "Fuzhou Airport": [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
                "Xiamen Airport": [14,15,16,17,18,19,20,21,22,23,24,25,26],
                "Quanzhou Airport": [27,28,29,30,31,32,33,34,35,36,37,38,39]
            }
        }
        
        # 生成初始无人机（10侦察机+40拦截机，含ID）
        scouts, interceptors = generate_fixed_drones(red_airports, red_drones_config)
        init_S = 10  # 初始侦察机数量
        init_L = 40  # 初始拦截机数量


        # 2. 初始优化（拉格朗日松弛法）
        print(f"\n" + "="*70)
        print("初始优化：拉格朗日松弛法（双机型同步求解）")
        print("="*70)
        init_start = time.time()
        init_result = lagrangian_relaxation(
            K=K, S=init_S, L=init_L,
            a_list=init_a_list, e_list=init_e_list, r_list=init_r_list
        )
        init_time = time.time() - init_start
        
        # 输出初始优化结果
        print(f"初始优化结果：")
        print(f"  求解时间：{init_time:.2f}s，收敛迭代：{len(init_result['history'])}次")
        print(f"  总毁伤期望：{init_result['total_damage']:.4f}")
        print(f"  侦察机分配：{init_result['s']}（总和：{sum(init_result['s'])} = {init_S}架）")
        print(f"  拦截机分配：{init_result['l']}（总和：{sum(init_result['l'])} = {init_L}架）")
        print(f"  最优拉格朗日乘子：λ_s={init_result['lambda_s']:.4f}，λ_l={init_result['lambda_l']:.4f}")
        
        # 初始双机型分配
        assigned_scouts_init, assigned_interceptors_init = assign_both_drones(
            scouts, interceptors, blue_clusters, cluster_names,
            init_result['s'], init_result['l'],
            {name: [] for name in cluster_names},  # 初始无历史分配
            {name: [] for name in cluster_names}
        )
        
        # 生成初始JSON与可视化
        generate_drone_allocation_json(
            cluster_names, assigned_scouts_init, assigned_interceptors_init,
            args.output_json, "初始场景", init_S, init_L,
            "初始优化：拉格朗日松弛法，双机型同步求解",
            algorithm_result=init_result  # <--- 传入算法结果
        )
        # plot_results(init_result, init_S, init_L, K, "Initial Optimization", "Scouts=10, Interceptors=40")


        # 3. 场景1：双机场故障（福州1架拦截机 + 厦门1架侦察机，输出故障ID）
        print("="*70)
        print("场景1：双机场故障（福州1架拦截机 + 厦门1架侦察机）")
        print("="*70)
        scene1_start = time.time()
        all_failed_ids = []  # 汇总所有故障ID
        
        # 步骤1：福州机场故障1架拦截机（返回故障ID）
        new_S1_1, new_L1_1, scouts1_1, interceptors1_1, assigned_s1_1, assigned_l1_1, failed_interceptor_ids = trigger_drone_failure(
            scouts, interceptors, assigned_scouts_init, assigned_interceptors_init,
            fail_type="interceptor", fail_airport="Fuzhou Airport", fail_count=1
        )
        all_failed_ids.extend(failed_interceptor_ids)
        
        # 步骤2：厦门机场故障1架侦察机（返回故障ID）
        new_S1, new_L1, scouts1, interceptors1, assigned_s1, assigned_l1, failed_scout_ids = trigger_drone_failure(
            scouts1_1, interceptors1_1, assigned_s1_1, assigned_l1_1,
            fail_type="scout", fail_airport="Xiamen Airport", fail_count=1
        )
        all_failed_ids.extend(failed_scout_ids)
        print(f" 故障后红方总资源：侦察机{new_S1}架，拦截机{new_L1}架，累计故障ID：{all_failed_ids}")
        
        # 增量优化（基于初始结果加速）
        try:
            scene1_result = incremental_lagrangian(
                K=K, S=new_S1, L=new_L1,
                a_list=init_a_list, e_list=init_e_list, r_list=init_r_list,  # 蓝方配置不变
                init_s=init_result['s'], init_l=init_result['l'],
                init_lambda_s=init_result['lambda_s'], init_lambda_l=init_result['lambda_l'],
                start_time=scene1_start, timeout=4.5
            )
        except TimeoutError:
            print("  警告：重规划超时，使用兜底解")
            scene1_result = init_result.copy()
            # 调整侦察机至9架（10-1）
            s_diff = new_S1 - sum(scene1_result['s'])
            impact_scores = []
            for i in range(K):
                temp_s = scene1_result['s'].copy()
                temp_s[i] = max(1, temp_s[i] - 1)
                impact = damage_expectation(
                    temp_s[i], scene1_result['l'][i], init_a_list[i], init_e_list[i], init_r_list[i],
                    S=new_S1, R_total=sum(init_r_list)
                )
                original = damage_expectation(
                    scene1_result['s'][i], scene1_result['l'][i], init_a_list[i], init_e_list[i], init_r_list[i],
                    S=new_S1, R_total=sum(init_r_list)
                )
                impact_scores.append(original - impact)
            sorted_indices = np.argsort(impact_scores)
            idx = 0
            while s_diff != 0 and idx < len(sorted_indices):
                i = sorted_indices[idx]
                if s_diff < 0 and scene1_result['s'][i] > 1:
                    scene1_result['s'][i] -= 1
                    s_diff += 1
                idx += 1
            # 调整拦截机至39架（40-1）
            l_diff = new_L1 - sum(scene1_result['l'])
            sorted_l_idx = np.argsort([init_a_list[i] for i in range(K)])[::-1]
            idx = 0
            while l_diff != 0 and idx < len(sorted_l_idx):
                i = sorted_l_idx[idx]
                if l_diff < 0 and scene1_result['l'][i] > 1:
                    scene1_result['l'][i] -= 1
                    l_diff += 1
                idx += 1
            scene1_result['history'] = init_result['history']  # 复用初始历史（超时简化）
        
        # 故障后双机型分配
        assigned_s1_final, assigned_l1_final = assign_both_drones(
            scouts1, interceptors1, blue_clusters, cluster_names,
            scene1_result['s'], scene1_result['l'],
            assigned_s1, assigned_l1
        )
        
        # 输出场景1结果（含故障ID）
        scene1_time = time.time() - scene1_start
        print(f"场景1重规划结果：")
        print(f"  重规划时间：{scene1_time:.2f}s")
        print(f"  总毁伤期望：{scene1_result['total_damage']:.4f}（初始：{init_result['total_damage']:.4f}）")
        print(f"  侦察机分配：{scene1_result['s']}（总和：{sum(scene1_result['s'])} = {new_S1}架）")
        print(f"  拦截机分配：{scene1_result['l']}（总和：{sum(scene1_result['l'])} = {new_L1}架）")
        print(f"  故障无人机ID汇总：{all_failed_ids}")
        
        # 生成场景1 JSON（含故障ID）
        generate_drone_allocation_json(
            cluster_names, assigned_s1_final, assigned_l1_final,
            args.output_scene1, "双机场故障场景", new_S1, new_L1,
            "故障详情：福州机场1架拦截机+厦门机场1架侦察机，蓝方配置不变",
            failed_ids=all_failed_ids,  # 传入故障ID
            algorithm_result=scene1_result  # <--- 传入算法结果
        )
        # plot_results(
        #     scene1_result, new_S1, new_L1, K,
        #     "Scene 1: Dual Airport Drone Failure",
        #     f"Scouts={new_S1}, Interceptors={new_L1} - Failed IDs: {all_failed_ids}"
        # )


        # 4. 场景2：双集群转移（护航机+攻击机，输出转移ID）
        print("="*70)
        print("场景2：双集群转移（集群1→2护航机3架 + 集群5→4攻击机2架）")
        print("="*70)
        scene2_start = time.time()
        moved_escort_ids = []  # 转移的护航机ID
        moved_attack_ids = []  # 转移的攻击机ID
        
        # 步骤1：集群1→2转移3架护航机（返回转移ID）
        new_e_list2, blue_escorts2, moved_escort_ids = trigger_target_change(
            init_e_list.copy(), blue_escorts.copy(), 
            from_cluster_idx=0, to_cluster_idx=1, move_count=3
        )
        
        # 步骤2：集群5→4转移2架攻击机（返回转移ID，集群5对应索引4）
        new_a_list2, blue_attacks2, moved_attack_ids = trigger_attack_transfer(
            init_a_list.copy(), blue_attacks.copy(), 
            from_cluster_idx=4, to_cluster_idx=3, move_count=2
        )
        
        # 打印转移后蓝方各集群的飞机ID
        print(f" 转移后蓝方各集群配置：")
        for i, name in enumerate(cluster_names):
            cluster_attack_ids = [uav['id'] for uav in blue_attacks2 if uav['cluster_idx'] == i]
            cluster_escort_ids = [uav['id'] for uav in blue_escorts2 if uav['cluster_idx'] == i]
            print(f"    {name}：攻击机{new_a_list2[i]}架（ID：{cluster_attack_ids}），"
                  f"护航机{new_e_list2[i]}架（ID：{cluster_escort_ids}）")
        
        # 增量优化（基于初始结果，使用转移后蓝方配置）
        try:
            scene2_result = incremental_lagrangian(
                K=K, S=init_S, L=init_L,  # 红方资源不变
                a_list=new_a_list2, e_list=new_e_list2, r_list=init_r_list,  # 蓝方配置更新
                init_s=init_result['s'], init_l=init_result['l'],
                init_lambda_s=init_result['lambda_s'], init_lambda_l=init_result['lambda_l'],
                start_time=scene2_start, timeout=4.5
            )
        except TimeoutError:
            print("  警告：重规划超时，使用兜底解")
            scene2_result = init_result.copy()
            # 攻击机多的集群（集群4）多分配拦截机
            max_a_idx = np.argmax(new_a_list2)
            min_a_idx = np.argmin(new_a_list2)
            if scene2_result['l'][max_a_idx] <= scene2_result['l'][min_a_idx]:
                scene2_result['l'][max_a_idx] += 1
                scene2_result['l'][min_a_idx] -= 1
            # 护航机多的集群（集群2）多分配侦察机
            max_e_idx = np.argmax(new_e_list2)
            min_e_idx = np.argmin(new_e_list2)
            if scene2_result['s'][max_e_idx] <= scene2_result['s'][min_e_idx]:
                scene2_result['s'][max_e_idx] += 1
                scene2_result['s'][min_e_idx] -= 1
            scene2_result['history'] = init_result['history']
        
        # 转移后双机型分配
        assigned_s2_final, assigned_l2_final = assign_both_drones(
            scouts.copy(), interceptors.copy(), blue_clusters, cluster_names,
            scene2_result['s'], scene2_result['l'],
            assigned_scouts_init.copy(), assigned_interceptors_init.copy()
        )
        
        # 输出场景2结果（含转移ID）
        scene2_time = time.time() - scene2_start
        print(f"场景2重规划结果：")
        print(f"  重规划时间：{scene2_time:.2f}s")
        print(f"  总毁伤期望：{scene2_result['total_damage']:.4f}（初始：{init_result['total_damage']:.4f}）")
        print(f"  侦察机分配：{scene2_result['s']}（总和：{sum(scene2_result['s'])} = {init_S}架）")
        print(f"  拦截机分配：{scene2_result['l']}（总和：{sum(scene2_result['l'])} = {init_L}架）")
        print(f"  转移飞机ID汇总：护航机{moved_escort_ids}，攻击机{moved_attack_ids}")
        
        # 生成场景2 JSON（含转移ID）
        generate_drone_allocation_json(
            cluster_names, assigned_s2_final, assigned_l2_final,
            args.output_scene2, "双集群转移场景", init_S, init_L,
            "转移详情：集群1→2护航机3架+集群5→4攻击机2架，红方资源不变",
            moved_escort_ids=moved_escort_ids, moved_attack_ids=moved_attack_ids,  # 传入转移ID
            algorithm_result=scene2_result  # <--- 传入算法结果
        )
        # plot_results(
        #     scene2_result, init_S, init_L, K,
        #     "Scene 2: Dual Cluster UAV Transfer",
        #     f"Escort Moved IDs: {moved_escort_ids}, Attack Moved IDs: {moved_attack_ids}"
        # )


        # 5. 场景3：通信干扰（厦门机场侦察机效能降低50%）
        print("="*70)
        print("场景3：通信干扰（厦门机场侦察机效能降低50%）")
        print("="*70)
        scene3_start = time.time()
        
        # 触发通信干扰（获取干扰系数）
        jam_factor3, jam_airport3 = trigger_comm_jam(jam_airport="Xiamen Airport")
        
        # 增量优化（考虑干扰系数）
        try:
            scene3_result = incremental_lagrangian(
                K=K, S=init_S, L=init_L,  # 红方资源不变
                a_list=init_a_list, e_list=init_e_list, r_list=init_r_list,  # 蓝方配置不变
                init_s=init_result['s'], init_l=init_result['l'],
                init_lambda_s=init_result['lambda_s'], init_lambda_l=init_result['lambda_l'],
                jam_factor=jam_factor3,  # 加入干扰系数
                start_time=scene3_start, timeout=4.5
            )
        except TimeoutError:
            print("  警告：重规划超时，使用兜底解")
            scene3_result = init_result.copy()
            # 非干扰机场（福州、泉州）多分配侦察机
            s_idx = np.argmax(scene3_result['s'])
            if [d for d in scouts if d['id'] == scene3_result['s'][s_idx] and d['airport'] == jam_airport3]:
                scene3_result['s'][s_idx] -= 1
                scene3_result['s'][np.argmin(scene3_result['s'])] += 1
            scene3_result['history'] = init_result['history']
        
        # 干扰后双机型分配（优先非干扰机场无人机）
        scouts3 = scouts.copy()
        # 非干扰机场侦察机优先分配
        non_jam_scouts = [d for d in scouts3 if d['airport'] != jam_airport3 and d['available']]
        jam_scouts = [d for d in scouts3 if d['airport'] == jam_airport3 and d['available']]
        scouts3 = non_jam_scouts + jam_scouts
        
        assigned_s3_final, assigned_l3_final = assign_both_drones(
            scouts3, interceptors.copy(), blue_clusters, cluster_names,
            scene3_result['s'], scene3_result['l'],
            assigned_scouts_init.copy(), assigned_interceptors_init.copy()
        )
        
        # 输出场景3结果
        scene3_time = time.time() - scene3_start
        print(f"场景3重规划结果：")
        print(f"  重规划时间：{scene3_time:.2f}s")
        print(f"  总毁伤期望：{scene3_result['total_damage']:.4f}（初始：{init_result['total_damage']:.4f}）")
        print(f"  侦察机分配：{scene3_result['s']}（总和：{sum(scene3_result['s'])} = {init_S}架）")
        print(f"  拦截机分配：{scene3_result['l']}（总和：{sum(scene3_result['l'])} = {init_L}架）")
        print(f"  干扰机场：{jam_airport3}，受影响侦察机ID：{[d['id'] for d in jam_scouts]}")
        
        # 生成场景3 JSON
        generate_drone_allocation_json(
            cluster_names, assigned_s3_final, assigned_l3_final,
            args.output_scene3, "通信干扰场景", init_S, init_L,
            f"干扰详情：{jam_airport3}侦察机效能降低50%（jam_factor={jam_factor3}），蓝方配置不变",
            algorithm_result=scene3_result  # <--- 传入算法结果
        )
        # plot_results(
        #     scene3_result, init_S, init_L, K,
        #     "Scene 3: Communication Jam",
        #     f"Jam at {jam_airport3} (Scout Efficacy=50%) - Jam Factor={jam_factor3}"
        # )


        # # 6. 所有场景结果汇总（含ID信息）
        # print(f"\n" + "="*70)
        # print("所有场景结果汇总（含故障/转移ID）")
        # print("="*70)
        # print(f"初始场景：毁伤{init_result['total_damage']:.4f}，侦察机10架（ID：{[d['id'] for d in scouts]}），"
        #       f"拦截机40架（ID：{[d['id'] for d in interceptors]}）")
        # print(f"场景1（双机场故障）：毁伤{scene1_result['total_damage']:.4f}，故障ID：{all_failed_ids}，"
        #       f"剩余侦察机{new_S1}架，拦截机{new_L1}架")
        # print(f"场景2（双集群转移）：毁伤{scene2_result['total_damage']:.4f}，转移ID：护航机{moved_escort_ids}、"
        #       f"攻击机{moved_attack_ids}，红方资源不变")
        # print(f"场景3（通信干扰）：毁伤{scene3_result['total_damage']:.4f}，干扰机场{jam_airport3}，"
        #       f"受影响侦察机ID：{[d['id'] for d in jam_scouts]}")
        # print("\n所有场景执行完成！")

    except Exception as e:
        print(f"\n 程序运行错误: {str(e)}")
        import traceback
        traceback.print_exc()