import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial import distance
import argparse
import os

def load_config(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def generate_fixed_drones(red_airports, drone_config):
    """根据传入的无人机配置生成无人机列表"""
    scouts = []
    for airport, ids in drone_config['scouts'].items():
        x, y = red_airports[airport]
        for drone_id in ids:
            scouts.append({"id": drone_id, "type": "scout", "airport": airport, "coords": (x, y)})
    
    interceptors = []
    for airport, ids in drone_config['interceptors'].items():
        x, y = red_airports[airport]
        for drone_id in ids:
            interceptors.append({"id": drone_id, "type": "interceptor", "airport": airport, "coords": (x, y)})
    
    return scouts, interceptors

def damage_expectation(s, l, a, e, r, p_le, p_la, eps, S, epsilon, R):
    rho = e / (a + e) if (a + e) > 0 else 0.5
    tau = eps * s / S
    sigma = epsilon * r / R
    l_e = l * rho
    l_a = l * (1 - rho)
    p_le_eff = max(0, min(1, p_le * (1 + tau) * (1 - sigma)))
    p_la_eff = max(0, min(1, p_la * (1 + tau) * (1 - sigma)))
    e_e = e * (1 - np.exp(-p_le_eff * l_e)) if l_e > 0 else 0
    e_a = a * (1 - np.exp(-p_la_eff * l_a)) if l_a > 0 else 0
    return e_e + e_a

def lagrangian_relaxation(K, S, L, a_list, e_list, r_list, p_le=0.8, p_la=0.7, 
                         eps=0.5, epsilon=0.2, max_iter=1000, tol=1e-3):
    if S < K or L < K:
        raise ValueError("资源不足: 需要 S ≥ K 和 L ≥ K")
    
    s_total = S - K
    l_total = L - K
    lambda_s, lambda_l = 0.0, 0.0
    history = []
    best_primal = {'s': None, 'l': None, 'value': -np.inf}
    
    step_size = 0.8
    step_decay = 0.9
    min_step = 0.001
    
    for iteration in range(max_iter):
        s_prime, l_prime, total_value = [], [], 0
        
        for k in range(K):
            a, e, r = a_list[k], e_list[k], r_list[k]
            max_s, max_l = S, L
            best_s, best_l, best_val = 0, 0, -np.inf
            
            for s in range(1, max_s + 1):
                for l in range(1, max_l + 1):
                    damage = damage_expectation(s, l, a, e, r, p_le, p_la, eps, S, epsilon, sum(r_list))
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
            'iter': iteration, 
            'lambda_s': lambda_s, 
            'lambda_l': lambda_l,
            'sum_s': sum_s, 
            'sum_l': sum_l, 
            'dual_value': dual_value,
            'primal_feasible': primal_feasible
        })
        
        if primal_feasible:
            s_actual = [sp + 1 for sp in s_prime]
            l_actual = [lp + 1 for lp in l_prime]
            primal_value = sum(
                damage_expectation(s_actual[k], l_actual[k], a_list[k], e_list[k], r_list[k],
                                p_le, p_la, eps, S, epsilon, sum(r_list)) 
                for k in range(K)
            )
            best_primal = {
                's': s_actual, 
                'l': l_actual, 
                'value': primal_value,
                'total_damage': primal_value,
                'dual_gap': dual_value - primal_value
            }
            break
        
        grad_s, grad_l = sum_s - s_total, sum_l - l_total
        grad_norm = np.sqrt(grad_s**2 + grad_l**2)
        if grad_norm > 0:
            step_size = max(min_step, step_size * step_decay)
            lambda_s = max(0, lambda_s + step_size * grad_s / grad_norm)
            lambda_l = max(0, lambda_l + step_size * grad_l / grad_norm)
    
    if best_primal['s'] is None:
        s_actual = [sp + 1 for sp in s_prime]
        l_actual = [lp + 1 for lp in l_prime]
        primal_value = sum(
            damage_expectation(s_actual[k], l_actual[k], a_list[k], e_list[k], r_list[k],
                            p_le, p_la, eps, S, epsilon, sum(r_list)) 
            for k in range(K)
        )
        best_primal = {
            's': s_actual, 
            'l': l_actual, 
            'value': primal_value,
            'total_damage': primal_value,
            'dual_gap': history[-1]['dual_value'] - primal_value if history else np.inf
        }
    
    best_primal['history'] = history  
    return best_primal

def assign_fixed_drones(drones, target_coords, need):
    drone_distances = [
        (drone, distance.euclidean(drone['coords'], target_coords), drone['id'])
        for drone in drones
    ]
    drone_distances.sort(key=lambda x: (x[1], x[2]))  # 距离优先，编号为辅
    return [d[0] for d in drone_distances[:need]]

def plot_drone_allocation(config, assigned_scouts, assigned_interceptors):
    """绘制无人机分配地图"""
    red_airports = config['problem']['positions']['red_airports']
    blue_clusters = config['problem']['positions']['blue_clusters']
    cluster_names = list(blue_clusters.keys())
    cluster_coords = {name: blue_clusters[name] for name in cluster_names}
    
    plt.figure(figsize=(12, 8))
    
    # 红方机场
    for name, (x, y) in red_airports.items():
        plt.scatter(x, y, c='red', s=300, marker='^', label='Red Airport' if name == 'Fuzhou Airport' else "")
        plt.text(x-0.3, y+0.1, name.split()[0], fontsize=9, color='red')
    
    # 蓝方集群
    for name, (x, y) in blue_clusters.items():
        plt.scatter(x, y, c='blue', s=300, marker='o', label='Blue Cluster' if name == 'Blue Cluster 1' else "")
        plt.text(x+0.1, y+0.1, name.split()[-1], fontsize=9, color='blue')
    
    # 侦察机路径
    for name in assigned_scouts:
        cx, cy = cluster_coords[name]
        for drone in assigned_scouts[name]:
            dx, dy = drone['coords']
            plt.plot([dx, cx], [dy, cy], 'g--', alpha=0.5)
            mx, my = (dx + cx)/2, (dy + cy)/2
            plt.text(mx+0.1, my+0.1, str(drone['id']), fontsize=8, color='green')
    
    # 拦截机路径
    for name in assigned_interceptors:
        cx, cy = cluster_coords[name]
        for drone in assigned_interceptors[name]:
            dx, dy = drone['coords']
            plt.plot([dx, cx], [dy, cy], 'orange', alpha=0.5)
            mx, my = (dx + cx)/2, (dy + cy)/2
            plt.text(mx-0.3, my-0.1, str(drone['id']), fontsize=8, color='orange')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Drone Allocation Map')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.show()

def plot_results(result, S, L, K):
    """绘制优化结果和收敛曲线（四张子图）"""
    plt.figure(figsize=(15, 10))
    
    # 1. 拉格朗日乘子收敛
    plt.subplot(2, 2, 1)
    plt.plot([h['iter'] for h in result['history']], [h['lambda_s'] for h in result['history']], 'b-', label='λ_s')
    plt.plot([h['iter'] for h in result['history']], [h['lambda_l'] for h in result['history']], 'g-', label='λ_l')
    plt.xlabel('Iteration')
    plt.ylabel('Lagrangian Multipliers')
    plt.title('Multipliers Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 侦察机分配收敛
    plt.subplot(2, 2, 2)
    plt.plot([h['iter'] for h in result['history']], [h['sum_s'] for h in result['history']], 'b-', label='Actual Allocation')
    plt.axhline(y=S-K, color='r', linestyle='--', label='Target')
    plt.xlabel('Iteration')
    plt.ylabel('Recon Aircraft (Transformed)')
    plt.title('Recon Allocation Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 拦截机分配收敛
    plt.subplot(2, 2, 3)
    plt.plot([h['iter'] for h in result['history']], [h['sum_l'] for h in result['history']], 'g-', label='Actual Allocation')
    plt.axhline(y=L-K, color='r', linestyle='--', label='Target')
    plt.xlabel('Iteration')
    plt.ylabel('Interceptors (Transformed)')
    plt.title('Interceptor Allocation Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 目标函数收敛
    plt.subplot(2, 2, 4)
    plt.plot([h['iter'] for h in result['history']], [h['dual_value'] for h in result['history']], 'k-', label='Dual Value')
    if result['dual_gap'] < np.inf:
        plt.axhline(y=result['total_damage'], color='r', linestyle='--', label='Primal Value')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --------------------------
# 4. 生成无人机分配详情JSON
# --------------------------
def generate_drone_allocation_json(cluster_names, assigned_scouts, assigned_interceptors, output_path):
    drone_allocation = []
    for name in cluster_names:
        drone_allocation.append({
            "cluster": name,
            "scout_ids": [d['id'] for d in assigned_scouts[name]],
            "interceptor_ids": [d['id'] for d in assigned_interceptors[name]]
        })
    
    output = {"drone_allocation_details": drone_allocation}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n无人机分配详情已保存至 {output_path}")

# --------------------------
# 主程序（适配新JSON标识）
# --------------------------
if __name__ == "__main__":
    # 加载配置
    parser = argparse.ArgumentParser(description='Infer deterministic alg in here.')
    parser.add_argument('--input_json', type=str, default="test_deterministic.json",
                        help='Path to input JSON file with configuration')
    parser.add_argument('--output_json', type=str, default="output.json", help="Path of the output json.")
    args = parser.parse_args()
    config = load_config(args.input_json)
    problem = config['problem']
    blue_clusters = problem['positions']['blue_clusters']
    cluster_names = list(blue_clusters.keys())
    
    # 解析参数
    K = problem['K']  # 集群数量
    # 红方资源：侦察机数量(scout)和拦截机数量(interceptor)
    red_uavs = problem['red_uavs']
    S = red_uavs['scout']  # 侦察机总数
    L = red_uavs['interceptor']  # 拦截机总数
    # 蓝方资源：攻击机、护航机、侦察机列表
    blue_uavs = problem['blue_uavs']
    a_list = blue_uavs['attack_list']  # 攻击机列表
    e_list = blue_uavs['escort_list']  # 护航机列表
    r_list = blue_uavs['recon_list']   # 侦察机列表
    # 位置信息
    red_airports = problem['positions']['red_airports']
    
    # 无人机编号配置
    red_drones = {  
        "scouts": {
            "Fuzhou Airport": [40, 41, 42, 43],  
            "Xiamen Airport": [44, 45, 46],
            "Quanzhou Airport": [47, 48, 49]
        },
        "interceptors": {
            "Fuzhou Airport": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "Xiamen Airport": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            "Quanzhou Airport": [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        }
    }
    
    # 生成无人机
    scouts, interceptors = generate_fixed_drones(red_airports, red_drones)
    print(f"已加载无人机: 侦察机 {len(scouts)} 架, 拦截机 {len(interceptors)} 架")
    
    # 运行优化
    result = lagrangian_relaxation(K, S, L, a_list, e_list, r_list)
    print(f"优化完成: 总毁伤期望 = {result['total_damage']:.4f}")
    
    # 分配无人机
    assigned_scouts = {name: [] for name in cluster_names}
    remaining_scouts = scouts.copy()
    for i, name in enumerate(cluster_names):
        need = result['s'][i]
        if need > 0:
            assigned = assign_fixed_drones(remaining_scouts, blue_clusters[name], need)
            assigned_scouts[name] = assigned
            for drone in assigned:
                remaining_scouts.remove(drone)
    
    assigned_interceptors = {name: [] for name in cluster_names}
    remaining_interceptors = interceptors.copy()
    for i, name in enumerate(cluster_names):
        need = result['l'][i]
        if need > 0:
            assigned = assign_fixed_drones(remaining_interceptors, blue_clusters[name], need)
            assigned_interceptors[name] = assigned
            for drone in assigned:
                remaining_interceptors.remove(drone)
    
    # 可视化结果
    # plot_drone_allocation(config, assigned_scouts, assigned_interceptors)  # 无人机分配地图
    # plot_results(result, S, L, K)  # 四张子图的收敛曲线
    
    # 生成并保存无人机分配详情JSON
    output_path = os.path.join("outputs_json/deterministic", args.output_json)
    generate_drone_allocation_json(cluster_names, assigned_scouts, assigned_interceptors, output_path)