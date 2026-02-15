import joblib
import numpy as np
import matplotlib.pyplot as plt
from Simulation_Calculation import calculate_trajectory, draw_field

def find_effective_trajectory(v_kmh, angle_deg, direction_deg, target_dist_m):
    """
    透過二分搜尋法找到一個有效阻力係數 Cd，使得模擬距離等於模型預測距離
    """
    low_cd, high_cd = 0.1, 1.0  # 阻力係數的可能範圍
    best_sim = None
    
    # 進行 10 次迭代即可達到極高精度
    for _ in range(10):
        mid_cd = (low_cd + high_cd) / 2
        # 暫時修改物理模擬中的 Cd 邏輯 (這裡假設我們微調原本的函式)
        # 為了不改動原檔案，我們在此處模擬 calculate_trajectory 的行為
        sim_data = calculate_trajectory_with_custom_cd(v_kmh, angle_deg, direction_deg, (0,0,1.0), mid_cd)
        
        # 計算落地距離 (最後一個點的 x, y 向量長度)
        sim_dist = np.sqrt(sim_data['x'][-1]**2 + sim_data['y'][-1]**2)
        
        if sim_dist > target_dist_m:
            low_cd = mid_cd  # 飛太遠，增加阻力
        else:
            high_cd = mid_cd # 飛太近，減少阻力
        best_sim = sim_data
            
    return best_sim, mid_cd

def calculate_trajectory_with_custom_cd(v_kmh, angle_deg, direction_deg, start_pos, custom_cd):
    # 這是原本 Simulation_Calculation.py 的邏輯，但允許自定義 Cd
    g, rho, area, m, dt = 9.81, 1.225, 0.00421, 0.145, 0.01
    x, y, z = [start_pos[0]], [start_pos[1]], [start_pos[2]]
    v0 = v_kmh * (1000 / 3600)
    theta, phi = np.radians(angle_deg), np.radians(direction_deg)
    vx, vy, vz = v0 * np.cos(theta) * np.cos(phi), v0 * np.cos(theta) * np.sin(phi), v0 * np.sin(theta)
    
    curr_x, curr_y, curr_z = start_pos
    while curr_z >= 0:
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        f_drag = 0.5 * rho * v**2 * custom_cd * area
        ax = -(f_drag * (vx / v)) / m
        ay = -(f_drag * (vy / v)) / m
        az = -g - (f_drag * (vz / v)) / m
        vx += ax * dt; vy += ay * dt; vz += az * dt
        curr_x += vx * dt; curr_y += vy * dt; curr_z += vz * dt
        x.append(curr_x); y.append(curr_y); z.append(curr_z)
    return {'x': x, 'y': y, 'z': z}

# --- 主程式：結合模型預測 ---
def run_hybrid_prediction(speed_mph, angle_deg, spray_deg):
    # 1. 載入模型並預測距離
    model_data = joblib.load('baseball_dual_model.pkl')
    reg = model_data['regressor'].set_params(device="cpu")
    
    # 構造輸入 (簡化版，請確保與你訓練時的 features 一致)
    # 假設輸入包含: speed, angle, spray, is_fair
    input_df = ... # 此處略過 features 構造，參考前次 Predict_and_Visualize.py
    
    pred_dist_ft = reg.predict(input_df)[0]
    target_dist_m = pred_dist_ft * 0.3048 # 轉公制
    
    # 2. 反推物理曲線
    v_kmh = speed_mph * 1.60934
    direction_deg = 45 + spray_deg
    fitted_traj, effective_cd = find_effective_trajectory(v_kmh, angle_deg, direction_deg, target_dist_m)
    
    # 3. 繪圖
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    draw_field(ax)
    ax.plot(fitted_traj['x'], fitted_traj['y'], fitted_traj['z'], color='blue', lw=3, label='ML-Fitted Trajectory')
    
    print(f"預測距離: {pred_dist_ft:.1f} ft")
    print(f"反推有效阻力係數 (Effective Cd): {effective_cd:.3f}")
    plt.show()