import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Draw_Utils import draw_field 
import random # 導入 random 模組

# --- 核心物理模擬函式 (支援自定義 Cd 並回傳飛行時間) ---
def calculate_custom_trajectory(v_kmh, angle_deg, direction_deg, start_pos, custom_cd):
    g, rho, area, m, dt = 9.81, 1.225, 0.00421, 0.145, 0.01
    x, y, z = [start_pos[0]], [start_pos[1]], [start_pos[2]]
    v0 = v_kmh * (1000 / 3600)
    theta, phi = np.radians(angle_deg), np.radians(direction_deg)
    
    vx = v0 * np.cos(theta) * np.cos(phi)
    vy = v0 * np.cos(theta) * np.sin(phi)
    vz = v0 * np.sin(theta)
    
    curr_x, curr_y, curr_z = start_pos
    t = 0.0
    
    while curr_z >= 0:
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        accel_drag = 0.5 * rho * v * custom_cd * area / m
        ax = -accel_drag * vx
        ay = -accel_drag * vy
        az = -g - accel_drag * vz
        
        vx += ax * dt; vy += ay * dt; vz += az * dt
        curr_x += vx * dt; curr_y += vy * dt; curr_z += vz * dt
        x.append(curr_x); y.append(curr_y); z.append(curr_z)
        t += dt
        
    return {
        'x': np.array(x), 
        'y': np.array(y), 
        'z': np.array(z), 
        'hang_time': t
    }

# --- 二分搜尋尋找最接近模型預測距離的軌跡 ---
def find_ml_fitted_trajectory(v_kmh, angle_deg, direction_deg, target_dist_m):
    """
    ================================================================================
    關於 阻力係數 (Drag Coefficient, Cd) 的定義與本專案中的應用
    ================================================================================

    1. 物理定義 (Standard Physics):
    Cd 是一個無因次參數，描述球體在流體（空氣）中移動時所受到的阻力大小。
    阻力公式為： Fd = 1/2 * rho * v^2 * Cd * A
    - Cd 越高：阻力越大，球速減慢越快，飛行距離越短。
    - Cd 越低：阻力越小，飛行越流線型，飛行距離越長。
    - 棒球的標準平均 Cd 值通常設定為 0.30。

    2. 本專案中的「有效阻力係數 (Effective Cd)」:
    在 Hybrid 模式中，我們不使用固定的 0.30，而是反推 Cd，其意義如下：
    
    A. 補償轉速 (Spin Compensation): 
        現實中棒球帶有旋轉，後旋 (Backspin) 會產生升力（馬格努斯力 Magnus Effect），
        讓球飛得比單純物理公式更遠。此時模型反推出的 Effective Cd 會低於 0.30。
        
    B. 環境變因 (Environmental Factors): 
        氣溫、氣壓、海拔、風向等因素會改變飛行距離。模型透過 7 萬筆真實數據學到了這些影響，
        並將其壓縮在預測距離中。我們反推 Cd 就是為了將這些「看不見的變因」視覺化。
        
    C. 數據與物理的橋樑 (Bridge): 
        利用「二分搜尋法 (Binary Search)」調整 Cd，直到物理模擬的落地點精準對齊模型預測的距離，
        確保視覺軌跡與大數據分析結果 100% 一致。

    3. 數值參考指標:
    - Cd < 0.25: 代表該擊球具有極佳的 Carry (可能帶有強大後旋或順風)。
    - Cd ≈ 0.30: 符合標準物理環境下的飛行狀態。
    - Cd > 0.35: 代表阻力較大 (可能帶有前旋 Topspin 或遭遇逆風)，球下墜較快。
    ================================================================================
    """
    low_cd, high_cd = 0.1, 1.2
    best_sim = None
    for _ in range(12):
        mid_cd = (low_cd + high_cd) / 2
        sim_data = calculate_custom_trajectory(v_kmh, angle_deg, direction_deg, (0,0,1.0), mid_cd)
        sim_dist = np.sqrt(sim_data['x'][-1]**2 + sim_data['y'][-1]**2)
        if sim_dist > target_dist_m: low_cd = mid_cd
        else: high_cd = mid_cd
        best_sim = sim_data
    return best_sim, mid_cd

# --- 主預測繪圖函式 (新增繪圖設定參數) ---
def predict_and_visualize_hybrid(speed_mph, angle_deg, spray_deg, model_path='baseball_dual_model.pkl', plot_config=None):
    # 1. 載入模型並強制轉 CPU
    try:
        data = joblib.load(model_path)
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 {model_path}")
        return

    # 確保模型物件存在
    if 'classifier' not in data or 'regressor' not in data:
        print("錯誤：模型檔案內容不完整，缺少分類器或迴歸器。")
        return

    clf = data['classifier'].set_params(device="cpu")
    reg = data['regressor'].set_params(device="cpu")
    features = data['features']
    label_map = data['label_map']
    
    # 確保 label_map 在模型文件中存在，否則使用預設值
    if label_map is None:
        print("警告：模型檔案中缺少 label_map，使用預設值。")
        label_map = {'OUT': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'HR': 4}
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    # 2. 準備輸入數據
    is_physically_foul = abs(spray_deg) > 45
    
    # 自動判斷 bb_type (根據仰角)
    def get_bb_type_from_angle(angle):
        if angle < 10: return 'ground_ball'
        elif 10 <= angle < 25: return 'line_drive'
        elif 25 <= angle < 50: return 'fly_ball'
        else: return 'popup'
    
    bb_type = get_bb_type_from_angle(angle_deg) # 自動判斷擊球類型

    input_row = {
        'launch_speed': speed_mph,
        'launch_angle': angle_deg,
        'spray_angle': spray_deg,
        'is_theoretical_fair': 0 if is_physically_foul else 1
    }
    for f in features:
        if f.startswith('type_'):
            input_row[f] = 1 if f == f"type_{bb_type}" else 0
    
    input_df = pd.DataFrame([input_row])[features]

    # 3. 模型預測
    pred_probs = clf.predict_proba(input_df)[0]
    pred_dist_ft = reg.predict(input_df)[0]
    target_dist_m = pred_dist_ft * 0.3048

    # 處理界外遮罩邏輯
    if is_physically_foul:
        pred_class_name = "FOUL (Physical Override)"
        hit_prob = 0.0
    else:
        # 確保 pred_probs 有足夠的類別索引
        pred_class_idx = np.argmax(pred_probs) if pred_probs.size > 0 else -1
        pred_class_name = target_names[pred_class_idx] if pred_class_idx != -1 else "UNKNOWN"
        hit_prob = sum(pred_probs[1:]) if pred_probs.size > 1 else 0.0 # 排除 'OUT' 機率

    # 4. 反推符合模型預測距離的物理軌跡
    v_kmh = speed_mph * 1.60934
    direction_deg = 45 + spray_deg 
    fitted_traj, effective_cd = find_ml_fitted_trajectory(v_kmh, angle_deg, direction_deg, target_dist_m)
    
    # 5. 繪圖與佈局調整
    fig = plt.figure(figsize=(12, 8)) # 加寬畫布以容納兩個子圖

    # --- 左側子圖：資訊顯示區 ---
    ax_text = fig.add_subplot(121) # 1行2列的第1個
    ax_text.axis('off') # 隱藏座標軸

    analysis_text = (
        f"MLB HIT ANALYSIS REPORT\n"
        f"{'='*35}\n"
        f"  [ INPUT PARAMETERS ]\n"
        f"  - Launch Speed:   {speed_mph:.1f} mph\n"
        f"  - Launch Angle:   {angle_deg:.1f}°\n"
        f"  - Spray Angle:    {spray_deg:.1f}°\n"
        f"  - BB Type (Auto): {bb_type}\n\n"
        f"  [ ML PREDICTIONS ]\n"
        f"  - Predicted Result: {pred_class_name}\n"
        f"  - Hit Probability:  {hit_prob:.1%}\n"
        f"  - Pred Distance:    {pred_dist_ft:.1f} ft\n\n"
        f"  [ PHYSICS ESTIMATION ]\n"
        f"  - Hang Time:      {fitted_traj['hang_time']:.2f} s\n"
        f"  - Effective Cd:   {effective_cd:.3f}\n"
        f"{'='*35}\n"
        f"  Ball Status:    "
        f"{'!!! FOUL BALL !!!' if is_physically_foul else 'IN PLAY'}"
    )

    # 放置文字，使用等寬字體讓排版整齊
    ax_text.text(0.1, 0.5, analysis_text, transform=ax_text.transAxes, 
                 fontsize=14, verticalalignment='center', family='monospace',
                 bbox=dict(facecolor='aliceblue', alpha=0.5, edgecolor='blue', boxstyle='round,pad=1'))

    # --- 右側子圖：3D 軌跡圖 ---
    ax_3d = fig.add_subplot(122, projection='3d') # 1行2列的第2個
    draw_field(ax_3d) # 呼叫您現有的畫場地函式
    
    traj_color = 'blue' if not is_physically_foul else 'gray'
    ax_3d.plot(fitted_traj['x'], fitted_traj['y'], fitted_traj['z'], 
               color=traj_color, lw=3, label='Hybrid Path')

    # 統一顯示設定
    ax_3d.set_zlim(0, 60)
    ax_3d.set_xlim(-40, 160)
    ax_3d.set_ylim(-40, 160)
    ax_3d.view_init(elev=25, azim=-45)
    ax_3d.set_title("3D Trajectory Simulation", fontsize=14)
    
    plt.tight_layout()
    plt.show()

# --- 主執行區塊：測試十組隨機數據 ---
if __name__ == "__main__":
    # 定義繪圖的預設配置
    default_plot_config = {
        'figure_size': (6, 8),
        'title_fontsize': 14,
        'xlabel': "1st Base Line (m)",
        'ylabel': "3rd Base Line (m)",
        'zlabel': "Height (m)",
        'xlim': (-40, 160),
        'ylim': (-40, 160),
        'zlim': (0, 60),
        'elev': 25,     # 視角仰角
        'azim': -45,    # 視角方位角
        'legend_loc': 'upper right',
        'text_x': 0.05,
        'text_y': 0.85,
        'text_fontsize': 11,
        'text_alpha': 0.8
    }
    
    test_sets = 2  # 測試組數量
    print(f"--- 開始進行 {test_sets} 組隨機數據測試 ---")
    for i in range(test_sets):
        speed_mph = random.uniform(50, 110)
        angle_deg = random.uniform(-15, 60) 
        spray_deg = random.uniform(-60, 60) # 包含界外角度

        print(f"\n--- 測試組 {i+1} ---")
        predict_and_visualize_hybrid(speed_mph, angle_deg, spray_deg, plot_config=default_plot_config)