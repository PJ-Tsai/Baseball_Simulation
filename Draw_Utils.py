import math
import numpy as np
import matplotlib.pyplot as plt

"""
Phsical_Params = {
    'g': 9.81,        # 重力加速度 (m/s^
    'rho': 1.225,    # 空氣密度 (kg/m^3)
    'area': 0.00421, # 棒球截面積 (m^2)
    'm': 0.145,       # 棒球質量 (kg)
    'dt': 0.01,        # 模擬時間步長 (秒)
    'hit_pos': (0, 0, 1.0)  # 擊球位置 (x, y, z) (m)
}
"""
def calculate_trajectory(v_kmh, angle_deg, direction_deg, physical_params, Cd=0.30):
    # --- 物理常數 ---
    # Draw_Field 預設球場邊界為 0 ~ 90 度, 但實際上 Spray Angle 是以 0 度為中心，向左負、向右正，因此需要調整方向角
    # 例如：Spray Angle -30 度實際上是向左偏 30 度，應該對應到 Draw_Field 的 15 度 (45 - 30)，而 Spray Angle +30 度則對應到 Draw_Field 的 75 度 (45 + 30)
    direction_deg = direction_deg + 45
    g = physical_params['g']
    rho = physical_params['rho']
    area = physical_params['area']
    m = physical_params['m']
    dt = physical_params['dt']
    hit_pos = physical_params['hit_pos']

    x0, y0, z0 = hit_pos
    v0 = v_kmh * (1000 / 3600)
    theta = math.radians(angle_deg)
    phi = math.radians(direction_deg)

    # 初始速度向量
    vx = v0 * math.cos(theta) * math.cos(phi)
    vy = v0 * math.cos(theta) * math.sin(phi)
    vz = v0 * math.sin(theta)

    # 初始化數據儲存
    x, y, z = [x0], [y0], [z0]
    curr_x, curr_y, curr_z = x0, y0, z0
    
    # --- 數值模擬 (考慮空氣阻力) ---
    t = 0
    while curr_z >= 0:
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # 計算阻力加速度 (a = F/m = 0.5 * rho * v^2 * Cd * A / m)
        accel_drag = 0.5 * rho * v * Cd * area / m
        
        # 更新各軸加速度
        ax = -accel_drag * vx
        ay = -accel_drag * vy
        az = -g - accel_drag * vz
        
        # 更新速度
        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        
        # 更新位置
        curr_x += vx * dt
        curr_y += vy * dt
        curr_z += vz * dt
        
        x.append(curr_x)
        y.append(curr_y)
        z.append(curr_z)
        t += dt

    return {
        "x": np.array(x), "y": np.array(y), "z": np.array(z), 
        #"phi": direction_deg, "v_kmh": v_kmh,
        "hang_time": t,
        "distance": math.sqrt(x[-1]**2 + y[-1]**2)
    }

def judge_result(data, wall_height=3.0):
    angle = data['phi']
    x_pts, y_pts, z_pts = data['x'], data['y'], data['z']
    
    foul_line_m = 328 * 0.3048
    center_field_m = 400 * 0.3048
    
    if not (0 <= angle <= 90):
        return "Foul Ball"
    
    # 計算該角度的全壘打牆距離
    deviation = abs(math.radians(angle) - math.radians(45)) / math.radians(45)
    wall_dist = center_field_m - (center_field_m - foul_line_m) * max(0, min(1, deviation))

    # 尋找球經過全壘打牆距離時的索引
    dists = np.sqrt(x_pts**2 + y_pts**2)
    idx_at_wall = np.where(dists >= wall_dist)[0]

    if len(idx_at_wall) == 0:
        return "Fair Ball"
    
    # 取得抵達牆面時的高度
    first_idx = idx_at_wall[0]
    z_at_wall = z_pts[first_idx]

    if z_at_wall >= wall_height:
        return "Home Run!"
    elif z_at_wall > 0:
        return "Fair Ball and Hit the Wall"
    else:
        return "Fair Ball"

# --- 繪圖與場地繪製函式不變 ---
def draw_field(ax):
    # 單位轉換與基本設定
    foul_line_m = 328 * 0.3048
    center_field_m = 400 * 0.3048
    base_dist = 90 * 0.3048  # 壘間距離 27.43 公尺
    wall_height = 3.0
    
    # 1. 繪製邊線 (Foul Lines)
    ax.plot([foul_line_m, 0, 0], [0, 0, foul_line_m], [0, 0, 0], color='black', lw=2)
    
    # 2. 繪製內野壘包連線 (Infield Diamond)
    # 座標點：本壘(0,0), 一壘(dist, 0), 二壘(dist, dist), 三壘(0, dist)
    diamond_x = [0, base_dist, base_dist, 0, 0]
    diamond_y = [0, 0, base_dist, base_dist, 0]
    diamond_z = [0, 0, 0, 0, 0]
    ax.plot(diamond_x, diamond_y, diamond_z, color='brown', lw=1.5, label='Infield')

    # 3. 繪製外野圍牆 (Outfield Wall)
    wall_theta = np.linspace(0, math.radians(90), 100)
    mid_angle = math.radians(45)
    radii = center_field_m - (center_field_m - foul_line_m) * (np.abs(wall_theta - mid_angle) / mid_angle)
    wall_x, wall_y = radii * np.cos(wall_theta), radii * np.sin(wall_theta)
    
    ax.plot(wall_x, wall_y, 0, color='darkgreen', lw=1)
    ax.plot(wall_x, wall_y, wall_height, color='darkgreen', lw=2)

def plot_trajectory(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    draw_field(ax)
    res = judge_result(data)
    ax.plot(data['x'], data['y'], data['z'], color='red', lw=2, label=f'Result: {res}')
    ax.set_xlim(-40, 160); ax.set_ylim(-40, 160); ax.set_zlim(0, 60)
    ax.set_title(f"Baseball Simulation: {data['v_kmh']} km/h\nResult: {res}")
    ax.set_xlabel("X: 1st Base Line")
    ax.set_ylabel("Y: 3rd Base Line")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    hit_data = calculate_trajectory(125, 30, 30, (0, 0, 1.0))
    plot_trajectory(hit_data)