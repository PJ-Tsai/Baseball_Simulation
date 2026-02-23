import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Config_Loader import config
from Logger_Setup import setup_logger

logger = setup_logger(__name__)

logger = setup_logger(__name__)

# 從配置讀取球場設定
PARK_CONFIG = config.get('park')
FOUL_LINE_M = PARK_CONFIG['foul_line_ft'] * 0.3048
CENTER_FIELD_M = PARK_CONFIG['center_field_ft'] * 0.3048
BASE_DIST_M = PARK_CONFIG['base_dist_ft'] * 0.3048
WALL_HEIGHT = PARK_CONFIG['wall_height_m']

def calculate_trajectory(v_kmh, angle_deg, direction_deg, physical_params, Cd=0.30):
    """
    計算棒球軌跡 (考慮空氣阻力)
    
    Args:
        v_kmh: 初速 (km/h)
        angle_deg: 仰角 (度)
        direction_deg: 方向角 (度)
        physical_params: 物理參數字典
        Cd: 阻力係數
    
    Returns:
        包含軌跡點和飛行時間的字典
    """
    logger.debug(f"計算軌跡: v={v_kmh:.1f}km/h, 仰角={angle_deg}°, 方向={direction_deg}°, Cd={Cd}")
    
    # Draw_Field 預設球場邊界為 0 ~ 90 度, 但實際上 Spray Angle 是以 0 度為中心
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
    
    t = 0
    max_iterations = 10000  # 避免無限迴圈
    iteration = 0
    
    while curr_z >= 0 and iteration < max_iterations:
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # 計算阻力加速度
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
        iteration += 1
    
    if iteration >= max_iterations:
        logger.warning(f"軌跡計算達到最大迭代次數，可能不收斂")
    
    distance = math.sqrt(x[-1]**2 + y[-1]**2)
    logger.debug(f"軌跡完成: 飛行時間={t:.2f}s, 距離={distance:.2f}m")
    
    return {
        "x": np.array(x), 
        "y": np.array(y), 
        "z": np.array(z), 
        "hang_time": t,
        "distance": distance
    }

def draw_field(ax):
    """繪製棒球場地"""
    logger.debug("繪製球場地圖")
    
    # 1. 繪製邊線 (Foul Lines)
    ax.plot([FOUL_LINE_M, 0, 0], [0, 0, FOUL_LINE_M], [0, 0, 0], 
            color='black', lw=2, label='Foul Lines')
    
    # 2. 繪製內野壘包連線 (Infield Diamond)
    diamond_x = [0, BASE_DIST_M, BASE_DIST_M, 0, 0]
    diamond_y = [0, 0, BASE_DIST_M, BASE_DIST_M, 0]
    diamond_z = [0, 0, 0, 0, 0]
    ax.plot(diamond_x, diamond_y, diamond_z, color='brown', lw=1.5, label='Infield')

    # 3. 繪製外野圍牆 (Outfield Wall)
    wall_theta = np.linspace(0, math.radians(90), 100)
    mid_angle = math.radians(45)
    radii = CENTER_FIELD_M - (CENTER_FIELD_M - FOUL_LINE_M) * (np.abs(wall_theta - mid_angle) / mid_angle)
    wall_x, wall_y = radii * np.cos(wall_theta), radii * np.sin(wall_theta)
    
    ax.plot(wall_x, wall_y, 0, color='darkgreen', lw=1, label='Wall Base')
    ax.plot(wall_x, wall_y, WALL_HEIGHT, color='darkgreen', lw=2, label='Wall Top')
    
    logger.debug("球場繪製完成")

def judge_result(data, wall_height=None):
    """判斷擊球結果"""
    if wall_height is None:
        wall_height = WALL_HEIGHT
        
    angle = data.get('phi', 45)  # 預設值
    x_pts, y_pts, z_pts = data['x'], data['y'], data['z']
    
    if not (0 <= angle <= 90):
        return "Foul Ball"
    
    # 計算該角度的全壘打牆距離
    deviation = abs(math.radians(angle) - math.radians(45)) / math.radians(45)
    wall_dist = CENTER_FIELD_M - (CENTER_FIELD_M - FOUL_LINE_M) * max(0, min(1, deviation))

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

def plot_trajectory(data):
    """繪製單一軌跡"""
    logger.info("繪製軌跡圖")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    draw_field(ax)
    res = judge_result(data)
    
    ax.plot(data['x'], data['y'], data['z'], color='red', lw=2, 
            label=f'Result: {res}')
    
    ax.set_xlim(-40, 160)
    ax.set_ylim(-40, 160)
    ax.set_zlim(0, 60)
    ax.set_title(f"Baseball Simulation\nResult: {res}")
    ax.set_xlabel("X: 1st Base Line")
    ax.set_ylabel("Y: 3rd Base Line")
    ax.legend()
    
    logger.info(f"軌跡結果: {res}")
    plt.show()

if __name__ == "__main__":
    # 測試腳本
    from Logger_Setup import setup_logger
    logger = setup_logger(__name__)
    
    logger.info("執行 Draw_Utils 測試")
    
    physical_params = {
        'g': config.get('physics', 'g'),
        'rho': config.get('physics', 'rho'),
        'area': config.get('physics', 'area'),
        'm': config.get('physics', 'm'),
        'dt': config.get('physics', 'dt'),
        'hit_pos': tuple(config.get('physics', 'hit_pos'))
    }
    
    hit_data = calculate_trajectory(125, 30, 30, physical_params)
    plot_trajectory(hit_data)