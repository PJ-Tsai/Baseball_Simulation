import math
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Config_Loader import config
from Logger_Setup import setup_logger
from scipy import interpolate

logger = setup_logger(__name__)

# 從配置讀取球場設定
PARK_CONFIG = config.get('park')
DEFAULT_PARK_ID = PARK_CONFIG.get('default_id', 0)

# 通用球場設定
GENERIC_PARK = PARK_CONFIG.get('generic', {})
FOUL_LINE_M = GENERIC_PARK.get('foul_line_ft', 328) * 0.3048
CENTER_FIELD_M = GENERIC_PARK.get('center_field_ft', 400) * 0.3048
BASE_DIST_M = GENERIC_PARK.get('base_dist_ft', 90) * 0.3048
WALL_HEIGHT = GENERIC_PARK.get('wall_height_m', 3.0)

# 球場ID映射表
PARK_ID_MAPPING = PARK_CONFIG.get('park_id_mapping', {0: "generic"})

# 載入特定球場數據
specific_park_data = None
park_profiles = {}  # 按球場名稱索引
park_profiles_by_id = {}  # 按球場ID索引

def load_specific_park_data():
    """載入特定球場數據"""
    global specific_park_data, park_profiles, park_profiles_by_id
    
    if specific_park_data is not None:
        return specific_park_data
    
    park_file = PARK_CONFIG.get('specific_park_data')
    if not park_file:
        logger.warning("未設定特定球場數據檔案")
        return None
    
    try:
        with open(park_file, 'r', encoding='utf-8') as f:
            specific_park_data = json.load(f)
        
        # 建立快速查詢表
        for park in specific_park_data['ballparks']:
            park_id = park['park_id']
            park_name = park['name']
            
            park_profiles[park_name] = {
                'park_id': park_id,
                'name': park_name,
                'angles': np.array(park['angles_plot']),
                'distances': np.array(park['distances']) * 0.3048,  # 轉換為公尺
                'wall_heights': np.array(park['wall_heights']) * 0.3048
            }
            # 也按ID索引
            park_profiles_by_id[park_id] = park_profiles[park_name]
            
            # 創建插值函數，用於任意角度的距離和牆高計算
            park_profiles[park_name]['distance_interp'] = interpolate.interp1d(
                park_profiles[park_name]['angles'],
                park_profiles[park_name]['distances'],
                kind='linear',
                bounds_error=False,
                fill_value=(park_profiles[park_name]['distances'][0], 
                           park_profiles[park_name]['distances'][-1])
            )
            park_profiles[park_name]['height_interp'] = interpolate.interp1d(
                park_profiles[park_name]['angles'],
                park_profiles[park_name]['wall_heights'],
                kind='linear',
                bounds_error=False,
                fill_value=(park_profiles[park_name]['wall_heights'][0],
                           park_profiles[park_name]['wall_heights'][-1])
            )
        
        logger.info(f"成功載入 {len(park_profiles)} 個特定球場數據")
        return specific_park_data
    except Exception as e:
        logger.error(f"載入特定球場數據失敗: {e}")
        return None

def get_park_name_by_id(park_id):
    """
    根據球場ID獲取球場名稱
    
    Args:
        park_id: 球場ID，0表示通用球場
    
    Returns:
        球場名稱
    """
    if park_id == 0 or park_id not in PARK_ID_MAPPING:
        return "generic"
    return PARK_ID_MAPPING[park_id]

def get_park_config(park_id=None):
    """
    獲取球場配置
    
    Args:
        park_id: 球場ID，如果為 None 或 0 則使用預設球場
    
    Returns:
        球場配置字典
    """
    if park_id is None:
        park_id = DEFAULT_PARK_ID
    
    # 獲取球場名稱
    park_name = get_park_name_by_id(park_id)
    
    if park_name == "generic":
        return {
            'type': 'generic',
            'park_id': 0,
            'foul_line_m': FOUL_LINE_M,
            'center_field_m': CENTER_FIELD_M,
            'base_dist_m': BASE_DIST_M,
            'wall_height': WALL_HEIGHT,
            'name': 'Generic Park'
        }
    else:
        # 載入特定球場數據
        load_specific_park_data()
        
        if park_name in park_profiles:
            return {
                'type': 'specific',
                'park_id': park_id,
                'name': park_name,
                'angles': park_profiles[park_name]['angles'],
                'distances': park_profiles[park_name]['distances'],
                'wall_heights': park_profiles[park_name]['wall_heights'],
                'distance_interp': park_profiles[park_name]['distance_interp'],
                'height_interp': park_profiles[park_name]['height_interp'],
                'base_dist_m': BASE_DIST_M,  # 內野距離仍使用通用設定
                'max_distance': np.max(park_profiles[park_name]['distances'])  # 最大距離用於標籤位置
            }
        else:
            logger.warning(f"找不到球場ID '{park_id}' (名稱: {park_name})，使用通用球場")
            return get_park_config(0)

def get_wall_distance(angle_deg, park_config):
    """
    獲取指定角度的全壘打牆距離
    
    Args:
        angle_deg: 角度 (0~90度)
        park_config: 球場配置
    
    Returns:
        牆距離 (公尺)
    """
    if park_config['type'] == 'generic':
        # 通用球場的距離計算
        mid_angle = 45
        deviation = abs(angle_deg - mid_angle) / mid_angle
        wall_dist = (park_config['center_field_m'] - 
                    (park_config['center_field_m'] - park_config['foul_line_m']) * 
                    max(0, min(1, deviation)))
        return wall_dist
    else:
        # 特定球場，使用插值
        return float(park_config['distance_interp'](angle_deg))

def get_wall_height(angle_deg, park_config):
    """
    獲取指定角度的全壘打牆高度
    
    Args:
        angle_deg: 角度 (0~90度)
        park_config: 球場配置
    
    Returns:
        牆高度 (公尺)
    """
    if park_config['type'] == 'generic':
        return park_config['wall_height']
    else:
        return float(park_config['height_interp'](angle_deg))

def calculate_trajectory(v_kmh, angle_deg, direction_deg, physical_params, Cd=0.30, park_id=None):
    """
    計算棒球軌跡 (考慮空氣阻力)
    
    Args:
        v_kmh: 初速 (km/h)
        angle_deg: 仰角 (度)
        direction_deg: 方向角 (度)
        physical_params: 物理參數字典
        Cd: 阻力係數
        park_id: 球場ID
    
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
    
    # 獲取球場配置
    park_config = get_park_config(park_id)

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
        "distance": distance,
        "park_id": park_id,
        "park_config": park_config,
        "cd": Cd
    }

def draw_field(ax, park_id=None):
    """繪製棒球場地"""
    park_name = get_park_name_by_id(park_id) if park_id else "generic"
    logger.debug(f"繪製球場地圖: ID={park_id}, 名稱={park_name}")
    
    park_config = get_park_config(park_id)
    
    if park_config['type'] == 'generic':
        # 繪製通用球場
        _draw_generic_field(ax, park_config)
    else:
        # 繪製特定球場
        _draw_specific_field(ax, park_config)
    
    # 繪製內野壘包連線 (共用)
    base_dist = park_config['base_dist_m']
    diamond_x = [0, base_dist, base_dist, 0, 0]
    diamond_y = [0, 0, base_dist, base_dist, 0]
    diamond_z = [0, 0, 0, 0, 0]
    ax.plot(diamond_x, diamond_y, diamond_z, color='brown', lw=1.5, label='Infield')
    
    logger.debug(f"球場繪製完成: {park_config['name']}")

def _draw_generic_field(ax, park_config):
    """繪製通用球場"""
    foul_line = park_config['foul_line_m']
    center_field = park_config['center_field_m']
    
    # 繪製邊線 (Foul Lines)
    ax.plot([foul_line, 0, 0], [0, 0, foul_line], [0, 0, 0], 
            color='black', lw=2, label='Foul Lines')
    
    # 繪製外野圍牆
    wall_theta = np.linspace(0, math.radians(90), 100)
    mid_angle = math.radians(45)
    radii = center_field - (center_field - foul_line) * (np.abs(wall_theta - mid_angle) / mid_angle)
    wall_x, wall_y = radii * np.cos(wall_theta), radii * np.sin(wall_theta)
    
    # 用淺綠色填滿全壘打牆
    wall_height = park_config['wall_height']
    
    # 建立牆面網格
    for i in range(len(wall_x)-1):
        # 建立四邊形頂點
        x_quad = [wall_x[i], wall_x[i+1], wall_x[i+1], wall_x[i]]
        y_quad = [wall_y[i], wall_y[i+1], wall_y[i+1], wall_y[i]]
        z_quad = [0, 0, wall_height, wall_height]
        
        # 填滿牆面
        ax.plot_surface(
            np.array([x_quad[:2], x_quad[2:]]),
            np.array([y_quad[:2], y_quad[2:]]),
            np.array([[0, 0], [wall_height, wall_height]]),
            color='lightgreen', alpha=0.3
        )
    
    # 繪製牆基線和牆頂線
    ax.plot(wall_x, wall_y, 0, color='darkgreen', lw=1)
    ax.plot(wall_x, wall_y, wall_height, color='darkgreen', lw=2, label='Wall Top')
    
    # 在0度和90度處畫上黃色標竿
    # 0度（右外野邊線）
    x_0 = foul_line * math.cos(0)
    y_0 = foul_line * math.sin(0)
    ax.plot([x_0, x_0], [y_0, y_0], [0, 15], 
            color='yellow', lw=3, marker='o', markersize=5)
    
    # 90度（左外野邊線）
    x_90 = foul_line * math.cos(math.radians(90))
    y_90 = foul_line * math.sin(math.radians(90))
    ax.plot([x_90, x_90], [y_90, y_90], [0, 15], 
            color='yellow', lw=3, marker='o', markersize=5)

def _draw_specific_field(ax, park_config):
    """繪製特定球場"""
    angles = park_config['angles']
    distances = park_config['distances']
    
    # 轉換為笛卡爾座標
    angles_rad = np.radians(angles)
    wall_x = distances * np.cos(angles_rad)
    wall_y = distances * np.sin(angles_rad)
    wall_heights = park_config['wall_heights']
    
    # 繪製邊線 (連接端點)
    ax.plot([wall_x[0], 0, 0], [wall_y[0], 0, wall_y[-1]], [0, 0, 0],
            color='black', lw=2, label='Foul Lines')
    
    # 用淺綠色填滿全壘打牆
    for i in range(len(wall_x)-1):
        # 使用該段牆的平均高度或插值
        avg_height = (wall_heights[i] + wall_heights[i+1]) / 2
        
        # 建立四邊形頂點
        x_quad = [wall_x[i], wall_x[i+1], wall_x[i+1], wall_x[i]]
        y_quad = [wall_y[i], wall_y[i+1], wall_y[i+1], wall_y[i]]
        z_quad = [0, 0, avg_height, avg_height]
        
        # 填滿牆面
        ax.plot_surface(
            np.array([x_quad[:2], x_quad[2:]]),
            np.array([y_quad[:2], y_quad[2:]]),
            np.array([[0, 0], [avg_height, avg_height]]),
            color='lightgreen', alpha=0.3
        )
    
    # 繪製牆基線
    ax.plot(wall_x, wall_y, 0, color='darkgreen', lw=1, label='Wall Base')
    
    # 繪製牆頂（使用牆高數據）
    ax.plot(wall_x, wall_y, wall_heights, color='darkgreen', lw=2, label='Wall Top')
    
    # 在0度和90度處畫上黃色標竿
    # 0度（右外野邊線） - 使用第一個數據點的位置
    x_0 = wall_x[0]
    y_0 = wall_y[0]
    pole_height_0 = 15  # 標竿高度固定為15公尺
    ax.plot([x_0, x_0], [y_0, y_0], [0, pole_height_0], 
            color='yellow', lw=3, marker='o', markersize=5)
    
    # 90度（左外野邊線） - 使用最後一個數據點的位置
    x_90 = wall_x[-1]
    y_90 = wall_y[-1]
    pole_height_90 = 15  # 標竿高度固定為15公尺
    ax.plot([x_90, x_90], [y_90, y_90], [0, pole_height_90], 
            color='yellow', lw=3, marker='o', markersize=5)
    
    # 添加球場名稱標籤
    max_dist = park_config.get('max_distance', np.max(distances))
    label_x = max_dist * 0.7 * math.cos(math.radians(45))
    label_y = max_dist * 0.7 * math.sin(math.radians(45))
    ax.text(label_x, label_y, 0, 
            park_config['name'], fontsize=12, fontweight='bold', color='darkgreen')

def plot_trajectory(trajectory_data, title=None, park_id=None):
    """
    繪製單一軌跡（供外部呼叫）
    
    Args:
        trajectory_data: 包含 'x', 'y', 'z' 的軌跡數據
        title: 圖表標題
        park_id: 球場ID
    """
    logger.info("繪製軌跡圖")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    draw_field(ax, park_id)
    
    # 繪製軌跡
    ax.plot(trajectory_data['x'], trajectory_data['y'], trajectory_data['z'], 
            color='red', lw=2, label='Trajectory')
    
    # 標記落點
    if len(trajectory_data['x']) > 0:
        ax.scatter(trajectory_data['x'][-1], trajectory_data['y'][-1], 0, 
                  color='red', s=100, marker='o', label='Landing Point')
    
    # 獲取球場配置以動態調整顯示範圍
    park_config = get_park_config(park_id)
    if park_config['type'] == 'generic':
        max_field_dist = max(park_config['center_field_m'], FOUL_LINE_M) * 1.2
    else:
        max_field_dist = np.max(park_config['distances']) * 1.2
    
    max_ball_dist = np.sqrt(np.max(trajectory_data['x'])**2 + np.max(trajectory_data['y'])**2)
    max_dist = max(max_field_dist, max_ball_dist * 1.2)
    
    ax.set_xlim(-max_dist*0.2, max_dist)
    ax.set_ylim(-max_dist*0.2, max_dist)
    ax.set_zlim(0, max(np.max(trajectory_data['z']), 30) * 1.2)
    
    park_display = park_config['name']
    if title:
        ax.set_title(f"{title}\n{park_display}")
    else:
        ax.set_title(f"Baseball Trajectory - {park_display}")
    
    ax.set_xlabel("X: 1st Base Line (m)")
    ax.set_ylabel("Y: 3rd Base Line (m)")
    ax.set_zlabel("Height (m)")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def get_park_id_by_name(park_name):
    """
    根據球場名稱獲取球場ID
    
    Args:
        park_name: 球場名稱
    
    Returns:
        球場ID，如果找不到則返回0
    """
    for pid, name in PARK_ID_MAPPING.items():
        if name == park_name:
            return pid
    return 0

def list_available_parks():
    """列出所有可用的球場"""
    load_specific_park_data()
    
    print("=== 可用球場列表 (ID -> 名稱) ===")
    # 按ID排序
    for park_id in sorted(PARK_ID_MAPPING.keys()):
        park_name = PARK_ID_MAPPING[park_id]
        if park_id == 0:
            print(f"ID: {park_id:4d} -> {park_name} (通用球場)")
        elif park_name in park_profiles:
            print(f"ID: {park_id:4d} -> {park_name} (✓ 已載入)")
        else:
            print(f"ID: {park_id:4d} -> {park_name} (✗ 數據未載入)")
    
    return PARK_ID_MAPPING

def check_wall_collision(trajectory_data):
    """
    分析軌跡數據，判定是否撞牆、過牆或普通飛球。
    
    Args:
        trajectory_data: calculate_trajectory 回傳的字典
        
    Returns:
        str: "HR" (過牆), "DOUBLE" (撞牆安打), "In_PLAY" (交由模型判斷)
    """
    park_config = trajectory_data['park_config']
    x, y, z = trajectory_data['x'], trajectory_data['y'], trajectory_data['z']
    
    # 遍歷軌跡點
    for i in range(len(x)):
        # 計算相對於本壘的水平距離 (m)
        curr_dist = math.sqrt(x[i]**2 + y[i]**2)
        
        # 計算當前點的方向角 (0~90度，對齊 ballpark_data 座標系)
        curr_angle = math.degrees(math.atan2(y[i], x[i]))
        
        # 取得該角度下的圍牆物理限制
        wall_dist = get_wall_distance(curr_angle, park_config)
        
        if curr_dist >= wall_dist:
            # 球到達或越過圍牆所在的距離
            wall_height = get_wall_height(curr_angle, park_config)
            ball_height = z[i]
            
            if ball_height > wall_height and ball_height > 2.5:  # 某些場地全壘打牆低，即使球高過牆也可能被接殺
                return "HR"  # 飛行高度高於牆頂
            elif ball_height > 2.5:
                return "DOUBLE"  # 飛行高度高於地面但未過牆
            else:
                return "IN_PLAY"  # 可能被接殺或形成安打候補，交由模型判斷
                
    # 如果球落地(z<0)前都沒超過 wall_dist，代表是場內球
    return "IN_PLAY"

if __name__ == "__main__":
    # 測試腳本
    logger.info("執行 Draw_Utils 測試")
    
    physical_params = {
        'g': config.get('physics', 'g'),
        'rho': config.get('physics', 'rho'),
        'area': config.get('physics', 'area'),
        'm': config.get('physics', 'm'),
        'dt': config.get('physics', 'dt'),
        'hit_pos': tuple(config.get('physics', 'hit_pos'))
    }
    
    # 列出可用球場
    list_available_parks()
    
    # 測試不同球場ID
    test_park_ids = [0, 3, 17, 3313]  # 0=通用, 3=Fenway, 17=Wrigley, 3313=Yankee
    
    for park_id in test_park_ids:
        print(f"\n=== 測試球場 ID: {park_id} ({get_park_name_by_id(park_id)}) ===")
        traj_data = calculate_trajectory(160, 25, 30, physical_params, Cd=0.30, park_id=park_id)
        
        # 可以選擇是否繪圖
        plot_trajectory(traj_data, title=f"Test Trajectory", park_id=park_id)