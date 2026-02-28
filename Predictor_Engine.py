import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import datetime
from Draw_Utils import draw_field, calculate_trajectory, get_park_config, get_park_name_by_id, check_wall_collision
from Config_Loader import config
from Logger_Setup import setup_logger, log_execution_time, ProgressLogger
from Visualization_3D import Baseball3DVisualizer, TrajectoryVideoRecorder


# 設定日誌
logger = setup_logger(__name__)

# 從配置讀取物理參數
PHYSICAL_PARAMS = {
    'g': config.get('physics', 'g'),
    'rho': config.get('physics', 'rho'),
    'area': config.get('physics', 'area'),
    'm': config.get('physics', 'm'),
    'dt': config.get('physics', 'dt'),
    'hit_pos': tuple(config.get('physics', 'hit_pos'))
}

CD_RANGE = config.get('physics', 'cd_range')
CD_ITERATIONS = CD_RANGE['iterations']
CHEAT_CONFIG = config.get('cheat_mode')

class BaseballPredictorEngine:
    @log_execution_time()
    def __init__(self, model_path=None):
        """初始化引擎，載入模型與特徵設定
        
        Args:
            model_path: 模型檔案路徑
            park_id: 預設球場ID (0=通用球場)
        """
        if model_path is None:
            model_path = config.get('model', 'name')
        
        self.park_id = config.get('park', 'default_id', default=0)
        logger.info(f"正在啟動預測引擎 (模型: {model_path}, 球場ID: {self.park_id})")
        
        try:
            # 載入模型 Bundle
            self.data_bundle = joblib.load(model_path)
            
            # 檢查模型中的配置
            if 'config' in self.data_bundle:
                logger.debug("模型包含訓練時的配置資訊")
            
            # 強制使用 CPU 進行推理
            self.clf = self.data_bundle['classifier'].set_params(device="cpu")
            self.reg = self.data_bundle['regressor'].set_params(device="cpu")
            
            # 關鍵修改：區分兩階段特徵
            self.reg_features = self.data_bundle['reg_features']
            self.clf_features = self.data_bundle['clf_features']
            
            self.label_map = self.data_bundle['label_map']
            self.target_names = [k for k, v in sorted(self.label_map.items(), 
                                                       key=lambda item: item[1])]
            
            logger.info(f"引擎啟動成功")
            
        except Exception as e:
            logger.error(f"引擎啟動失敗: {e}", exc_info=True)
            raise

        # 初始化影片錄製器
        self.video_recorder = TrajectoryVideoRecorder(
            output_dir=config.get('video', 'output_dir', default='videos')
        )
        self.visualizer = Baseball3DVisualizer()
        self.cheat_config = CHEAT_CONFIG
    
    @log_execution_time()
    def set_park(self, park_id):
        """設定球場ID
        
        Args:
            park_id: 球場ID (0=通用球場)
        """
        self.park_id = park_id
        park_name = get_park_name_by_id(park_id)
        logger.info(f"球場已切換為: ID={park_id}, 名稱={park_name}")
    
    def _get_bb_type(self, angle):
        """根據仰角自動判定擊球類型"""
        if angle < 10: return 'ground_ball'
        elif 10 <= angle < 25: return 'line_drive'
        elif 25 <= angle < 50: return 'fly_ball'
        else: return 'popup'

    def find_fitted_trajectory(self, v_kmh, angle_deg, spray_deg, target_dist_ft):
        """根據模型預測的距離，反推物理軌跡"""
        target_dist_m = target_dist_ft * 0.3048
        low_cd, high_cd = 0.1, 1.0
        best_traj = None
        
        # 二分搜尋尋找最接近預測距離的 Cd 值
        for _ in range(10):
            mid_cd = (low_cd + high_cd) / 2
            traj = calculate_trajectory(v_kmh, angle_deg, spray_deg, PHYSICAL_PARAMS, 
                                        Cd=mid_cd, park_id=self.park_id)
            sim_dist = np.sqrt(traj['x'][-1]**2 + traj['y'][-1]**2)
            
            if sim_dist > target_dist_m:
                low_cd = mid_cd
            else:
                high_cd = mid_cd
            best_traj = traj
            best_traj['cd'] = mid_cd
            
        return best_traj

    def adaptive_boost(self, value, boost_factor, boost_type='EV'):
        """
        自適應補償增益
        :param value: 原始數值
        :param boost_factor: 增益係數 (1.0 為不變)
        """
        target_threshold = CHEAT_CONFIG['ev_threshold'] if boost_type == 'EV' else CHEAT_CONFIG['dist_threshold']
        max_range = CHEAT_CONFIG['ev_max'] if boost_type == 'EV' else CHEAT_CONFIG['dist_max']

        if boost_factor == 1.0 or value >= target_threshold:
            return value

        # 計算差距比例 (0.0 ~ 1.0)
        gap_ratio = max(0, (target_threshold - value) / target_threshold)
        
        # 補償公式：基本差距 * 增益倍率 * (1 + 距離衰減修正)
        compensation = (target_threshold - value) * (boost_factor - 1) * (1 + gap_ratio)
        new_value = value + compensation
        
        boosted = max(value, min(new_value, max_range))
        logger.debug(f"{boost_type} Boost: {value:.1f} -> {boosted:.1f}")
        
        return boosted
    
    @log_execution_time()
    def run_inference(self, speed_mph, angle_deg, spray_deg, Is_plot=False, Video_save=False,
                  ev_boost=1.0, dist_boost=1.0):
        """執行 AI 串接預測與物理擬合"""
        logger.info(f"推論輸入: 速度={speed_mph:.1f}mph, 仰角={angle_deg:.1f}°, 噴射角={spray_deg:.1f}°")
        
        if ev_boost != 1.0:
            logger.info(f"啟用 EV 補償: {ev_boost}x")
        if dist_boost != 1.0:
            logger.info(f"啟用距離補償: {dist_boost}x")
        
        speed_mph = self.adaptive_boost(speed_mph, ev_boost, boost_type='EV')
        v_kmh = speed_mph * 1.60934
        bb_type = self._get_bb_type(angle_deg)

        bins = [-90, -30, -15, 0, 15, 30, 90]
        labels = ['LF_Ext', 'LF', 'LC', 'RC', 'RF', 'RF_Ext']
        # 判斷當前 spray_deg 屬於哪個區塊
        zone_idx = np.digitize(spray_deg, bins) - 1
        zone_idx = max(0, min(zone_idx, len(labels) - 1)) # 邊界保護
        current_zone = labels[zone_idx]
        current_pkzn = f"{self.park_id}_{current_zone}"
        
        # 1. 第一階段：迴歸預測飛行距離
        # 建立一個字典來存放所有特徵，避免逐次插入欄位
        reg_data = {
            'launch_speed': speed_mph,
            'launch_angle': angle_deg,
            'spray_angle': spray_deg
        }
        
        for feat in self.reg_features:
            if feat.startswith('type_'):
                reg_data[feat] = 1 if f"type_{bb_type}" == feat else 0
            elif feat.startswith('park_'):
                reg_data[feat] = 1 if f"park_{self.park_id}" == feat else 0
        
        reg_df = pd.DataFrame([reg_data])
        pred_dist_ft = float(self.reg.predict(reg_df[self.reg_features])[0])
        
        # 2. 第二階段：分類預測結果
        # 直接在字典中加入預測距離
        clf_data = reg_data.copy()
        clf_data['hit_distance_sc'] = pred_dist_ft
        
        # 批量處理分類器特徵，消除 PerformanceWarning
        for feat in self.clf_features:
            if feat not in clf_data:
                if feat.startswith('pkzn_'):
                    clf_data[feat] = 1 if feat == current_pkzn else 0
                else:
                    clf_data[feat] = 0
        
        clf_df = pd.DataFrame([clf_data])
        
        y_probs = self.clf.predict_proba(clf_df[self.clf_features])[0]
        pred_label = self.target_names[np.argmax(y_probs)]
        
        logger.info(f"分類結果: {pred_label} (機率: {y_probs[np.argmax(y_probs)]:.3f})")
        
        # 3. 物理軌跡擬合
        fitted_traj = self.find_fitted_trajectory(v_kmh, angle_deg, spray_deg, pred_dist_ft)

        if -45 <= spray_deg <= 45:
        # 直接利用算好的軌跡進行判定
            physics_result = check_wall_collision(fitted_traj)
            
            # 只要物理判定認為是長打，就強制覆蓋 AI (AI 常因樣本不平衡誤判為 OUT)
            if physics_result in ["HR", "DOUBLE"]:
                logger.info(f"物理判定覆蓋 AI ({pred_label} -> {physics_result})")
                pred_label = physics_result
                hit_prob = 1.0  # 強制提升命中機率
            else:  
                hit_prob = 1 - y_probs[0]  # OUT 的機率
        else:
            pred_label = "Foul"
            hit_prob = 0


        result = {
            'input_speed': speed_mph,
            'input_angle': angle_deg,
            'input_spray': spray_deg,
            'boosted_speed': speed_mph,
            'pred_dist_ft': pred_dist_ft,
            'result_class': pred_label,
            'hit_prob': hit_prob,
            'cd': fitted_traj['cd'],
            'trajectory': fitted_traj,  # 紀錄軌跡提供影片繪畫使用
            'park_id': self.park_id,
            'bb_type': bb_type,
            'is_foul': not (-45 <= spray_deg <= 45)
        }

        # 4. 展示結果
        if Is_plot:
            self.visualize_result(speed_mph, angle_deg, spray_deg, result)
        
        # 5. 儲存影片（如果需要）- 自動旋轉視角
        if Video_save:
            video_path = self._save_trajectory_video_from_result(
                result, 
                speed_mph, 
                angle_deg, 
                spray_deg
            )
            if video_path:
                logger.info(f"軌跡影片已儲存至: {video_path}")
                result['video_path'] = video_path
        
        # 6. 顯示互動式動畫（如果需要）- 可手動控制視角
        if self.video_recorder.show_animation:
            print("\n顯示互動式動畫（可用滑鼠拖曳旋轉視角）...")
            self.video_recorder.show_trajectory_animation(
                fitted_traj,
                title=f"{result['result_class']}: {speed_mph:.0f}mph, {angle_deg:.0f}°"
            )

        return result

    def visualize_result(self, speed_mph, angle_deg, spray_deg, result):
        """視覺化繪圖 - 支援特定球場"""
        # 獲取球場名稱
        park_id = result['park_id']
        park_name = get_park_name_by_id(park_id)
        park_config = get_park_config(park_id)
        
        fig = plt.figure(figsize=(14, 8))
        
        # --- 左側：數據分析面板 ---
        ax_text = fig.add_subplot(121)
        ax_text.axis('off')

        # 球場資訊
        park_info = f"Stadium: {park_config['name']} (ID: {park_id})"
        if park_id == 0:
            park_info = "Stadium: Generic Park"

        analysis_text = (
            f"MLB HIT ANALYSIS REPORT\n"
            f"{'='*45}\n"
            f"  [ BALLPARK INFORMATION ]\n"
            f"  {park_info}\n\n"
            f"  [ INPUT PARAMETERS ]\n"
            f"  - Launch Speed:   {speed_mph:.1f} mph\n"
            f"  - Launch Angle:   {angle_deg:.1f}°\n"
            f"  - Spray Angle:    {spray_deg:.1f}°\n"
            f"  - BB Type:        {result['bb_type']}\n\n"
            f"  [ ML PREDICTIONS ]\n"
            f"  - Predicted Result: {'FOUL' if result['is_foul'] else result['result_class']}\n"
            f"  - Hit Probability:  {result['hit_prob']:.1%}\n"
            f"  - Pred Distance:    {result['pred_dist_ft']:.1f} ft ({result['pred_dist_ft']*0.3048:.1f} m)\n\n"
            f"  [ PHYSICS ESTIMATION ]\n"
            f"  - Hang Time:      {result['trajectory']['hang_time']:.2f} s\n"
            f"  - Effective Cd:   {result['cd']:.3f}\n"
            f"{'='*45}\n"
            f"  Status: {'!!! FOUL BALL !!!' if result['is_foul'] else 'IN PLAY'}"
        )

        ax_text.text(0.1, 0.5, analysis_text, transform=ax_text.transAxes, 
                    fontsize=11, verticalalignment='center', family='monospace',
                    bbox=dict(facecolor='aliceblue', alpha=0.8, edgecolor='navy', boxstyle='round'))

        # --- 右側：3D 場地與軌跡 ---
        ax_3d = fig.add_subplot(122, projection='3d')
        
        # 使用新的 draw_field 函數，傳入 park_id
        draw_field(ax_3d, park_id)
        
        traj_data = result['trajectory']
        
        # 根據結果設定軌跡顏色
        if result['is_foul']:
            traj_color = 'gray'
        elif result['result_class'] == 'HR':
            traj_color = 'red'
        elif result['result_class'] in ['DOUBLE', 'TRIPLE']:
            traj_color = 'orange'
        else:
            traj_color = 'blue'

        # 繪製軌跡
        ax_3d.plot(traj_data['x'], traj_data['y'], traj_data['z'], 
                   color=traj_color, lw=3, label='ML-Hybrid Path')
        
        # 標記落點
        if len(traj_data['x']) > 0:
            ax_3d.scatter(traj_data['x'][-1], traj_data['y'][-1], 0, 
                         color=traj_color, s=100, marker='o', label='Landing Point')

        # 動態調整顯示範圍
        if park_config['type'] == 'generic':
            max_field_dist = park_config['center_field_m'] * 1.2
        else:
            max_field_dist = np.max(park_config['distances']) * 1.2
        
        max_ball_dist = np.sqrt(np.max(traj_data['x'])**2 + np.max(traj_data['y'])**2)
        max_dist = max(max_field_dist, max_ball_dist * 1.2)
        
        ax_3d.set_xlim(-max_dist*0.2, max_dist)
        ax_3d.set_ylim(-max_dist*0.2, max_dist)
        ax_3d.set_zlim(0, max(np.max(traj_data['z']), 30) * 1.2)
        
        # 設定視角
        ax_3d.view_init(elev=20, azim=-45)
        
        park_display = park_config['name']
        ax_3d.set_title(f"3D Trajectory Simulation - {park_display}")
        ax_3d.set_xlabel("X: 1st Base Line (m)")
        ax_3d.set_ylabel("Y: 3rd Base Line (m)")
        ax_3d.set_zlabel("Height (m)")
        ax_3d.legend()
        
        plt.tight_layout()
        plt.show()

    def _save_trajectory_video_from_result(self, result, speed_mph, angle_deg, spray_deg):
        """
        從已有的 result 物件儲存影片
        
        Args:
            result: run_inference 回傳的結果字典
            speed_mph: 初速 (mph)
            angle_deg: 仰角
            spray_deg: 噴射角
        
        Returns:
            str: 影片儲存路徑，失敗則回傳 None
        """
        try:
            if 'trajectory' not in result:
                logger.error("結果中沒有軌跡資料")
                return None
            
            # 自動產生檔名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_type = result['result_class']
            park_name = get_park_name_by_id(result.get('park_id', 0))
            filename = f"{timestamp}_{park_name}_{result_type}_{speed_mph:.0f}mph.mp4"
            
            # 產生影片（使用配置中的設定）
            video_path = self.video_recorder.record_single_trajectory(
                result['trajectory'],
                filename=filename,
                title=f"{park_name}: {result['result_class']}, {speed_mph:.0f}mph, {angle_deg:.0f}°"
            )
            
            return video_path
            
        except Exception as e:
            logger.error(f"儲存影片時發生錯誤: {e}", exc_info=True)
            return None

    # 保留 save_trajectory_video 方法（用於外部直接呼叫）
    def save_trajectory_video(self, speed_mph, angle_deg, spray_deg, 
                            filename=None, rotation=True):
        """
        儲存單一軌跡的影片（外部呼叫用）
        
        Returns:
            str: 影片儲存路徑，失敗則回傳 None
        """
        logger.info(f"產生軌跡影片: {speed_mph}mph, {angle_deg}°, {spray_deg}°")
        
        # 執行推論取得軌跡
        result = self.run_inference(speed_mph, angle_deg, spray_deg, Is_plot=False)
        
        if 'trajectory' not in result:
            logger.error("無法取得軌跡資料")
            return None
        
        # 自動產生檔名（如果沒指定）
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_type = 'FOUL' if not (-45 <= spray_deg <= 45) else result['result_class']
            park_name = get_park_name_by_id(result.get('park_id', 0))
            filename = f"{timestamp}_{park_name}_{result_type}_{speed_mph:.0f}mph.mp4"
        
        # 產生影片
        video_path = self.video_recorder.record_single_trajectory(
            result['trajectory'],
            filename=filename,
            title=f"{park_name}: {result['result_class']}, {speed_mph}mph, {angle_deg}°",
            rotation=rotation
        )
        
        return video_path
    
if __name__ == "__main__":
    # 測試腳本
    engine = BaseballPredictorEngine('baseball_dual_model.pkl')
    test_sets = 2  # 測試組數量
    print(f"--- 開始進行 {test_sets} 組隨機數據測試 ---")
    for i in range(test_sets):
        speed_mph = random.uniform(50, 110)
        angle_deg = random.uniform(-15, 60) 
        spray_deg = random.uniform(-60, 60) # 包含界外角度

        print(f"\n--- 測試組 {i+1} ---")
        engine.run_inference(speed_mph, angle_deg, spray_deg)


"""
================================================================================
補充：關於 阻力係數 (Drag Coefficient, Cd) 的定義與本專案中的應用
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
        
C. 物理擬合 (Physical Fitting): 
    利用 Binary Search 調整 Cd，直到物理模擬的落地點精準對齊模型預測的距離，
    確保視覺軌跡與大數據分析結果 100% 一致。

3. 數值參考指標:
- Cd < 0.25: 代表該擊球具有極佳的 Carry (可能帶有強大後旋或順風)。
- Cd ≈ 0.30: 符合標準物理環境下的飛行狀態。
- Cd > 0.35: 代表阻力較大 (可能帶有前旋 Topspin 或遭遇逆風)，球下墜較快。
================================================================================
"""