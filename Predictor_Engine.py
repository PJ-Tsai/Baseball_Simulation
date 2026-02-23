import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
# 從你之前的模組匯入場地繪製工具
from Draw_Utils import draw_field, calculate_trajectory
from Config_Loader import config
from Logger_Setup import setup_logger, log_execution_time, ProgressLogger

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
        """初始化引擎，載入模型與特徵設定"""
        if model_path is None:
            model_path = config.get('model', 'name')
        
        logger.info(f"正在啟動預測引擎 (模型: {model_path})")
        
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
    
    @log_execution_time()

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
            traj = calculate_trajectory(v_kmh, angle_deg, spray_deg, PHYSICAL_PARAMS, Cd=mid_cd)
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

    def run_inference(self, speed_mph, angle_deg, spray_deg, Is_plot=False, 
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
        
        # 1. 第一階段：迴歸預測飛行距離
        reg_df = pd.DataFrame([{
            'launch_speed': speed_mph,
            'launch_angle': angle_deg,
            'spray_angle': spray_deg
        }])
        
        # 自動補齊 One-hot 欄位
        for feat in self.reg_features:
            if feat.startswith('type_'):
                reg_df[feat] = 1 if f"type_{bb_type}" == feat else 0
        
        pred_dist_ft = float(self.reg.predict(reg_df[self.reg_features])[0])
        logger.debug(f"迴歸預測距離: {pred_dist_ft:.1f}ft")
        
        pred_dist_ft = self.adaptive_boost(pred_dist_ft, dist_boost, boost_type='DIST')

        # 2. 第二階段：分類預測結果
        clf_df = reg_df.copy()
        clf_df['hit_distance_sc'] = pred_dist_ft
        for feat in self.clf_features:
            if feat not in clf_df.columns: 
                clf_df[feat] = 0
            
        y_probs = self.clf.predict_proba(clf_df[self.clf_features])[0]
        pred_label = self.target_names[np.argmax(y_probs)]
        
        logger.info(f"分類結果: {pred_label} (機率: {y_probs[np.argmax(y_probs)]:.3f})")
        
        # 3. 物理軌跡擬合
        fitted_traj = self.find_fitted_trajectory(v_kmh, angle_deg, spray_deg, pred_dist_ft)

        result = {
            'input_speed': speed_mph,
            'input_angle': angle_deg,
            'input_spray': spray_deg,
            'boosted_speed': speed_mph,
            'pred_dist_ft': pred_dist_ft,
            'result_class': pred_label,
            'hit_prob': 1 - y_probs[0],
            'cd': fitted_traj['cd'],
        }

        # 4. 展示結果
        if Is_plot:
            self.visualize_result(speed_mph, angle_deg, spray_deg, {
                'class': pred_label,
                'hit_prob': 1 - y_probs[0],
                'dist_ft': pred_dist_ft,
                'trajectory': fitted_traj,
                'cd': fitted_traj['cd'],
                'bb_type': bb_type,
                'is_foul': not (-45 <= spray_deg <= 45)
            })
        
        return result # 回傳結果

    def visualize_result(self, speed_mph, angle_deg, spray_deg, result):
        """視覺化繪圖"""
        fig = plt.figure(figsize=(12, 7))

        # --- 左側：數據分析面板 ---
        ax_text = fig.add_subplot(121)
        ax_text.axis('off')

        analysis_text = (
            f"MLB HIT ANALYSIS REPORT\n"
            f"{'='*35}\n"
            f"  [ INPUT PARAMETERS ]\n"
            f"  - Launch Speed:   {speed_mph:.1f} mph\n"
            f"  - Launch Angle:   {angle_deg:.1f}°\n"
            f"  - Spray Angle:    {spray_deg:.1f}°\n"
            f"  - BB Type:        {result['bb_type']}\n\n"
            f"  [ ML PREDICTIONS ]\n"
            f"  - Predicted Result: {result['class']}\n"
            f"  - Hit Probability:  {result['hit_prob']:.1%}\n"
            f"  - Pred Distance:    {result['dist_ft']:.1f} ft\n\n"
            f"  [ PHYSICS ESTIMATION ]\n"
            f"  - Hang Time:      {result['trajectory']['hang_time']:.2f} s\n"
            f"  - Effective Cd:   {result['cd']:.3f}\n"
            f"{'='*35}\n"
            f"  Status: {'!!! FOUL BALL !!!' if result['is_foul'] else 'IN PLAY'}"
        )

        ax_text.text(0.1, 0.5, analysis_text, transform=ax_text.transAxes, 
                    fontsize=12, verticalalignment='center', family='monospace',
                    bbox=dict(facecolor='aliceblue', alpha=0.8, edgecolor='navy', boxstyle='round'))

        # --- 右側：3D 場地與軌跡 ---
        ax_3d = fig.add_subplot(122, projection='3d')
        draw_field(ax_3d) # 呼叫 Draw_Utils 中的場地繪製
        
        traj_data = result['trajectory']
        traj_color = 'red' if result['class'] == 'HR' else 'blue'
        if result['is_foul']: traj_color = 'gray'

        ax_3d.plot(traj_data['x'], traj_data['y'], traj_data['z'], 
                   color=traj_color, lw=3, label='ML-Hybrid Path')

        # 設定視角與範圍
        ax_3d.set_zlim(0, 50)
        ax_3d.set_xlim(-20, 140)
        ax_3d.set_ylim(-20, 140)
        ax_3d.view_init(elev=20, azim=-45)
        ax_3d.set_title("3D Trajectory Simulation")
        
        plt.tight_layout()
        plt.show()

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