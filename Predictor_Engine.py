import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
# 從你之前的模組匯入場地繪製工具
from Draw_Utils import draw_field, calculate_trajectory

Phsical_Params = {
    'g': 9.81,        # 重力加速度 (m/s^
    'rho': 1.225,    # 空氣密度 (kg/m^3)
    'area': 0.00421, # 棒球截面積 (m^2)
    'm': 0.145,       # 棒球質量 (kg)
    'dt': 0.01,        # 模擬時間步長 (秒)
    'hit_pos': (0, 0, 1.0)  # 擊球位置 (x, y, z) (m)
}

class BaseballPredictorEngine:
    def __init__(self, model_path='baseball_dual_model.pkl'):
        """初始化引擎，載入模型與特徵設定"""
        print(f"--- 正在啟動預測引擎 (模型: {model_path}) ---")
        try:
            # 載入模型 Bundle
            self.data_bundle = joblib.load(model_path)
            # 強制使用 CPU 進行推理以提高相容性
            self.clf = self.data_bundle['classifier'].set_params(device="cpu")
            self.reg = self.data_bundle['regressor'].set_params(device="cpu")
            self.features = self.data_bundle['features']
            self.label_map = self.data_bundle['label_map']
            # 建立類別名稱清單
            self.target_names = [k for k, v in sorted(self.label_map.items(), key=lambda item: item[1])]
        except Exception as e:
            print(f"引擎啟動失敗: {e}")
            raise

    def _get_bb_type(self, angle):
        """根據仰角自動判定擊球類型"""
        if angle < 10: return 'ground_ball'
        elif 10 <= angle < 25: return 'line_drive'
        elif 25 <= angle < 50: return 'fly_ball'
        else: return 'popup'

    def find_fitted_trajectory(self, v_kmh, angle_deg, direction_deg, target_dist_m):
        """使用二分搜尋法找到最符合模型預測距離的有效阻力係數 Cd"""
        low_cd, high_cd = 0.1, 1.5
        best_sim = None
        for _ in range(12): # 12 次迭代足以達到高精度
            mid_cd = (low_cd + high_cd) / 2
            sim_data = calculate_trajectory(v_kmh, angle_deg, direction_deg, Phsical_Params, Cd=mid_cd)
            
            if sim_data['distance'] > target_dist_m:
                low_cd = mid_cd # 飛太遠，增加阻力
            else:
                high_cd = mid_cd # 飛太近，減少阻力
            best_sim = sim_data
        return best_sim, mid_cd

    def run_inference(self, speed_mph, angle_deg, spray_deg, show_plot=True):
        """執行ML 預測 + 物理擬合"""
        is_physically_foul = abs(spray_deg) > 45
        bb_type = self._get_bb_type(angle_deg)

        # 1. 構造特徵 DataFrame
        input_row = {
            'launch_speed': speed_mph, # 擊球初速
            'launch_angle': angle_deg, # 擊球仰角
            'spray_angle': spray_deg # 擊球偏角
        }
        # 處理 One-hot 編碼特徵
        for f in self.features:
            if f.startswith('type_'):
                input_row[f] = 1 if f == f"type_{bb_type}" else 0
        
        input_df = pd.DataFrame([input_row])
        # 補齊可能缺失的特徵欄位並排序
        for f in self.features:
            if f not in input_df.columns: input_df[f] = 0
        input_df = input_df[self.features]

        # 2. ML 模型預測
        pred_probs = self.clf.predict_proba(input_df)[0]
        pred_dist_ft = self.reg.predict(input_df)[0]
        
        # 3. 物理軌跡擬合 (將預測距離轉為公尺)
        v_kmh = speed_mph * 1.60934
        direction_deg = 45 + spray_deg # 45度為球場中軸線
        traj, cd = self.find_fitted_trajectory(v_kmh, angle_deg, direction_deg, pred_dist_ft * 0.3048)

        # 4. 封裝結果
        result = {
            "class": self.target_names[np.argmax(pred_probs)],
            "hit_prob": sum(pred_probs[1:]), # 排除 'OUT' 的機率
            "dist_ft": pred_dist_ft,
            "cd": cd,
            "trajectory": traj,
            "bb_type": bb_type,
            "is_foul": is_physically_foul
        }

        if show_plot:
            self.plot_result(speed_mph, angle_deg, spray_deg, result)
        
        return result

    def plot_result(self, speed_mph, angle_deg, spray_deg, result):
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