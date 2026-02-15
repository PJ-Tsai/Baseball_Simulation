import pandas as pd
import numpy as np
import warnings
import pybaseball
from pybaseball import statcast
import os  # 新增：用於處理資料夾路徑

# 開啟快取功能
pybaseball.cache.enable()
warnings.filterwarnings("ignore")

def fetch_and_refine_data(start_date, end_date):
    print(f"--- 啟動場內球數據抓取 ({start_date} 至 {end_date}) ---")
    
    try:
        # 抓取原始資料
        raw_data = statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        print(f"抓取中斷！錯誤訊息: {e}")
        return None
    
    # 1. 篩選場內球 (hit_into_play)
    df = raw_data[raw_data['description'] == 'hit_into_play'].copy()
    
    # 2. 移除關鍵參數缺失的資料
    df = df.dropna(subset=['launch_speed', 'launch_angle', 'hc_x', 'hc_y']).copy()
    
    # 3. 座標轉換為 Spray Angle
    # 原理：利用擊球落點座標 (hc_x, hc_y) 計算相對於本壘的角度
    df['spray_angle'] = np.degrees(np.arctan((df['hc_x'] - 125.42) / (198.27 - df['hc_y'])))

    # 4. 標籤映射 (定義 5 類結果，排除 FOUL)
    def label_result(row):
        ev = row['events']
        if ev == 'home_run': return 'HR'
        if ev == 'single': return 'SINGLE'
        if ev == 'double': return 'DOUBLE'
        if ev == 'triple': return 'TRIPLE'
        # 整合各種出局情況
        if ev in ['field_out', 'force_out', 'sac_fly', 'field_error', 
                  'grounded_into_double_play', 'fielders_choice']: 
            return 'OUT'
        return None

    df['result_label'] = df.apply(label_result, axis=1)
    df = df.dropna(subset=['result_label']).copy()
    
    # 5. 選取最終特徵欄位
    cols = ['result_label', 'launch_speed', 'launch_angle', 'spray_angle', 
            'hit_distance_sc', 'bb_type']
    
    return df[cols]

if __name__ == "__main__":
    start_date = "2025-07-01"
    end_date = "2025-07-31"
    
    # 設定存檔目錄與檔名
    target_dir = "datasets"
    filename = f"ml_data_{start_date}_{end_date}.csv"
    
    # 建立資料夾 (如果不存在)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已建立資料夾: {target_dir}")

    # 執行抓取
    combined_data = fetch_and_refine_data(start_date, end_date)
    
    if combined_data is not None:
        # 結合路徑存檔
        full_path = os.path.join(target_dir, filename)
        combined_data.to_csv(full_path, index=False)
        print(f"成功儲存至 {full_path}")
        
        # 釋放快取空間 (選用)
        # pybaseball.cache.purge()