import pandas as pd
import numpy as np
import warnings
import pybaseball
from pybaseball import statcast
import os  # 新增：用於處理資料夾路徑

# 開啟快取功能
pybaseball.cache.enable()
warnings.filterwarnings("ignore")

from datetime import datetime

def date_check(selected_year, selected_month):
    """檢查輸入的年月是否在 MLB 賽季合理範圍內，並處理未來日期"""
    current_date = datetime.now()
    # 基本年份檢查 (Statcast 始於 2015)
    if not (2015 <= selected_year <= current_date.year):
        raise ValueError(f"年份必須在 2015 到 {current_date.year} 之間")
    
    # 賽季月份檢查 (3月春訓開始至11月世界大賽結束)
    if not (3 <= selected_month <= 11):
        raise ValueError("MLB 賽季數據通常只存在於 3 月至 11 月之間")
    
    start_date = f"{selected_year}-{selected_month:02d}-01"
    end_dt_obj = pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)
    
    # 若查詢的是本月，結束日期不能超過今天
    if selected_year == current_date.year and selected_month == current_date.month:
        if end_dt_obj > current_date:
            end_dt_obj = current_date
            
    end_date = end_dt_obj.strftime("%Y-%m-%d")
    
    return start_date, end_date
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
    selected_year = 2024
    selected_month = 3
    start_date, end_date = date_check(selected_year, selected_month)

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