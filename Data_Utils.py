import pandas as pd
import numpy as np
import warnings
import pybaseball
from pybaseball import statcast
import argparse
import os

# 開啟快取功能
pybaseball.cache.enable()
warnings.filterwarnings("ignore")

def fetch_and_refine_data(start_date, end_date, data_dir="datasets"):
    """
    抓取 MLB Statcast 數據並進行初步清洗與特徵工程，最後存成 CSV。
    """
    print(f"--- 啟動場內球數據抓取 ({start_date} 至 {end_date}) ---")
    
    # 建立資料夾
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"已建立資料夾: {data_dir}")
    
    try:
        # 抓取原始資料
        raw_data = statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        print(f"抓取中斷！錯誤訊息: {e}")
        return None
    
    if raw_data.empty:
        print("警告：該時間範圍內沒有數據。")
        return None

    # 1. 篩選場內球 (hit_into_play)
    df = raw_data[raw_data['description'] == 'hit_into_play'].copy()
    
    # 2. 移除關鍵參數缺失的資料
    df = df.dropna(subset=['launch_speed', 'launch_angle', 'hc_x', 'hc_y']).copy()
    
    # 3. 座標轉換為 Spray Angle
    df['spray_angle'] = np.degrees(np.arctan((df['hc_x'] - 125.42) / (198.27 - df['hc_y'])))

    # 4. 標籤映射 (排除 FOUL)
    def label_result(row):
        ev = row['events']
        if ev == 'home_run': return 'HR'
        if ev == 'single': return 'SINGLE'
        if ev == 'double': return 'DOUBLE'
        if ev == 'triple': return 'TRIPLE'
        if ev in ['field_out', 'force_out', 'sac_fly', 'field_error', 
                  'grounded_into_double_play', 'fielders_choice']: 
            return 'OUT'
        return None

    df['result_label'] = df.apply(label_result, axis=1)
    df = df.dropna(subset=['result_label']).copy()
    
    # 5. 選取最終特徵欄位
    cols = ['result_label', 'launch_speed', 'launch_angle', 'spray_angle', 
            'hit_distance_sc', 'bb_type']
    final_df = df[cols]
    
    # 儲存檔案
    file_name = f"ml_data_{start_date}_{end_date}.csv"
    save_path = os.path.join(data_dir, file_name)
    final_df.to_csv(save_path, index=False)
    
    print(f"成功儲存至: {save_path} (樣本數: {len(final_df)})")
    return final_df

def combine_datasets(csv_list, data_dir="datasets"):
    """
    保留函式：合併多個 CSV 檔案。
    """
    print(f"--- 開始合併資料集 (共 {len(csv_list)} 個檔案) ---")
    df_list = []
    for f in csv_list:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            df_list.append(pd.read_csv(path))
        else:
            print(f"警告：找不到檔案 {path}")
            
    return pd.concat(df_list, axis=0, ignore_index=True) if df_list else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Static Data Fetcher")
    parser.add_argument("--start", type=str, required=True, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End Date (YYYY-MM-DD)")
    parser.add_argument("--dir", type=str, default="datasets", help="Data Directory")
    
    args = parser.parse_args()
    fetch_and_refine_data(args.start, args.end, args.dir)