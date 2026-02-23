import pandas as pd
import numpy as np
import warnings
import pybaseball
from pybaseball import statcast
import argparse
import os
from datetime import datetime
from Config_Loader import config
from Logger_Setup import setup_logger

# 設定日誌
logger = setup_logger(__name__)

# 從配置讀取設定
DATA_DIR = config.get('data', 'data_dir')
CACHE_ENABLED = config.get('data', 'cache_enabled')

if CACHE_ENABLED:
    pybaseball.cache.enable()
    logger.info("已啟用 pybaseball 快取功能")

warnings.filterwarnings("ignore")

def date_check(selected_year, selected_month):
    """檢查輸入的年月是否在 MLB 賽季合理範圍內，並處理未來日期"""
    current_date = datetime.now()
    start_year = config.get('data', 'start_year')
    
    # 基本年份檢查
    if not (start_year <= selected_year <= current_date.year):
        error_msg = f"年份必須在 {start_year} 到 {current_date.year} 之間"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 賽季月份檢查 (3月春訓開始至11月世界大賽結束)
    if not (3 <= selected_month <= 11):
        error_msg = "MLB 賽季數據通常只存在於 3 月至 11 月之間"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    start_date = f"{selected_year}-{selected_month:02d}-01"
    end_dt_obj = pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)
    
    # 若查詢的是本月，結束日期不能超過今天
    if selected_year == current_date.year and selected_month == current_date.month:
        if end_dt_obj > current_date:
            end_dt_obj = current_date
            logger.info(f"查詢當月數據，結束日期調整為今天: {current_date}")
            
    end_date = end_dt_obj.strftime("%Y-%m-%d")
    
    logger.info(f"日期範圍: {start_date} 至 {end_date}")
    return start_date, end_date

def fetch_and_refine_data(start_date, end_date, data_dir=None):
    """
    抓取 MLB Statcast 數據並進行初步清洗與特徵工程，最後存成 CSV。
    """
    if data_dir is None:
        data_dir = DATA_DIR
        
    logger.info(f"啟動場內球數據抓取 ({start_date} 至 {end_date})")
    
    # 建立資料夾
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"已建立資料夾: {data_dir}")
    
    try:
        # 抓取原始資料
        logger.debug("正在呼叫 Statcast API...")
        raw_data = statcast(start_dt=start_date, end_dt=end_date)
        logger.info(f"成功抓取原始數據，筆數: {len(raw_data)}")
    except Exception as e:
        logger.error(f"抓取中斷！錯誤訊息: {e}", exc_info=True)
        return None
    
    if raw_data.empty:
        logger.warning("該時間範圍內沒有數據")
        return None

    # 1. 篩選場內球 (hit_into_play)
    df = raw_data[raw_data['description'] == 'hit_into_play'].copy()
    logger.info(f"篩選後場內球筆數: {len(df)}")
    
    # 2. 移除關鍵參數缺失的資料
    initial_count = len(df)
    df = df.dropna(subset=['launch_speed', 'launch_angle', 'hc_x', 'hc_y']).copy()
    logger.info(f"移除缺失值後筆數: {len(df)} (移除 {initial_count - len(df)} 筆)")
    
    # 3. 座標轉換為 Spray Angle
    df['spray_angle'] = np.degrees(np.arctan((df['hc_x'] - 125.42) / (198.27 - df['hc_y'])))
    logger.debug("已完成 spray angle 計算")

    # 4. 標籤映射
    label_map = config.get('labels')
    
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
    
    # 統計各類別數量
    result_counts = df['result_label'].value_counts()
    logger.info(f"結果類別分布:\n{result_counts}")

    # 5. 選取最終特徵欄位
    cols = ['result_label', 'launch_speed', 'launch_angle', 'spray_angle', 
            'hit_distance_sc', 'bb_type']
    final_df = df[cols]
    
    # 儲存檔案
    file_name = f"ml_data_{start_date}_{end_date}.csv"
    save_path = os.path.join(data_dir, file_name)
    final_df.to_csv(save_path, index=False)
    
    logger.info(f"成功儲存至: {save_path} (樣本數: {len(final_df)})")
    return final_df

def combine_datasets(csv_list, data_dir=None):
    """
    保留函式：合併多個 CSV 檔案。
    """
    if data_dir is None:
        data_dir = DATA_DIR
        
    logger.info(f"開始合併資料集 (共 {len(csv_list)} 個檔案)")
    df_list = []
    for f in csv_list:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            df_list.append(pd.read_csv(path))
            logger.debug(f"已載入: {f}")
        else:
            logger.warning(f"找不到檔案 {path}")
            
    if df_list:
        combined = pd.concat(df_list, axis=0, ignore_index=True)
        logger.info(f"合併完成，總筆數: {len(combined)}")
        return combined
    else:
        logger.error("沒有找到任何可合併的檔案")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Static Data Fetcher")
    parser.add_argument("--year", type=str, required=True, help="Year of Data (2015-2026)")
    parser.add_argument("--month", type=str, required=True, help="Month of Data (3-11)")
    parser.add_argument("--dir", type=str, default=None, help="Data Directory")
    
    args = parser.parse_args()
    
    try:
        start_date, end_date = date_check(int(args.year), int(args.month))
        fetch_and_refine_data(start_date, end_date, args.dir)
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
        raise