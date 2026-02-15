import pandas as pd
import os

def clean_and_overwrite_csv(file_path):
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return

    # 1. 讀取資料
    df = pd.read_csv(file_path)
    original_len = len(df)
    
    # 2. 移除界外球 (FOUL)
    # 我們只保留 'OUT', 'SINGLE', 'DOUBLE', 'TRIPLE', 'HR'
    df_cleaned = df[df['result_label'] != 'FOUL'].copy()
    
    # 3. 額外檢查：移除場內球但不合理缺失的數據 (如缺少 spray_angle)
    df_cleaned = df_cleaned.dropna(subset=['spray_angle', 'hit_distance_sc'])
    
    # 4. 存回原始路徑 (覆蓋)
    df_cleaned.to_csv(file_path, index=False)
    
    print(f"--- 數據清洗完成 ---")
    print(f"原始樣本數: {original_len}")
    print(f"清理後樣本數: {len(df_cleaned)}")
    print(f"已成功覆寫至: {file_path}")

if __name__ == "__main__":
    # 請在此輸入你當前的資料檔名
    target_file = 'ml_data_2024-04-01_2024-05-31.csv'
    clean_and_overwrite_csv(target_file)