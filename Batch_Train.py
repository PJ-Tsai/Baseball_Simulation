import pandas as pd
import os
from Train_Model import train_hit_only_model  # 直接匯入你現有的函式

def train_with_multiple_datasets(csv_list, output_model_name="baseball_dual_model.pkl"):
    """
    一次合併多個 CSV 檔案並進行完整重新訓練
    :param csv_list: 包含所有資料路徑的清單，例如 ['data1.csv', 'data2.csv']
    :param output_model_name: 最終儲存的模型名稱
    """
    temp_combined_csv = "temp_combined_data.csv"
    
    print(f"--- 開始合併資料集 (共 {len(csv_list)} 個檔案) ---")
    
    df_list = []
    for file in csv_list:
        data_dir = "datasets"
        file = f'{data_dir}/{file}'
        if os.path.exists(file):
            print(f"讀取中: {file}")
            df_list.append(pd.read_csv(file))
        else:
            print(f"警告：找不到檔案 {file}，已跳過")

    if not df_list:
        print("錯誤：沒有可用的資料進行訓練")
        return

    # 1. 合併所有 DataFrames
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    # 2. 存成臨時檔案，供原有訓練函式讀取
    combined_df.to_csv(temp_combined_csv, index=False)
    print(f"合併完成，總樣本數: {len(combined_df)}")

    # 3. 呼叫你原有的訓練函式
    # 這會自動執行特徵處理、三相切分與 RandomizedSearchCV
    try:
        train_hit_only_model(temp_combined_csv, model_name=output_model_name)
        print(f"--- 模型更新成功：{output_model_name} ---")
    finally:
        # 訓練完畢後刪除臨時檔案，保持資料夾乾淨
        if os.path.exists(temp_combined_csv):
            os.remove(temp_combined_csv)

if __name__ == "__main__":
    # 這裡放入你所有的資料集路徑
    data_files = [
        'ml_data_2024-04-01_2024-05-31.csv',
        'ml_data_2024-06-01_2024-06-30.csv',
        'ml_data_2024-07-01_2024-07-31.csv',
        'ml_data_2024-08-01_2024-08-31.csv', 
        'ml_data_2024-09-01_2024-09-30.csv',
        'ml_data_2024-10-01_2024-11-02.csv',
        'ml_data_2025-04-01_2025-04-30.csv',
        'ml_data_2025-05-01_2025-05-31.csv',
        'ml_data_2025-06-01_2025-06-30.csv',
        'ml_data_2025-07-01_2025-07-31.csv'
    ]
    
    train_with_multiple_datasets(data_files)