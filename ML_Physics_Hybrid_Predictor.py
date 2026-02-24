from Predictor_Engine import BaseballPredictorEngine
from Logger_Setup import setup_logger, ProgressLogger
import pandas as pd
import argparse
import random
import datetime
import os

# 設定日誌
logger = setup_logger(__name__)

def save_results_to_csv(results, filename_prefix="prediction_results"):
    """將預測結果清單存入 CSV"""
    if not results:
        return
    df = pd.DataFrame(results)
    # 移除軌跡大資料，避免 CSV 肥大，只保留文字結果
    if 'trajectory' in df.columns:
        df = df.drop(columns=['trajectory'])
    
    # 確認 outputs 資料夾存在
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{filename_prefix}_{timestamp}.csv"
    df.to_csv(f"{output_dir}/{filename}", index=False, encoding='utf-8-sig')
    print(f"\n預測紀錄已儲存至: {output_dir}/{filename}")

def batch_process_csv(engine, csv_path, ev_boost=1.0, dist_boost=1.0):
    """批量讀取 CSV 並執行預測"""
    logger.info(f"批量處理模式: {csv_path}")
    
    save_video = input("是否要儲存每筆預測的軌跡影片？(y/n，耗時較長): ").lower() == 'y'
    if save_video:
        print("已啟用步影片儲存功能")
    
    df = pd.read_csv(csv_path)
    
    required_cols = ['launch_speed', 'launch_angle', 'spray_angle']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"CSV 必須包含以下欄位: {required_cols}")
        return
    
    results_list = []
    
    # 使用進度追蹤器
    with ProgressLogger(len(df), logger, log_interval=10, name="批次預測") as progress:
        for idx, row in df.iterrows():
            res = engine.run_inference(
                row['launch_speed'], row['launch_angle'], row['spray_angle'],
                Is_plot=False, 
                Video_save=save_video,
                ev_boost=ev_boost, 
                dist_boost=dist_boost
            )
            results_list.append(res)
            
            progress.update()
    
    # 儲存預測結果
    save_results_to_csv(results_list, "batch_output")

def real_time_input_mode(engine, ev_boost=1.0, dist_boost=1.0):
    """持續接收外部輸入模式"""
    print("--- 即時輸入模式啟動 (輸入 'exit' 退出並存檔) ---")
    
    save_video = input("是否要儲存軌跡影片？(y/n): ").lower() == 'y'
    if save_video:
        print("已啟用影片儲存功能")
    
    results_list = []
    
    try:
        while True:
            val = input("\n請輸入擊球參數 (初速,仰角,方位) 或 'exit': ")
            if val.lower() == 'exit':
                break
            
            try:
                params = [float(x) for x in val.replace(',', ' ').split()]
                if len(params) < 3:
                    print("請輸入三個數值: 初速(mph) 仰角(°) 噴射角(°)")
                    continue
                
                # 即時模式顯示靜態圖，影片背景產生
                res = engine.run_inference(
                    *params[:3], 
                    Is_plot=True,  # 顯示靜態圖
                    Video_save=save_video,
                    ev_boost=ev_boost, 
                    dist_boost=dist_boost
                )
                results_list.append(res)
                    
            except Exception as e:
                print(f"輸入格式錯誤: {e}")
    
    finally:
        # 儲存預測記錄
        if results_list:
            save_results_to_csv(results_list, "realtime_log")
            print(f"\n預測記錄已儲存")

def random_test_mode(engine, ev_boost=1.0, dist_boost=1.0):
    """隨機生成測試數據模式"""
    test_sets = int(input("請輸入要生成的測試數據筆數: "))
    
    # 詢問是否儲存影片（隨機測試通常不需要影片，但還是提供選項）
    save_video = input("是否要儲存軌跡影片？(y/n，建議n): ").lower() == 'y'
    if save_video:
        print("已啟用影片儲存功能")
    
    results_list = []
    for i in range(test_sets):
        speed_mph = random.uniform(50, 110)
        angle_deg = random.uniform(-15, 60) 
        spray_deg = random.uniform(-60, 60) # 包含界外角度

        print(f"\n--- 測試組 {i+1} ---")
        print(f"隨機參數: {speed_mph:.1f}mph, {angle_deg:.1f}°, {spray_deg:.1f}°")
        
        res = engine.run_inference(
            speed_mph, angle_deg, spray_deg,
            Is_plot=True, 
            Video_save=save_video,
            ev_boost=ev_boost, 
            dist_boost=dist_boost
        )
        results_list.append(res)
    
    save_results_to_csv(results_list, "random_test_output")
    print(f"\n隨機測試完成，共 {test_sets} 筆結果已儲存")

if __name__ == "__main__":
    # 初始化引擎 (只載入模型一次，效率最高)
    parser = argparse.ArgumentParser(description="Baseball Predictor Engine with Optional Boosts")
    parser.add_argument('--model', type=str, default="baseball_dual_model.pkl", help='模型檔案路徑')
    parser.add_argument('--ev_boost', type=float, default=1.0, help='EV 補償增益係數 (預設 1.0, 大於 1.0 為增強)')
    parser.add_argument('--dist_boost', type=float, default=1.0, help='距離補償增益係數 (預設 1.0, 大於 1.0 為增強)')
    args = parser.parse_args()
    
    model_path = args.model
    ev_boost = args.ev_boost
    dist_boost = args.dist_boost

    # 啟動引擎
    engine = BaseballPredictorEngine(model_path=model_path)
    
    if ev_boost != 1.0 or dist_boost != 1.0:
        print(f"\n注意: 已啟用補償增益")
        if ev_boost != 1.0:
            print(f"   - EV Boost: {ev_boost}")
        if dist_boost != 1.0:
            print(f"   - Distance Boost: {dist_boost}")
    
    print("\n請選擇模式:")
    print("  (1) 批量 CSV 處理")
    print("  (2) 即時手動輸入")
    print("  (3) 隨機生成測試數據")
    print("-" * 50)
    
    choice = input("選擇 (1-3): ").strip()

    if choice == '1':
        path = input("請輸入 CSV 路徑: ")
        batch_process_csv(engine, path, ev_boost, dist_boost)
    elif choice == '2':
        real_time_input_mode(engine, ev_boost, dist_boost)
    elif choice == '3':
        random_test_mode(engine, ev_boost, dist_boost)
    else:
        print("無效選項，退出。")


    