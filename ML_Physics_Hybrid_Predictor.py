from Predictor_Engine import BaseballPredictorEngine
import pandas as pd
import argparse
import re
import random

def batch_process_csv(engine, csv_path, ev_boost=1.0, dist_boost=1.0):
    """批量讀取 CSV 並執行預測"""
    print(f"--- 批量處理模式: {csv_path} ---")
    df = pd.read_csv(csv_path)
    # 假設 CSV 包含 launch_speed, launch_angle, spray_angle
    # Format Check
    required_cols = ['launch_speed', 'launch_angle', 'spray_angle']
    if not all(col in df.columns for col in required_cols):
        print(f"錯誤：CSV 必須包含以下欄位: {required_cols}")
        return
    
    for idx, row in df.iterrows():
        print(f"處理第 {idx+1} 筆數據...")
        engine.run_inference(row['launch_speed'], row['launch_angle'], row['spray_angle'], show_plot=False, ev_boost=ev_boost, dist_boost=dist_boost)
    print("批量處理完成。")

def real_time_input_mode(engine, ev_boost=1.0, dist_boost=1.0):
    """持續接收外部輸入模式"""
    print("--- 即時輸入模式啟動 (輸入 'exit' 退出) ---")
    while True:
        try:
            val = input("請輸入擊球參數 (初速,仰角,噴射角) 例如 105,25,10 : ")
            if val.lower() == 'exit': break
            
            s, a, d = map(float, re.split(r'[,\s]+', val.strip())) # 支持逗號或空格分隔
            engine.run_inference(s, a, d, ev_boost=1.0, dist_boost=1.0) # 預設不開啟 Cheat Mode
            # Test code start
            if ev_boost != 1.0 or dist_boost != 1.0:
                print("補償增益後數據")
                engine.run_inference(s, a, d, ev_boost=ev_boost, dist_boost=dist_boost) # 補償增益後預測
            # Test code end

        except ValueError:
            print("格式錯誤，請確保輸入為三個數字並以逗號分隔。")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # 初始化引擎 (只載入模型一次，效率最高)
    # Get ev_boost and dist_boost from script
    ev_boost = 1.0
    dist_boost = 1.0
    parser = argparse.ArgumentParser(description="Baseball Predictor Engine with Optional Boosts")
    parser.add_argument('--model', type=str, default="baseball_dual_model.pkl", help='模型檔案路徑')
    parser.add_argument('--ev_boost', type=float, default=1.0, help='EV 補償增益係數 (預設 1.0, 大於 1.0 為增強)')
    parser.add_argument('--dist_boost', type=float, default=1.0, help='距離補償增益係數 (預設 1.0, 大於 1.0 為增強)')
    args = parser.parse_args()
    model_path = args.model
    ev_boost = args.ev_boost
    dist_boost = args.dist_boost

    engine = BaseballPredictorEngine(model_path=model_path)
    if(ev_boost != 1.0 or dist_boost != 1.0):
            print(f"注意: 已啟用補償增益 (EV Boost: {ev_boost}, Distance Boost: {dist_boost})")

    print("請選擇模式: (1) 批量 CSV 處理  (2) 即時手動輸入  (3) 隨機生成 n 筆測試數據並預測")
    choice = input("選擇: ")

    if choice == '1':
        path = input("請輸入 CSV 路徑: ")
        batch_process_csv(engine, path, ev_boost, dist_boost)
    elif choice == '2':
        real_time_input_mode(engine, ev_boost, dist_boost)
    elif choice == '3':
        test_sets = int(input("請輸入要生成的測試數據筆數: "))
        for i in range(test_sets):
            speed_mph = random.uniform(50, 110)
            angle_deg = random.uniform(-15, 60) 
            spray_deg = random.uniform(-60, 60) # 包含界外角度

            print(f"\n--- 測試組 {i+1} ---")
            engine.run_inference(speed_mph, angle_deg, spray_deg, ev_boost=ev_boost, dist_boost=dist_boost)
    else:
        print("無效選項，退出。")


    