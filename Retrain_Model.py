import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

def continue_training_hit_only(new_csv_path, model_name='baseball_dual_model.pkl'):
    # 1. 載入現有模型 Bundle
    try:
        bundle = joblib.load(model_name)
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 {model_name}")
        return

    old_clf = bundle['classifier']
    old_reg = bundle['regressor']
    features = bundle['features']
    
    # 2. 標籤映射 (關鍵修改：移除 FOUL，縮減為 5 類)
    # 這樣模型會專注於預測場內球的品質
    label_map = {'OUT': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'HR': 4}
    
    # 3. 讀取並處理新資料
    df_new = pd.read_csv(new_csv_path)
    
    # 過濾標籤：只保留 label_map 中有的類別 (會自動排除 result_label 為 'FOUL' 的列)
    df_new['target_class'] = df_new['result_label'].map(label_map)
    df_new = df_new.dropna(subset=['target_class'])
    
    # 基礎前處理
    df_new['spray_angle'] = df_new['spray_angle'].fillna(999)
    
    # One-hot 編碼與特徵對齊
    df_new = pd.get_dummies(df_new, columns=['bb_type'], prefix='type')
    for col in features:
        if col not in df_new.columns:
            df_new[col] = 0
            
    # 4. 資料切分 (80% 訓練 / 20% 驗證)
    # 使用 stratify 確保各個結果類別比例在訓練與驗證集中一致
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
        df_new[features], 
        df_new[['target_class', 'hit_distance_sc']], 
        test_size=0.20, 
        random_state=42, 
        stratify=df_new['target_class']
    )

    # 5. 配置增量訓練參數
    # 使用較小的學習率，並開啟 Early Stopping
    incremental_params = {
        'learning_rate': 0.05, 
        'early_stopping_rounds': 10,
        'device': 'cuda' if old_clf.get_params().get('device') == 'cuda' else 'cpu'
    }

    # --- 分類模型增量訓練 (5-Class) ---
    print(f"正在接續訓練分類模型 (樣本數: {len(X_train_new)}, 類別: 5)...")
    old_clf.set_params(**incremental_params)
    old_clf.fit(
        X_train_new, y_train_new['target_class'],
        xgb_model=old_clf.get_booster(),
        eval_set=[(X_val_new, y_val_new['target_class'])],
        verbose=False
    )

    # --- 迴歸模型增量訓練 ---
    print("正在接續訓練迴歸模型...")
    reg_mask = y_train_new['hit_distance_sc'].notna()
    val_reg_mask = y_val_new['hit_distance_sc'].notna()
    
    if reg_mask.any() and val_reg_mask.any():
        old_reg.set_params(**incremental_params)
        old_reg.fit(
            X_train_new[reg_mask], y_train_new['hit_distance_sc'][reg_mask],
            xgb_model=old_reg.get_booster(),
            eval_set=[(X_val_new[val_reg_mask], y_val_new['hit_distance_sc'][val_reg_mask])],
            verbose=False
        )

    # 6. 更新 Bundle 資訊
    bundle['classifier'] = old_clf
    bundle['regressor'] = old_reg
    bundle['label_map'] = label_map # 更新標籤映射表
    
    # 儲存模型
    joblib.dump(bundle, model_name)
    print(f"增量訓練完成！模型更新至 {model_name}")

if __name__ == "__main__":
    # 使用新抓取的數據檔進行更新
    data_dir = "datasets"
    continue_training_hit_only(f'{data_dir}/ml_data_2025-04-01_2025-04-30.csv')