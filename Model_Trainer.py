import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

LABEL_MAP = {'OUT': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'HR': 4} # one-hot 編碼的類別映射
DATA_DIR = "datasets" # 預設數據資料夾
"""
Model Trainer 模組說明：
1. 功能 A: 全量訓練 (train_full)
    - 輸入：一個或多個 CSV 檔案，包含 Statcast 數據。
    - 流程：合併數據、預處理、三相切分 (訓練/驗證/測試)、訓練 XGBoost 分類器與迴歸器、儲存模型 Bundle。
2. 功能 B: 增量訓練 (train_incremental)
    - 輸入：一個新的 CSV 檔案，和已存在的模型檔案。
    - 流程：載入現有模型 Bundle、預處理新數據、切分訓練/驗證集、接續訓練分類器與迴歸器（包含 eval_set）、儲存更新後的模型 Bundle。

全局設定：
    - LABEL_MAP：將擊球結果映射為數字類別。
    - DATA_DIR：預設的數據存放資料夾。 

預處理函式 preprocess_data：
    - 將 result_label 映射為 target_class。
    - 填補 spray_angle 的缺失值為 999。
    - 對 bb_type 進行 one-hot encoding。
模型訓練參數 get_common_params：
    - 定義 XGBoost 模型的共用參數，如 tree_method、device、random_state 和 early_stopping_rounds。

訓練流程：
- train_full：從零開始訓練或合併多檔數
據，包含超參數搜尋。
- train_incremental：接續現有模型訓練，包含 eval_set 以確保增量訓練的穩定性。
"""

def preprocess_data(df, features_list=None):
    """統一的數據預處理邏輯"""
    df['target_class'] = df['result_label'].map(LABEL_MAP)
    df = df.dropna(subset=['target_class']).copy()
    df['spray_angle'] = df['spray_angle'].fillna(999)
    df = pd.get_dummies(df, columns=['bb_type'], prefix='type')
    
    if features_list is None:
        features = ['launch_speed', 'launch_angle', 'spray_angle']
        features += [c for c in df.columns if c.startswith('type_')]
    else:
        features = features_list
        for col in features:
            if col not in df.columns: df[col] = 0
    return df, features

def get_common_params():
    # 模型訓練的共用參數
    return {
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
        'early_stopping_rounds': 15
    }

# --- 功能 A: 全量訓練 (從零開始或合併多檔) ---
def train_full(csv_files, model_name):
    print(f"--- 啟動全量訓練 (檔案數: {len(csv_files)}) ---")
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df_list.append(df)
            else:
                print(f"警告: 檔案 {f} 是空的，已跳過。")
        except Exception as e:
            print(f"錯誤: 無法讀取檔案 {f}。原因: {e}")

    # --- 檢查點 ---
    if not df_list:
        print("df_list 是空的，沒有資料可以合併！請檢查檔案路徑。")
        return 

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    df, features = preprocess_data(combined_df)
    
    # 三相切分
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[features], df[['target_class', 'hit_distance_sc']], 
        test_size=0.15, random_state=42, stratify=df['target_class']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp['target_class']
    )

    # 模型訓練參數
    param_dist = {
        'n_estimators': [500, 800, 1000], 
        'learning_rate': [0.07, 0.1, 0.01, 0.05],
        'max_depth': [8, 10, 12],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.8],
        'gamma': [0.1, 0.5, 1]    # 新增 gamma：節點分裂所需的最小損失減少量，能有效抑制樹的生長
    }

    print(f"--- 啟動擊球落點模型訓練 ---")
    print(f"訓練樣本: {len(X_train)} | 分類目標: 5 類 (無界外)")

    print("正在訓練擊球類型分類模型...")
    # 訓練分類器
    xgb_clf = xgb.XGBClassifier(**get_common_params(), objective='multi:softprob')
    rs_clf = RandomizedSearchCV(xgb_clf, param_dist, n_iter=5, cv=3, scoring='f1_weighted')
    rs_clf.fit(X_train, y_train['target_class'], eval_set=[(X_val, y_val['target_class'])], verbose=False)

    # 訓練迴歸器
    print("正在訓練飛行距離迴歸模型...")
    reg_mask = y_train['hit_distance_sc'].notna()
    val_reg_mask = y_val['hit_distance_sc'].notna()
    xgb_reg = xgb.XGBRegressor(**get_common_params())
    rs_reg = RandomizedSearchCV(xgb_reg, param_dist, n_iter=5, cv=3, scoring='neg_mean_absolute_error')
    rs_reg.fit(X_train[reg_mask], y_train['hit_distance_sc'][reg_mask], 
               eval_set=[(X_val[val_reg_mask], y_val['hit_distance_sc'][val_reg_mask])], verbose=False)

    bundle = {
        'classifier': rs_clf.best_estimator_, 'regressor': rs_reg.best_estimator_,
        'features': features, 'label_map': LABEL_MAP, 'test_data': (X_test, y_test)
    }
    joblib.dump(bundle, model_name)
    print(f"成功儲存全新模型: {model_name}")

# --- 功能 B: 增量訓練 (接續現有模型) ---
def train_incremental(csv_file, model_name):
    if os.path.exists(csv_file):
        target_path = csv_file
    else:
        target_path = os.path.join(DATA_DIR, csv_file)

    if not os.path.exists(target_path):
        print(f"錯誤: 找不到數據檔案 '{target_path}'")
        return

    print(f"--- 啟動增量訓練: {target_path} ---")
    bundle = joblib.load(model_name)
    
    # 預處理新數據
    df_raw = pd.read_csv(target_path)
    df_new, _ = preprocess_data(df_raw, bundle['features'])
    
    # 分割訓練與驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        df_new[bundle['features']], 
        df_new[['target_class', 'hit_distance_sc']], 
        test_size=0.2, 
        random_state=42, 
        stratify=df_new['target_class']
    )

    inc_params = {'learning_rate': 0.01, 'early_stopping_rounds': 10}
    
    # 1. 接續分類器訓練 (已包含 eval_set)
    print("正在更新分類模型...")
    bundle['classifier'].set_params(**inc_params)
    bundle['classifier'].fit(
        X_train, y_train['target_class'], 
        xgb_model=bundle['classifier'].get_booster(),
        eval_set=[(X_val, y_val['target_class'])], 
        verbose=False
    )
    
    # 2. 接續迴歸器訓練 (修正重點：加入 eval_set)
    print("正在更新迴歸模型...")
    reg_mask = y_train['hit_distance_sc'].notna()
    val_reg_mask = y_val['hit_distance_sc'].notna()
    
    if reg_mask.any() and val_reg_mask.any():
        bundle['regressor'].set_params(**inc_params)
        bundle['regressor'].fit(
            X_train[reg_mask], y_train['hit_distance_sc'][reg_mask], 
            xgb_model=bundle['regressor'].get_booster(),
            eval_set=[(X_val[val_reg_mask], y_val['hit_distance_sc'][val_reg_mask])], # 補上這行
            verbose=False
        )
    else:
        print("警告：新數據中缺乏有效的距離資料，跳過迴歸器更新。")

    # 儲存更新後的 Bundle
    joblib.dump(bundle, model_name)
    print(f"模型更新成功: {model_name}")

if __name__ == "__main__":
    """
    Usage:
    1. Full data training => mode: full, files: one or more csv, model: output model name
       Example: python Model_Trainer.py --mode full --files data1.csv data2.csv
    2. Incremental training => mode: inc, files: one csv, model: existing model name
       Example: python Model_Trainer.py --mode inc --files new_data.csv --model existing_model.pkl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['full', 'inc'], required=True) 
    parser.add_argument("--files", nargs='+', required=True)
    parser.add_argument("--model", default="baseball_dual_model.pkl")
    args = parser.parse_args()

    if args.mode == 'full':
        train_full(args.files, args.model)
    else:
        train_incremental(args.files[0], args.model)