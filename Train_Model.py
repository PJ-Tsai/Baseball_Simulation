import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib
import warnings

warnings.filterwarnings("ignore")

def train_hit_only_model(data_path, model_name="baseball_dual_model.pkl"):
    # 1. 載入資料 (此時 CSV 已不含 FOUL)
    df = pd.read_csv(data_path)
    
    # 2. 類別映射 (移除 FOUL)
    label_map = {'OUT': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'HR': 4}
    df['target_class'] = df['result_label'].map(label_map)
    
    # 確保沒有殘留的 NaN (以防萬一)
    df = df.dropna(subset=['target_class'])
    
    # 3. 特徵處理
    df = pd.get_dummies(df, columns=['bb_type'], prefix='type')
    features = ['launch_speed', 'launch_angle', 'spray_angle']
    features += [c for c in df.columns if c.startswith('type_')]
    
    # 4. 三相切分 (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[features], df[['target_class', 'hit_distance_sc']], 
        test_size=0.15, random_state=42, stratify=df['target_class']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp['target_class']
    )

    # 5. 模型訓練參數
    param_dist = {
        'n_estimators': [500, 800, 1000], 
        'learning_rate': [0.07, 0.1, 0.01, 0.05],
        'max_depth': [8, 10, 12],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.8],
        'gamma': [0.1, 0.5, 1]    # 新增 gamma：節點分裂所需的最小損失減少量，能有效抑制樹的生長
    }

    common_params = {
        'tree_method': 'hist',
        'device': 'cuda', # 若無 GPU 請改為 'cpu'
        'random_state': 42,
        'early_stopping_rounds': 15
    }

    print(f"--- 啟動擊球落點模型訓練 ---")
    print(f"訓練樣本: {len(X_train)} | 分類目標: 5 類 (無界外)")

    # --- 模型 A: 分類 (5-Class) ---
    xgb_clf = xgb.XGBClassifier(**common_params, objective='multi:softprob')
    rs_clf = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, 
                                n_iter=10, cv=3, scoring='f1_weighted')
    
    rs_clf.fit(X_train, y_train['target_class'], 
               eval_set=[(X_val, y_val['target_class'])], 
               verbose=False)

    # --- 模型 B: 迴歸 (距離) ---
    print("正在訓練飛行距離迴歸模型...")
    
    # 建立遮罩，排除 hit_distance_sc 為 NaN 的資料
    reg_mask = y_train['hit_distance_sc'].notna()
    X_train_reg = X_train[reg_mask]
    y_train_reg = y_train['hit_distance_sc'][reg_mask]
    
    val_reg_mask = y_val['hit_distance_sc'].notna()
    X_val_reg = X_val[val_reg_mask]
    y_val_reg = y_val['hit_distance_sc'][val_reg_mask]

    xgb_reg = xgb.XGBRegressor(**common_params)
    rs_reg = RandomizedSearchCV(
        xgb_reg, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error'
    )
    
    # 使用過濾後的數據進行訓練
    rs_reg.fit(
        X_train_reg, y_train_reg,
        eval_set=[(X_val_reg, y_val_reg)], 
        verbose=False
    )

    # 6. 儲存模型
    model_data = {
        'classifier': rs_clf.best_estimator_,
        'regressor': rs_reg.best_estimator_,
        'features': features,
        'label_map': label_map,
        'test_data': (X_test, y_test) 
    }
    joblib.dump(model_data, model_name)
    print(f"訓練完畢！模型已儲存至 {model_name}")

if __name__ == "__main__":
    data_dir = "datasets"
    train_hit_only_model(f'{data_dir}/ml_data_2025-07-01_2025-07-31.csv')