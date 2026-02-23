import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
import joblib
import os
import argparse
import warnings
from Config_Loader import config
from Logger_Setup import setup_logger

warnings.filterwarnings("ignore")

# Set up logger
logger = setup_logger(__name__)

# Get global configurations from config.yaml
LABEL_MAP = config.get('labels')
DATA_DIR = config.get('data', 'data_dir')
MODEL_CONFIG = config.get('model')
TRAIN_CONFIG = MODEL_CONFIG['training']
PARAM_DIST = MODEL_CONFIG['param_dist']
INC_CONFIG = MODEL_CONFIG['incremental']

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

def get_base_params(is_regressor=True):
    """根據 config 返回訓練參數"""
    params = {
        'tree_method': TRAIN_CONFIG['tree_method'],
        'device': TRAIN_CONFIG['device'],
        'random_state': MODEL_CONFIG['random_state'],
        'early_stopping_rounds': TRAIN_CONFIG['early_stopping_rounds']
    }
    if not is_regressor:
        params['objective'] = 'multi:softprob'
        params['num_class'] = len(LABEL_MAP)
    return params

def align_and_preprocess(df, reg_feat=None, clf_feat=None):
    """統一預處理與特徵對齊"""
    df = df.copy()
    df['target_class'] = df['result_label'].map(LABEL_MAP)
    df = df.dropna(subset=['target_class', 'hit_distance_sc'])
    
    # One-hot 擊球類型
    df = pd.get_dummies(df, columns=['bb_type'], prefix='type')
    
    if reg_feat is None:
        reg_feat = ['launch_speed', 'launch_angle', 'spray_angle']
        reg_feat += [c for c in df.columns if c.startswith('type_')]
    
    if clf_feat is None:
        clf_feat = reg_feat + ['hit_distance_sc'] # 串接：分類器特徵包含距離

    # 增量訓練關鍵：補齊缺失特徵
    for col in clf_feat:
        if col not in df.columns:
            df[col] = 0
            
    return df, reg_feat, clf_feat

def train_full(csv_files, model_name):
    logger.info("=== 啟動全量訓練 ===")
    logger.info(f"輸入檔案: {csv_files}")
    logger.info(f"輸出模型: {model_name}")
    
    df_list = []
    for f in csv_files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df_list.append(pd.read_csv(path))
            logger.debug(f"成功載入: {path}")
        else:
            logger.error(f"找不到檔案: {path}")

    if not df_list:
        logger.error("完全讀取不到任何資料，請檢查路徑與檔名！")
        return
    
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    logger.info(f"合併後資料筆數: {len(full_df)}")
    
    df, reg_feat, clf_feat = align_and_preprocess(full_df)
    
    # 使用配置中的切分比例
    test_size = MODEL_CONFIG['test_size']
    val_size = MODEL_CONFIG['val_size']
    
    train_df, temp_df = train_test_split(df, test_size=test_size, 
                                         random_state=MODEL_CONFIG['random_state'])
    val_df, test_df = train_test_split(temp_df, test_size=val_size, 
                                       random_state=MODEL_CONFIG['random_state'])
    
    logger.info(f"訓練集: {len(train_df)}, 驗證集: {len(val_df)}, 測試集: {len(test_df)}")

    # 訓練迴歸器
    logger.info("Step 1: 尋找距離迴歸器最佳參數...")
    base_reg = xgb.XGBRegressor(**get_base_params(True))
    rs_reg = RandomizedSearchCV(base_reg, PARAM_DIST, n_iter=5, cv=3, 
                                scoring='neg_mean_absolute_error')
    rs_reg.fit(train_df[reg_feat], train_df['hit_distance_sc'],
               eval_set=[(val_df[reg_feat], val_df['hit_distance_sc'])], verbose=False)
    regressor = rs_reg.best_estimator_
    logger.info(f"迴歸器最佳參數: {rs_reg.best_params_}")

    # 訓練分類器
    logger.info("Step 2: 尋找結果分類器最佳參數...")
    base_clf = xgb.XGBClassifier(**get_base_params(False))
    rs_clf = RandomizedSearchCV(base_clf, PARAM_DIST, n_iter=5, cv=3, 
                                scoring='f1_weighted')
    rs_clf.fit(train_df[clf_feat], train_df['target_class'],
               eval_set=[(val_df[clf_feat], val_df['target_class'])], verbose=False)
    classifier = rs_clf.best_estimator_
    logger.info(f"分類器最佳參數: {rs_clf.best_params_}")

    # 存檔
    bundle = {
        'classifier': classifier, 
        'regressor': regressor,
        'reg_features': reg_feat, 
        'clf_features': clf_feat,
        'label_map': LABEL_MAP,
        'test_data': (test_df, test_df[['target_class', 'hit_distance_sc']]),
        'config': config.get_all()  # 將配置也存入模型
    }
    joblib.dump(bundle, model_name)
    logger.info(f"模型訓練完成並儲存: {model_name}")

def train_incremental(new_csv, model_name):
    logger.info("=== 啟動增量接續訓練 ===")
    logger.info(f"新數據: {new_csv}, 模型: {model_name}")
    
    bundle = joblib.load(model_name)
    df_new = pd.read_csv(os.path.join(DATA_DIR, new_csv))
    df_new, _, _ = align_and_preprocess(df_new, bundle['reg_features'], bundle['clf_features'])
    
    train_df, val_df = train_test_split(df_new, test_size=0.2, 
                                        random_state=MODEL_CONFIG['random_state'])
    
    logger.info(f"增量訓練 - 訓練集: {len(train_df)}, 驗證集: {len(val_df)}")

    # 使用配置中的增量參數
    inc_params = {
        'learning_rate': INC_CONFIG['learning_rate'],
        'n_estimators': INC_CONFIG['n_estimators'],
        'device': TRAIN_CONFIG['device']
    }

    logger.info("更新迴歸器...")
    bundle['regressor'].set_params(**inc_params)
    bundle['regressor'].fit(train_df[bundle['reg_features']], train_df['hit_distance_sc'],
                            xgb_model=bundle['regressor'].get_booster(),
                            eval_set=[(val_df[bundle['reg_features']], val_df['hit_distance_sc'])], 
                            verbose=False)

    logger.info("更新分類器...")
    bundle['classifier'].set_params(**inc_params)
    bundle['classifier'].fit(train_df[bundle['clf_features']], train_df['target_class'],
                             xgb_model=bundle['classifier'].get_booster(),
                             eval_set=[(val_df[bundle['clf_features']], val_df['target_class'])], 
                             verbose=False)

    joblib.dump(bundle, model_name)
    logger.info("增量模型更新完成")

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