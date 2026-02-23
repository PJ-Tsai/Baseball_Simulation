import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
import warnings
from Config_Loader import config
from Logger_Setup import setup_logger

warnings.filterwarnings("ignore")

logger = setup_logger(__name__)

def evaluate_baseball_model(model_path=None):
    if model_path is None:
        model_path = config.get('model', 'name')
        
    logger.info(f"啟動模型評估流水線: {model_path}")
    
    try:
        bundle = joblib.load(model_path)
        logger.info("模型載入成功")
        
        # 如果模型有儲存配置，可以比較
        if 'config' in bundle:
            logger.info("模型包含訓練時的配置資訊")
    except FileNotFoundError:
        logger.error(f"找不到模型檔案 {model_path}")
        return

    # 獲取模型與特徵清單
    clf = bundle['classifier'].set_params(device="cpu")
    reg = bundle['regressor'].set_params(device="cpu")
    reg_features = bundle['reg_features']
    clf_features = bundle['clf_features']
    label_map = bundle['label_map']
    X_test_df, y_test_df = bundle['test_data']
    
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    logger.info(f"評估類別: {target_names}")

    # 模擬串接預測流程
    logger.info("正在執行預測評估...")
    
    # Step A: 先預測距離
    pred_distances = reg.predict(X_test_df[reg_features])
    
    # Step B: 將「預測出的距離」填入特徵中，再餵給分類器
    X_test_for_clf = X_test_df[clf_features].copy()
    X_test_for_clf['hit_distance_sc'] = pred_distances
    
    y_pred_class = clf.predict(X_test_for_clf)
    y_pred_dist = pred_distances

    # 分類模型評估報告
    logger.info("分類模型表現報告:")
    report = classification_report(y_test_df['target_class'], y_pred_class, 
                                   target_names=target_names, zero_division=0)
    print("\n[ 分類模型表現報告 ]")
    print(report)

    # 迴歸模型評估報告
    logger.info("迴歸模型表現報告:")
    mask = y_test_df['hit_distance_sc'].notna()
    mae = mean_absolute_error(y_test_df['hit_distance_sc'][mask], y_pred_dist[mask])
    r2 = r2_score(y_test_df['hit_distance_sc'][mask], y_pred_dist[mask])
    
    print("\n[ 迴歸模型表現報告 (距離) ]")
    print(f"平均絕對誤差 (MAE): {mae:.2f} ft")
    print(f"解釋方差 (R2 Score): {r2:.4f}")
    
    logger.info(f"MAE: {mae:.2f} ft, R2: {r2:.4f}")

    # 視覺化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    cm = confusion_matrix(y_test_df['target_class'], y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix (Hit Results)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplot(122)
    plt.scatter(y_test_df['hit_distance_sc'][mask], y_pred_dist[mask], alpha=0.3, color='green')
    plt.plot([y_test_df['hit_distance_sc'][mask].min(), y_test_df['hit_distance_sc'][mask].max()],
             [y_test_df['hit_distance_sc'][mask].min(), y_test_df['hit_distance_sc'][mask].max()], 
             'r--', lw=2)
    plt.title("Distance Prediction: Actual vs Predicted")
    plt.xlabel("Actual Distance (ft)")
    plt.ylabel("Predicted Distance (ft)")

    plt.tight_layout()
    logger.info("正在顯示評估圖表...")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseball Model")
    parser.add_argument('--model', type=str, default=None, help='模型檔案路徑')
    args = parser.parse_args()
    evaluate_baseball_model(model_path=args.model)