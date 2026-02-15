import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore")

def evaluate_baseball_model(model_path='baseball_dual_model.pkl'):
    print(f"--- 啟動模型評估流水線: {model_path} ---")
    
    # 1. 載入模型 Bundle
    try:
        bundle = joblib.load(model_path)
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 {model_path}")
        return

    # 2. 獲取模型與測試數據
    # 修正：強制將模型轉移至 CPU 進行評估，消除設備不匹配警告
    clf = bundle['classifier'].set_params(device="cpu")
    reg = bundle['regressor'].set_params(device="cpu")
    features = bundle['features']
    label_map = bundle['label_map']
    X_test, y_test = bundle['test_data']
    
    # 取得類別名稱列表
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    # 3. 執行預測
    y_pred_class = clf.predict(X_test)
    y_pred_dist = reg.predict(X_test)

    # 4. 分類模型評估報告
    print("\n[ 分類模型表現報告 ]")
    # 修正：加入 zero_division=0 消除精確度計算警告
    print(classification_report(y_test['target_class'], y_pred_class, 
                                target_names=target_names, zero_division=0))

    # 5. 迴歸模型評估報告 (飛行距離)
    print("\n[ 迴歸模型表現報告 (距離) ]")
    # 僅針對有距離數據的樣本進行評估
    mask = y_test['hit_distance_sc'].notna()
    mae = mean_absolute_error(y_test['hit_distance_sc'][mask], y_pred_dist[mask])
    r2 = r2_score(y_test['hit_distance_sc'][mask], y_pred_dist[mask])
    print(f"平均絕對誤差 (MAE): {mae:.2f} ft")
    print(f"解釋方差 (R2 Score): {r2:.4f}")

    # 6. 視覺化：混淆矩陣
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    cm = confusion_matrix(y_test['target_class'], y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix (Hit Results)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # 7. 視覺化：預測值 vs 真實值
    plt.subplot(122)
    plt.scatter(y_test['hit_distance_sc'][mask], y_pred_dist[mask], alpha=0.3, color='green')
    plt.plot([y_test['hit_distance_sc'][mask].min(), y_test['hit_distance_sc'][mask].max()],
             [y_test['hit_distance_sc'][mask].min(), y_test['hit_distance_sc'][mask].max()], 
             'r--', lw=2)
    plt.title("Distance Prediction: Actual vs Predicted")
    plt.xlabel("Actual Distance (ft)")
    plt.ylabel("Predicted Distance (ft)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_baseball_model()