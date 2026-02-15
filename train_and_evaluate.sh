#!/bin/bash

# --- 設定 ---
START_DATE="2024-04-01"
END_DATE="2024-05-31"
MODEL_FILE="baseball_dual_model.pkl"
DATA_FILE="ml_data_${START_DATE}_${END_DATE}.csv"

echo "=========================================="
echo "   棒球擊球模型自動化流水線 (Pipeline)    "
echo "=========================================="

# 1. 檢查訓練資料是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "訓練資料不存在，開始下載資料..."
    python Load_Data.py
    if [ $? -ne 0 ] || [ ! -f "$DATA_FILE" ]; then
        echo "錯誤: 無法取得訓練資料。"
        exit 1
    fi
fi

# 2. 開始訓練模型
echo "步驟 1: 開始訓練模型..."
python Train_Model.py

# 檢查訓練是否成功
if [ $? -eq 0 ] && [ -f "$MODEL_FILE" ]; then
    echo "訓練成功！模型已儲存至 $MODEL_FILE"
else
    echo "錯誤: 模型訓練失敗。"
    exit 1
fi

echo "------------------------------------------"

# 3. 開始效能評估
echo "步驟 2: 開始效能評估與驗證..."
python Evaluate_Model.py

echo "=========================================="
echo "             所有任務執行完畢             "
echo "=========================================="