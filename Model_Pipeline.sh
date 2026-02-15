#!/bin/bash

MODEL_NAME="baseball_dual_model.pkl"
TRAINER="Model_Trainer.py"
EVALUATOR="Evaluate_Model.py"
PREDICTOR="Predict_and_Visualize.py"
DEFAULT_START_DATE="2025-04-01"
DEFAULT_END_DATE="2025-04-30"

echo "------------------------------------------"
echo "   MLB 擊球模型管理流水線"
echo "------------------------------------------"
echo "1) 全量重新訓練 (使用單一新檔案)"
echo "2) 批量合併訓練 (多檔案合併重練)"
echo "3) 增量接續訓練 (不改變結構，僅更新權重)"
echo "4) 執行模型評估"
echo "5) 使用模型進行預測"

read -p "請選擇操作 (1-5): " choice

case $choice in
    1)
        read -p "請輸入數據檔名 (在 datasets 內): " file
        python $TRAINER --mode full --files "$file" --model "$MODEL_NAME"
        python $EVALUATOR
        ;;
    2)
        # 為防止資料量過大
        # 只使用 2024 年度的數據進行訓練
        files=$(ls datasets/ml_data_2024*.csv 2>/dev/null || echo "datasets/*.csv" | xargs -n 1 basename)
        echo "即將合併以下檔案: $files"
        python $TRAINER --mode full --files $files --model "$MODEL_NAME"
        python $EVALUATOR
        ;;
    3)  
        # 詢問是否要使用預設日期
        read -p "是否使用預設日期 $DEFAULT_START_DATE ~ $DEFAULT_END_DATE ? (y/n): " use_default
        if [ "$use_default" = "y" ] || [ "$use_default" = "Y" ]; then
            START_DT=$DEFAULT_START_DATE
            END_DT=$DEFAULT_END_DATE    
        else
            read -p "請輸入開始日期 (YYYY-MM-DD): " START_DT
            read -p "請輸入結束日期 (YYYY-MM-DD): " END_DT
        fi
        FILE_PATTERN="ml_data_${START_DT}_${END_DT}.csv"
        python $TRAINER --mode inc --files "$FILE_PATTERN" --model "$MODEL_NAME"
        python $EVALUATOR
        ;;
    4)
        python $EVALUATOR
        ;;
    5)
        python $PREDICTOR
        ;;
    *)
        echo "無效選項，退出。"
        ;;
esac