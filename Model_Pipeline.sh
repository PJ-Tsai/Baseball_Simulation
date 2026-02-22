#!/bin/bash

MODEL_NAME="baseball_dual_model.pkl"
TRAINER="Model_Trainer.py"
EVALUATOR="Evaluate_Model.py"
# PREDICTOR="Predictor_Engine.py"
PREDICTOR="ML_Physics_Hybrid_Predictor.py"
DEFAULT_START_DATE="2025-03-01"
DEFAULT_END_DATE="2025-03-31"
# Parameter for Cheat Mode
EV_BOOST=1.0
DIST_BOOST=1.2

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
        python $EVALUATOR --model "$MODEL_NAME"
        ;;
    2)
        # 為防止資料量過大
        # 只使用 2024 年度的數據進行訓練
        files=$(ls datasets/ | grep "ml_data_2024" | grep ".csv")
        echo "即將合併以下檔案: $files"

        python $TRAINER --mode full --files $files --model "$MODEL_NAME"
        python $EVALUATOR --model "$MODEL_NAME"
        ;;
    3)  
        read -p "是否使用預設日期 $DEFAULT_START_DATE ~ $DEFAULT_END_DATE ? (y/n): " use_default
        if [[ "$use_default" =~ ^[Yy]$ ]]; then
            START_DT=$DEFAULT_START_DATE
            END_DT=$DEFAULT_END_DATE    
        else
            read -p "請輸入資料年份 (YYYY): " YEAR
            read -p "請輸入資料月份 (3-11): " MONTH
            
            # 格式化起始日期為 YYYY-MM-01
            START_DT=$(printf "%04d-%02d-01" $YEAR $MONTH)
            
            # 使用更穩定的方式計算該月最後一天
            # 邏輯：下個月的第 0 天即為本月的最後一天
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS (BSD date) 語法
                NEXT_MONTH=$(printf "%02d" $((10#$MONTH % 12 + 1)))
                NEXT_YEAR=$((10#$MONTH == 12 ? YEAR + 1 : YEAR))
                END_DT=$(date -j -f "%Y-%m-%d" "$NEXT_YEAR-$NEXT_MONTH-01" "+%Y-%m-%d" -v-1d)
            else
                # Linux (GNU date) 語法
                END_DT=$(date -d "$START_DT +1 month -1 day" +"%Y-%m-%d")
            fi
        fi
        
        echo "設定處理區間為: $START_DT 至 $END_DT"
        FILE_PATTERN="ml_data_${START_DT}_${END_DT}.csv"
        python $TRAINER --mode inc --files "$FILE_PATTERN" --model "$MODEL_NAME"
        python $EVALUATOR --model "$MODEL_NAME"
        ;;
    4)
        python $EVALUATOR --model "$MODEL_NAME"
        ;;
    5)
        python $PREDICTOR --model "$MODEL_NAME" --ev_boost $EV_BOOST --dist_boost $DIST_BOOST
        ;;
    *)
        echo "無效選項，退出。"
        ;;
esac