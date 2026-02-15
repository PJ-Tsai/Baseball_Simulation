#!/bin/bash

# --- 設定預設日期  ---
DEFAULT_START="2025-08-01"
DEFAULT_END="2025-08-31"
DATA_DIR="datasets"

echo "------------------------------------------"
echo "   MLB Statcast 數據獲取與處理腳本"
echo "------------------------------------------"

# 詢問是否要使用預設日期
read -p "是否使用預設日期 $DEFAULT_START ~ $DEFAULT_END ? (y/n): " use_default

if [ "$use_default" = "y" ] || [ "$use_default" = "Y" ]; then
    START_DT=$DEFAULT_START
    END_DT=$DEFAULT_END
else
    read -p "請輸入開始日期 (YYYY-MM-DD): " START_DT
    read -p "請輸入結束日期 (YYYY-MM-DD): " END_DT
fi

# 執行 Python 程式碼
python Data_Utils.py --start "$START_DT" --end "$END_DT" --dir "$DATA_DIR"

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo "任務成功完成。"
else
    echo "任務失敗，請檢查網路連線或日期格式。"
fi