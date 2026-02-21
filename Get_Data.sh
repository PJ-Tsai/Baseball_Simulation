#!/bin/bash

# --- 設定預設日期  ---
DEFAULT_YEAR="2023"
DEFAULT_MONTH="3"
DATA_DIR="datasets"

echo "------------------------------------------"
echo "   MLB Statcast 數據獲取與處理腳本"
echo "------------------------------------------"

# 詢問是否要使用預設日期
read -p "是否使用預設日期 $DEFAULT_YEAR 年 $DEFAULT_MONTH 月 ? (y/n): " use_default

if [ "$use_default" = "y" ] || [ "$use_default" = "Y" ]; then
    YEAR=$DEFAULT_YEAR
    MONTH=$DEFAULT_MONTH    
else
    read -p "請輸入年份 (e.g., 2024): " YEAR
    read -p "請輸入月份 (3-11): " MONTH
fi

# 執行 Python 程式碼
python Data_Utils.py --year "$YEAR" --month "$MONTH" --dir "$DATA_DIR"

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo "資料抓取成功。"
else
    echo "資料抓取失敗，請檢查網路連線或日期格式。"
fi