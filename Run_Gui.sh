#!/bin/bash
# Run_Gui.sh - 給 Git Bash 使用的啟動腳本

# 設定顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   MLB 擊球分析系統 - Git Bash 啟動腳本  ${NC}"
echo -e "${GREEN}========================================${NC}"

# 檢查 Conda 環境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}警告: 未啟動 Conda 環境${NC}"
    echo "當前環境: $(which python)"
    echo ""
    echo "可用的 Conda 環境:"
    conda env list
    echo ""
    read -p "請輸入要使用的 Conda 環境名稱 [直接按 Enter 使用當前環境]: " CONDA_ENV
    if [ ! -z "$CONDA_ENV" ]; then
        echo "啟動環境: $CONDA_ENV"
        source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV
    fi
else
    echo -e "${GREEN}目前 Conda 環境: ${CONDA_DEFAULT_ENV}${NC}"
fi

echo -e "\n${YELLOW}檢查必要套件...${NC}"

# 檢查 Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}錯誤: 找不到 Python${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python: $(python --version)${NC}"

# 檢查 tkinter
python -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ tkinter 未安裝${NC}"
    echo "在 Conda 環境中安裝 tkinter:"
    echo "  conda install tk"
    echo ""
    read -p "是否現在安裝？(y/n): " install_tk
    if [[ "$install_tk" =~ ^[Yy]$ ]]; then
        conda install -y tk
    fi
else
    echo -e "${GREEN}✓ tkinter 已安裝${NC}"
fi

# 檢查其他套件
packages=("numpy" "matplotlib" "pandas" "sklearn" "joblib" "yaml")
for pkg in "${packages[@]}"; do
    python -c "import $pkg" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $pkg 已安裝${NC}"
    else
        echo -e "${RED}✗ $pkg 未安裝${NC}"
        missing=1
    fi
done

# 檢查 ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ ffmpeg 已安裝${NC}"
else
    echo -e "${RED}✗ ffmpeg 未安裝${NC}"
    echo "影片輸出功能可能無法使用"
    echo "安裝方式: conda install ffmpeg"
fi

# 檢查模型檔案
if [ ! -f "baseball_dual_model.pkl" ]; then
    echo -e "\n${YELLOW}警告: 找不到模型檔案 baseball_dual_model.pkl${NC}"
    echo "請確認模型檔案是否存在"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   啟動 MLB 擊球分析系統...${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 設定 Python 路徑（解決模組導入問題）
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 啟動 GUI
python Baseball_Predictor_GUI.py

# 如果執行失敗，顯示錯誤訊息
if [ $? -ne 0 ]; then
    echo -e "\n${RED}執行失敗！${NC}"
    echo "錯誤訊息如上所示"
    echo ""
    echo "可能的解決方案："
    echo "1. 確認所有套件都已安裝："
    echo "   conda install tk numpy matplotlib pandas scikit-learn joblib pyyaml"
    echo ""
    echo "2. 確認模型檔案存在："
    echo "   ls -l baseball_dual_model.pkl"
    echo ""
    echo "3. 確認 Python 路徑設定："
    echo "   echo \$PYTHONPATH"
    echo ""
    read -p "按 Enter 鍵繼續..."
fi