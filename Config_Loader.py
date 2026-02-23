# Config_Loader.py
import yaml
import os
from typing import Dict, Any
import logging

class ConfigLoader:
    """配置載入器，負責讀取和管理所有設定"""
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: str = "config.yaml"):
        """單例模式，確保整個應用程式使用同一份配置"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "config.yaml"):
        if self._config is None:
            self.config_path = config_path
            self._load_config()
    
    def _load_config(self):
        """載入 YAML 配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 確保必要的目錄存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """確保必要的目錄存在"""
        directories = [
            self._config['data']['data_dir'],
            os.path.dirname(self._config['logging']['file'])
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        安全的取得配置值
        用法: config.get('model', 'training', 'device')
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """取得完整配置"""
        return self._config.copy()
    
    def update(self, key_path: str, value: Any):
        """
        更新配置值 (支援點記法)
        用法: config.update('model.training.device', 'cpu')
        """
        keys = key_path.split('.')
        target = self._config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def reload(self):
        """重新載入配置"""
        self._load_config()

# 建立全域配置實例
config = ConfigLoader()