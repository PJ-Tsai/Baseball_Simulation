# Logger_Setup.py
import logging
import logging.handlers
import os
import sys
from typing import Optional
from Config_Loader import config

class LoggerSetup:
    """日誌設定管理器"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None):
        """
        取得或建立 logger
        
        Args:
            name: logger 名稱 (通常使用 __name__)
            log_file: 自訂日誌檔案路徑 (選填)
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 從配置讀取設定
        log_config = config.get('logging', default={})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 建立 logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # 避免重複添加 handler
        if logger.handlers:
            return logger
        
        # 建立格式器
        formatter = logging.Formatter(log_format)
        
        # 控制台 Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 檔案 Handler (如果指定或配置中有)
        log_file = log_file or log_config.get('file')
        if log_file:
            # 確保目錄存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # 使用 RotatingFileHandler 避免日誌檔案過大
            max_bytes = log_config.get('max_bytes', 10*1024*1024)  # 預設 10MB
            backup_count = log_config.get('backup_count', 5)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def set_level(cls, name: str, level: str):
        """動態調整特定 logger 的日誌級別"""
        if name in cls._loggers:
            cls._loggers[name].setLevel(getattr(logging, level))

# 方便使用的函式
def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """快速設定 logger 的輔助函式"""
    return LoggerSetup.get_logger(name, log_file)

# 效能追蹤裝飾器
def log_execution_time(logger: Optional[logging.Logger] = None):
    """裝飾器：記錄函式執行時間"""
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = setup_logger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"開始執行 {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.debug(f"完成執行 {func.__name__}，耗時: {elapsed_time:.3f} 秒")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"執行 {func.__name__} 失敗，耗時: {elapsed_time:.3f} 秒，錯誤: {str(e)}")
                raise
        
        return wrapper
    
    return decorator

# 進度追蹤器
class ProgressLogger:
    """用於批次處理的進度日誌"""
    
    def __init__(self, total: int, logger: logging.Logger, 
                 log_interval: int = 10, name: str = "進度"):
        self.total = total
        self.logger = logger
        self.log_interval = log_interval
        self.name = name
        self.current = 0
        self.start_time = None
    
    def __enter__(self):
        self.start_time = __import__('time').time()
        self.logger.info(f"開始 {self.name}，總共 {self.total} 筆")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = __import__('time').time() - self.start_time
        self.logger.info(f"完成 {self.name}，總耗時: {elapsed:.2f} 秒")
    
    def update(self, n: int = 1):
        """更新進度"""
        self.current += n
        if self.current % self.log_interval == 0 or self.current == self.total:
            percentage = (self.current / self.total) * 100
            self.logger.info(f"{self.name}: {self.current}/{self.total} ({percentage:.1f}%)")