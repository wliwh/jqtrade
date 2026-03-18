import logging
import sys

def setup_logger(name="backtest_executor", level=logging.INFO):
    """
    配置并返回一个 logger 实例。
    兼容 Python 3.6。
    """
    logger = logging.getLogger(name)
    
    # 防止重复添加 handler
    if not logger.handlers:
        logger.setLevel(level)
        
        # 创建控制台处理器
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # 创建格式化器
        # 考虑到 JQ Notebook 环境，使用简洁的格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger

# 默认全局 logger
logger = setup_logger()
