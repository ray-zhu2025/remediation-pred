"""
日志配置工具模块
"""
import logging
from typing import Optional

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    统一的日志配置方法
    
    Args:
        name: 日志器名称
        level: 日志级别
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        # 使用主程序的日志处理器
        main_logger = logging.getLogger('__main__')
        if main_logger.handlers:
            logger.handlers = main_logger.handlers
        else:
            # 如果主程序没有处理器，创建默认处理器
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger 