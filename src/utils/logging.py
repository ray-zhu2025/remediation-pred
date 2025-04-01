import logging
import os
from datetime import datetime
from src.config.version_config import VersionConfig

def setup_logging(log_level=logging.INFO):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别，默认为 INFO
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前版本号
    version = VersionConfig.get_version()
    
    # 创建日志文件名（包含时间戳和版本号）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_v{version}_{timestamp}.log")
    
    # 配置日志记录器
    logger = logging.getLogger("model_logger")
    logger.setLevel(log_level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 