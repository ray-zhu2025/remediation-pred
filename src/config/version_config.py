"""
版本配置文件
"""

from datetime import datetime

class VersionConfig:
    # 当前版本号
    CURRENT_VERSION = "1.0.5"
    
    # 版本历史记录
    VERSION_HISTORY = {
        "1.0.0": {
            "changes": [
                "初始版本",
                "基础模型训练功能",
                "无采样策略"
            ]
        },
        "1.0.1": {
            "changes": [
                "添加SMOTE采样策略",
                "优化模型训练流程"
            ]
        },
        "1.0.2": {
            "changes": [
                "添加ADASYN采样策略",
                "优化采样参数配置"
            ]
        },
        "1.0.3": {
            "changes": [
                "添加BorderlineSMOTE采样策略",
                "优化采样参数配置"
            ]
        },
        "1.0.4": {
            "changes": [
                "添加KMeansSMOTE采样策略",
                "优化采样参数配置"
            ]
        },
        "1.0.5": {
            "changes": [
                "使用ADASYN采样策略替代KMeansSMOTE",
                "优化采样参数配置"
            ],
            "model_save_path": {
                "soil": "models/soil/v1.0.5",
                "groundwater": "models/groundwater/v1.0.5"
            },
            "metrics_save_path": "output/metrics",
            "log_file": "logs/v1.0.5.log"
        }
    }
    
    # 日志配置
    LOG_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
    
    # 模型保存路径
    MODEL_SAVE_PATH = {
        "soil": "models/soil",
        "groundwater": "models/groundwater"
    }
    
    # 评估指标保存路径
    METRICS_SAVE_PATH = {
        "soil": "output/metrics/soil",
        "groundwater": "output/metrics/groundwater"
    }
    
    # 日志文件路径
    LOG_FILE_PATH = {
        "soil": "output/logs/soil",
        "groundwater": "output/logs/groundwater"
    }
    
    @classmethod
    def get_model_save_path(cls, model_type: str) -> str:
        """获取模型保存路径"""
        return cls.MODEL_SAVE_PATH.get(model_type, "models")
    
    @classmethod
    def get_metrics_save_path(cls, model_type: str) -> str:
        """获取评估指标保存路径"""
        return cls.METRICS_SAVE_PATH.get(model_type, "output/metrics")
    
    @classmethod
    def get_log_file_path(cls, model_type: str) -> str:
        """获取日志文件路径"""
        return cls.LOG_FILE_PATH.get(model_type, "output/logs")

    # 模型保存路径模板
    MODEL_SAVE_PATH_TEMPLATE = "models/{model_type}/v{version}/{timestamp}"
    
    # 评估指标保存路径模板
    METRICS_SAVE_PATH_TEMPLATE = "output/metrics/{model_type}_v{version}_{timestamp}.json"
    
    # 采样策略配置
    SAMPLING_STRATEGIES = {
        "SMOTE": {
            "k_neighbors": 5,
            "random_state": 42
        },
        "ADASYN": {
            "sampling_strategy": "auto",
            "random_state": 42,
            "n_neighbors": 5,
            "n_jobs": -1
        },
        "BorderlineSMOTE": {
            "k_neighbors": 5,
            "random_state": 42
        },
        "KMeansSMOTE": {
            "k_neighbors": 5,
            "random_state": 42
        }
    }
    
    @classmethod
    def get_model_save_path(cls, model_type: str) -> str:
        """获取模型保存路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.MODEL_SAVE_PATH_TEMPLATE.format(
            model_type=model_type,
            version=cls.CURRENT_VERSION,
            timestamp=timestamp
        )
    
    @classmethod
    def get_log_file_path(cls, model_type: str) -> str:
        """获取日志文件路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.LOG_CONFIG["file_template"].format(
            model_type=model_type,
            version=cls.CURRENT_VERSION,
            timestamp=timestamp
        )
    
    @classmethod
    def get_metrics_save_path(cls, model_type: str) -> str:
        """获取评估指标保存路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.METRICS_SAVE_PATH_TEMPLATE.format(
            model_type=model_type,
            version=cls.CURRENT_VERSION,
            timestamp=timestamp
        ) 