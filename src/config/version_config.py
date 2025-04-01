"""
版本配置文件
"""

from datetime import datetime

class VersionConfig:
    # 当前版本号
    VERSION = "1.1.6"
    
    # 版本历史记录
    VERSION_HISTORY = {
        "1.0.0": {
            "changes": [
                "初始版本",
                "使用原始数据分布"
            ]
        },
        "1.1.0": {
            "changes": [
                "添加KMeansSMOTE采样策略",
                "土壤数据采样比例提升到95%",
                "地下水数据采样比例提升到80%"
            ]
        },
        "1.1.6": {
            "changes": [
                "优化KMeansSMOTE采样策略",
                "土壤数据采样比例提升到98%",
                "地下水数据采样比例提升到90%"
            ]
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
        'SMOTE': {
            'random_state': 42,
            'k_neighbors': 5
        },
        'ADASYN': {
            'random_state': 42,
            'n_neighbors': 5
        },
        'BorderlineSMOTE': {
            'random_state': 42,
            'k_neighbors': 5
        },
        'KMeansSMOTE': {
            'random_state': 42,
            'k_neighbors': 5,
            'cluster_balance_threshold': 0.1
        }
    }
    
    # 模型参数配置
    MODEL_PARAMS = {
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'XGBoost': {
            'n_estimators': 100,
            'max_depth': 6,
            'random_state': 42
        },
        'LightGBM': {
            'n_estimators': 100,
            'max_depth': 6,
            'random_state': 42
        }
    }
    
    @classmethod
    def get_model_save_path(cls, model_type: str) -> str:
        """获取模型保存路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.MODEL_SAVE_PATH_TEMPLATE.format(
            model_type=model_type,
            version=cls.VERSION,
            timestamp=timestamp
        )
    
    @classmethod
    def get_log_file_path(cls, model_type: str) -> str:
        """获取日志文件路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.LOG_CONFIG["file_template"].format(
            model_type=model_type,
            version=cls.VERSION,
            timestamp=timestamp
        )
    
    @classmethod
    def get_metrics_save_path(cls, model_type: str) -> str:
        """获取评估指标保存路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.METRICS_SAVE_PATH_TEMPLATE.format(
            model_type=model_type,
            version=cls.VERSION,
            timestamp=timestamp
        ) 