"""
版本配置模块
用于管理模型版本和更新记录
"""

from typing import Dict, List
from datetime import datetime

class VersionConfig:
    """版本配置类"""
    
    # 当前版本号
    VERSION = "1.1.6"
    
    # 版本历史记录
    VERSION_HISTORY = {
        "1.0.0": {
            "date": "2024-03-20",
            "changes": "初始版本",
            "sampling_method": None
        },
        "1.1.0": {
            "date": "2024-03-28",
            "changes": "smote",
            "sampling_method": "smote"
        },
        "1.1.1": {
            "date": "2024-03-29",
            "changes": "adasyn",
            "sampling_method": "adasyn"
        },
        "1.1.2": {
            "date": "2024-03-30",
            "changes": "borderline_smote",
            "sampling_method": "borderline_smote"
        },
        "1.1.3": {
            "date": "2024-03-31",
            "changes": "svm_smote",
            "sampling_method": "svm_smote"
        },
        "1.1.4": {
            "date": "2024-04-01",
            "changes": "smote_tomek",
            "sampling_method": "smote_tomek"
        },
        "1.1.5": {
            "date": "2024-04-02",
            "changes": "smote_enn",
            "sampling_method": "smote_enn"
        },
        "1.1.6": {
            "date": "2024-04-03",
            "changes": "kmeans_smote",
            "sampling_method": "kmeans_smote"
        }
    }
    
    @classmethod
    def get_version(cls) -> str:
        """获取当前版本号"""
        return cls.VERSION
    
    @classmethod
    def set_version(cls, version: str) -> None:
        """
        设置当前版本号
        
        Args:
            version: 版本号
        """
        if version not in cls.VERSION_HISTORY:
            raise ValueError(f"版本号 {version} 不存在")
        cls.VERSION = version
    
    @classmethod
    def get_version_info(cls, version: str = None) -> Dict:
        """
        获取版本信息
        
        Args:
            version: 版本号，如果为None则获取当前版本信息
            
        Returns:
            版本信息字典
        """
        version = version or cls.VERSION
        if version not in cls.VERSION_HISTORY:
            raise ValueError(f"版本号 {version} 不存在")
        return cls.VERSION_HISTORY[version]
    
    @classmethod
    def get_model_type(cls, version: str = None) -> str:
        """
        获取模型类型
        
        Args:
            version: 版本号，如果为None则获取当前版本的模型类型
            
        Returns:
            模型类型
        """
        return cls.get_version_info(version)['model_type']
    
    @classmethod
    def get_sampling_method(cls, version: str = None) -> str:
        """
        获取采样方法
        
        Args:
            version: 版本号，如果为None则获取当前版本的采样方法
            
        Returns:
            采样方法
        """
        return cls.get_version_info(version)['sampling_method']
    
    @classmethod
    def get_versions_between(cls, start_version: str, end_version: str) -> List[str]:
        """
        获取两个版本之间的所有版本号
        
        Args:
            start_version: 起始版本号
            end_version: 结束版本号
            
        Returns:
            版本号列表
        """
        versions = list(cls.VERSION_HISTORY.keys())
        start_idx = versions.index(start_version)
        end_idx = versions.index(end_version)
        return versions[start_idx:end_idx + 1]
    
    @classmethod
    def get_version_config(cls, version: str = None) -> Dict:
        """
        获取版本配置
        
        Args:
            version: 版本号，如果为None则获取当前版本的配置
            
        Returns:
            版本配置字典
        """
        version = version or cls.VERSION
        info = cls.get_version_info(version)
        return {
            'version': version,
            'sampling_method': info['sampling_method'],
            'changes': info['changes'],
            'date': info['date']
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
    
    # 采样策略配置
    SAMPLING_STRATEGIES = {
        'smote': {
            'random_state': 42,
            'k_neighbors': 5
        },
        'adasyn': {
            'random_state': 42,
            'n_neighbors': 5
        },
        'borderline_smote': {
            'random_state': 42,
            'k_neighbors': 5
        },
        'svm_smote': {
            'random_state': 42,
            'k_neighbors': 5
        },
        'smote_tomek': {
            'random_state': 42,
            'k_neighbors': 5
        },
        'smote_enn': {
            'random_state': 42,
            'k_neighbors': 5
        },
        'kmeans_smote': {
            'random_state': 42,
            'k_neighbors': 5,
            'cluster_balance_threshold': 0.05,
            'n_clusters': 3
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
        },
        'TabPFN': {
            'device': 'cpu',
            'N_ensemble_configurations': 32,
            'batch_size_inference': 1024,
            'normalize_features': True,
            'feature_importance': True,
            'base_path': None,
            'c': 1.0,
            'seed': 42,
            'max_num_features': 1000,
            'eval_positions': None,
            'verbose': True
        }
    }
    
    # 模型类型配置
    MODEL_TYPES = {
        'autogluon': {
            'soil': 'DomainModel',
            'groundwater': 'DomainModel',
            'save_model': True
        },
        'tabpfn': {
            'soil': 'DomainTabPFNModel',
            'groundwater': 'DomainTabPFNModel',
            'save_model': False  # TabPFN是预训练模型，不需要保存
        }
    }
    
    # 模型保存路径模板
    MODEL_SAVE_PATH_TEMPLATE = "models/{model_type}/v{version}/{timestamp}"
    
    # 评估指标保存路径模板
    METRICS_SAVE_PATH_TEMPLATE = "output/metrics/{model_type}_v{version}_{timestamp}.json"
    
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