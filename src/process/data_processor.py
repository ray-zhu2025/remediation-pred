"""
数据处理器类
用于加载和处理训练和测试数据
"""

import os
import numpy as np
import pandas as pd
import time
from typing import Dict, Tuple, Any
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import KMeansSMOTE
from src.config.version_config import VersionConfig
import logging

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, 
                 data_dir: str = 'data/training',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 use_oversampling: bool = True,
                 sampling_method: str = None,
                 sampling_strategy: str = 'auto',
                 min_samples: int = 30,
                 cv_folds: int = 5):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录
            test_size: 测试集比例
            random_state: 随机种子
            use_oversampling: 是否使用过采样
            sampling_method: 过采样方法，如果为None则从版本配置中获取
            sampling_strategy: 采样策略 ('auto' 或 'minority' 或 dict)
            min_samples: 每个类别的最小样本数
            cv_folds: 交叉验证折数
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        self.use_oversampling = use_oversampling
        self.sampling_method = sampling_method or self._get_sampling_method_from_version()
        self.sampling_strategy = sampling_strategy
        self.min_samples = min_samples
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
        self.feature_importance = {}
        self.selected_features = {}
        
    def load_soil_data(self) -> Dict[str, np.ndarray]:
        """
        加载土壤数据
        
        Returns:
            包含训练和测试数据的字典
        """
        return self._load_data('soil')
        
    def load_groundwater_data(self) -> Dict[str, np.ndarray]:
        """
        加载地下水数据
        
        Returns:
            包含训练和测试数据的字典
        """
        return self._load_data('groundwater')
        
    def update_feature_importance(self, importance_dict: Dict[str, float], dataset_type: str):
        """
        更新特征重要性
        
        Args:
            importance_dict: 特征重要性字典
            dataset_type: 数据集类型
        """
        self.feature_importance[dataset_type] = importance_dict
        
    def _load_data(self, data_type: str) -> Dict[str, np.ndarray]:
        """
        通用的数据加载方法
        
        Args:
            data_type: 数据类型 ('soil' 或 'groundwater')
            
        Returns:
            包含训练和测试数据的字典
        """
        start_time = time.time()
        self.logger.info(f"开始加载{data_type}数据...")
        
        # 加载数据
        data = pd.read_csv(os.path.join(self.data_dir, f'{data_type}_training.csv'))
        self.logger.info(f"原始数据形状: {data.shape}")
        self.logger.info(f"原始类别分布: {Counter(data['修复技术'])}")
        
        # 预处理数据
        X, y = self._preprocess_data(data, data_type)
        self.logger.info(f"预处理后数据形状: {X.shape}")
        self.logger.info(f"预处理后类别分布: {Counter(y)}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        self.logger.info(f"训练集形状: {X_train.shape}")
        self.logger.info(f"测试集形状: {X_test.shape}")
        self.logger.info(f"训练集类别分布: {Counter(y_train)}")
        self.logger.info(f"测试集类别分布: {Counter(y_test)}")
        
        # 应用采样
        if self.use_oversampling:
            # 统计每个类别的样本数
            class_counts = Counter(y_train)
            self.logger.info(f"采样前训练集类别分布: {class_counts}")
            
            # 创建采样pipeline
            sampling_pipeline = self._create_sampling_pipeline(class_counts, is_soil=(data_type == 'soil'))
            
            # 应用采样
            if sampling_pipeline is not None:
                with tqdm(total=1, desc="数据采样") as pbar:
                    X_train, y_train = sampling_pipeline.fit_resample(X_train, y_train)
                    pbar.update(1)
                    
                self.logger.info(f"采样后训练集类别分布: {Counter(y_train)}")
                self.logger.info(f"采样后训练集形状: {X_train.shape}")
            else:
                self.logger.info("不使用过采样")
        
        load_time = time.time() - start_time
        self.logger.info(f"数据加载完成，耗时: {load_time:.2f}秒")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
    def _remove_high_missing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除高缺失率特征
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        missing_ratio = data.isnull().sum() / len(data)
        high_missing = missing_ratio[missing_ratio > 0.5].index
        if len(high_missing) > 0:
            self.logger.info(f"移除高缺失率特征: {list(high_missing)}")
            data = data.drop(columns=high_missing)
        return data
        
    def _remove_low_variance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除低方差特征
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            variance = data[numeric_cols].var()
            low_variance = variance[variance < 0.01].index
            if len(low_variance) > 0:
                self.logger.info(f"移除低方差特征: {list(low_variance)}")
                data = data.drop(columns=low_variance)
        return data
        
    def _remove_high_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除高相关特征
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
            if len(high_corr) > 0:
                self.logger.info(f"移除高相关特征: {high_corr}")
                data = data.drop(columns=high_corr)
        return data
        
    def _select_features(self, data: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        特征选择
        
        Args:
            data: 输入数据
            dataset_type: 数据集类型 ('soil' 或 'groundwater')
            
        Returns:
            处理后的数据
        """
        # 移除高缺失率特征
        data = self._remove_high_missing_features(data)
        
        # 移除低方差特征
        data = self._remove_low_variance_features(data)
        
        # 处理高相关特征
        data = self._remove_high_correlation_features(data)
        
        # 更新已选择的特征
        self.selected_features[dataset_type] = list(data.columns)
        self.logger.info(f"\n{dataset_type}数据集最终选择的特征:")
        self.logger.info(f"特征数量: {len(self.selected_features[dataset_type])}")
        self.logger.info(f"特征列表: {self.selected_features[dataset_type]}")
        return data
        
    def _preprocess_data(self, data: pd.DataFrame, dataset_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理数据
        
        Args:
            data: 输入数据
            dataset_type: 数据集类型
            
        Returns:
            处理后的特征和标签
        """
        start_time = time.time()
        self.logger.info("开始数据预处理...")
        
        # 分离特征和标签
        y = self.label_encoder.fit_transform(data['修复技术'])
        
        # 根据数据集类型选择要删除的列
        columns_to_drop = ['修复技术', '场地名称', '场地分区']  # 公共列
        if dataset_type == 'soil':
            # 土壤数据需要过滤的特征
            soil_drop_columns = [
                '修复时间', '修复时间.1','费用', '修复面积', '修复土方量', '修复成本'
            ]
            columns_to_drop.extend(soil_drop_columns)
        else:
            # 地下水数据需要过滤的特征
            groundwater_drop_columns = [
                '修复时间', '费用'
            ]
            columns_to_drop.extend(groundwater_drop_columns)
            
        # 删除指定的列
        X = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)
        
        # 记录被删除的特征
        self.logger.info(f"删除的特征: {[col for col in columns_to_drop if col in data.columns]}")
        
        # 特征选择
        X = self._select_features(X, dataset_type)
        
        # 过滤样本数少于阈值的类别
        class_counts = Counter(y)
        valid_classes = [cls for cls, count in class_counts.items() if count >= self.min_samples]
        mask = np.isin(y, valid_classes)
        X = X[mask]
        y = y[mask]
        
        self.logger.info(f"过滤后的类别分布: {Counter(y)}")
        
        # 识别数值列和分类列
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # 创建预处理器
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # 使用中位数填充
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # 使用众数填充
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # 转换数据
        with tqdm(total=1, desc="特征转换") as pbar:
            X = preprocessor.fit_transform(X)
            pbar.update(1)
        
        preprocess_time = time.time() - start_time
        self.logger.info(f"数据预处理完成，耗时: {preprocess_time:.2f}秒")
        
        return X, y
        
    def _get_sampling_method_from_version(self) -> str:
        """从版本配置中获取采样方法"""
        version = VersionConfig.get_version()
        version_changes = VersionConfig.VERSION_HISTORY[version]['changes']
        
        # 根据版本更新记录确定采样方法
        if "SMOTE" in str(version_changes):
            return 'smote'
        elif "ADASYN" in str(version_changes):
            return 'adasyn'
        elif "BorderlineSMOTE" in str(version_changes):
            return 'borderline_smote'
        elif "SVMSMOTE" in str(version_changes):
            return 'svm_smote'
        elif "SMOTETomek" in str(version_changes):
            return 'smote_tomek'
        elif "SMOTEENN" in str(version_changes):
            return 'smote_enn'
        elif "KMeansSMOTE" in str(version_changes):
            return 'kmeans_smote'
        else:
            return None  # 不使用过采样
            
    def _create_sampler(self, sampling_strategy: Dict[int, int], n_samples: Dict[int, int]) -> Any:
        """
        创建采样器
        
        Args:
            sampling_strategy: 采样策略
            n_samples: 每个类别的样本数量
            
        Returns:
            采样器实例
        """
        common_params = {
            'sampling_strategy': sampling_strategy,
            'random_state': self.random_state,
            'k_neighbors': min(5, min(n_samples.values()) - 1)
        }
        
        if self.sampling_method == 'smote':
            return SMOTE(**common_params)
        elif self.sampling_method == 'adasyn':
            return ADASYN(**common_params)
        elif self.sampling_method == 'BorderlineSMOTE':
            return BorderlineSMOTE(**common_params)
        elif self.sampling_method == 'KMeansSMOTE':
            return KMeansSMOTE(**common_params, cluster_balance_threshold=0.1)
        else:
            raise ValueError(f"不支持的过采样方法: {self.sampling_method}")
            
    def _create_sampling_pipeline(self, n_samples: Dict[int, int], is_soil: bool = True) -> Pipeline:
        """
        创建采样pipeline
        
        Args:
            n_samples: 每个类别的样本数量
            is_soil: 是否是土壤数据
            
        Returns:
            采样pipeline
        """
        if not self.use_oversampling or self.sampling_method is None:
            return None
            
        # 计算采样策略
        # 使用最大类样本数作为目标样本数
        max_samples = max(n_samples.values())
        sampling_strategy = {}
        for k, v in n_samples.items():
            if v < max_samples:
                sampling_strategy[k] = max_samples  # 所有类别采样到最大类样本数
                
        self.logger.info(f"最大类样本数: {max_samples}")
        self.logger.info(f"采样策略: {sampling_strategy}")
            
        # 创建过采样器
        over = self._create_sampler(sampling_strategy, n_samples)
            
        # 返回只包含过采样的pipeline
        return Pipeline(steps=[('over', over)]) 