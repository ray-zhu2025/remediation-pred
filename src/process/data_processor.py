"""
数据处理器类
用于加载和处理训练和测试数据
"""

import os
import numpy as np
import pandas as pd
import time
from typing import Dict, Tuple, Optional
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import logging

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, 
                 data_dir: str = 'data/training',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 use_oversampling: bool = False,
                 sampling_method: str = 'smote',
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
            sampling_method: 过采样方法 ('smote' 或 'adasyn')
            sampling_strategy: 采样策略 ('auto' 或 'minority' 或 dict)
            min_samples: 每个类别的最小样本数
            cv_folds: 交叉验证折数
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        self.use_oversampling = use_oversampling
        self.sampling_method = sampling_method
        self.sampling_strategy = sampling_strategy
        self.min_samples = min_samples
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
        self.feature_importance = {}
        self.selected_features = {}
        
    def _create_sampling_pipeline(self, n_samples: Dict[int, int], is_soil: bool = True) -> Pipeline:
        """
        创建采样pipeline
        
        Args:
            n_samples: 每个类别的样本数量
            is_soil: 是否是土壤数据
            
        Returns:
            采样pipeline
        """
        if not self.use_oversampling:
            return None
            
        # 计算采样策略
        if is_soil:
            # 土壤数据：只对样本数小于平均值的类别进行轻度过采样
            mean_samples = np.mean(list(n_samples.values()))
            sampling_strategy = {}
            for k, v in n_samples.items():
                if v < mean_samples:
                    # 将样本数增加到平均值的80%
                    sampling_strategy[k] = int(mean_samples * 0.8)
        else:
            # 地下水数据：对少数类进行适度过采样
            max_samples = max(n_samples.values())
            sampling_strategy = {}
            for k, v in n_samples.items():
                if v < max_samples:
                    # 将样本数增加到最大类的70%
                    sampling_strategy[k] = int(max_samples * 0.7)
                
        self.logger.info(f"采样策略: {sampling_strategy}")
            
        # 创建过采样器，使用更合适的参数
        if self.sampling_method == 'smote':
            over = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=min(5, min(n_samples.values()) - 1)  # 增加邻居数
            )
        elif self.sampling_method == 'adasyn':
            over = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_neighbors=min(5, min(n_samples.values()) - 1)  # 增加邻居数
            )
        else:
            raise ValueError(f"不支持的过采样方法: {self.sampling_method}")
            
        # 创建欠采样器，只对极度不平衡的类别进行欠采样
        under_strategy = {}
        max_ratio = 3  # 允许的最大类别比例
        max_samples = max(n_samples.values())
        for k, v in n_samples.items():
            if v > max_samples / max_ratio:
                under_strategy[k] = int(max_samples / max_ratio)
                
        under = RandomUnderSampler(
            sampling_strategy=under_strategy if under_strategy else 'not minority',
            random_state=self.random_state
        )
        
        # 创建pipeline
        steps = [('over', over)]
        if under_strategy:
            steps.append(('under', under))
        return Pipeline(steps=steps)
        
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
        missing_ratio = data.isnull().sum() / len(data)
        high_missing = missing_ratio[missing_ratio > 0.5].index
        if len(high_missing) > 0:
            self.logger.info(f"移除高缺失率特征: {list(high_missing)}")
            data = data.drop(columns=high_missing)
            
        # 移除低方差特征
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            variance = data[numeric_cols].var()
            low_variance = variance[variance < 0.01].index
            if len(low_variance) > 0:
                self.logger.info(f"移除低方差特征: {list(low_variance)}")
                data = data.drop(columns=low_variance)
                
        # 处理高相关特征
        if len(numeric_cols) > 0:
            remaining_numeric = data.select_dtypes(include=['int64', 'float64']).columns
            if len(remaining_numeric) > 1:
                corr_matrix = data[remaining_numeric].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
                if len(high_corr) > 0:
                    self.logger.info(f"移除高相关特征: {high_corr}")
                    data = data.drop(columns=high_corr)
                
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
                '修复时间', '费用', '修复面积', '修复土方量', '修复成本'
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
        
    def load_soil_data(self) -> Dict[str, np.ndarray]:
        """
        加载土壤数据
        
        Returns:
            包含训练和测试数据的字典
        """
        start_time = time.time()
        self.logger.info("开始加载土壤数据...")
        
        # 加载数据
        data = pd.read_csv(os.path.join(self.data_dir, 'soil_training.csv'))
        
        # 预处理数据
        X, y = self._preprocess_data(data, 'soil')
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # 应用采样
        if self.use_oversampling:
            # 统计每个类别的样本数
            class_counts = Counter(y_train)
            self.logger.info(f"原始类别分布: {class_counts}")
            
            # 创建采样pipeline
            sampling_pipeline = self._create_sampling_pipeline(class_counts, is_soil=True)
            
            # 应用采样
            with tqdm(total=1, desc="数据采样") as pbar:
                X_train, y_train = sampling_pipeline.fit_resample(X_train, y_train)
                pbar.update(1)
                
            self.logger.info(f"采样后类别分布: {Counter(y_train)}")
        
        load_time = time.time() - start_time
        self.logger.info(f"数据加载完成，耗时: {load_time:.2f}秒")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
    def load_groundwater_data(self) -> Dict[str, np.ndarray]:
        """
        加载地下水数据
        
        Returns:
            包含训练和测试数据的字典
        """
        # 加载数据
        data = pd.read_csv(os.path.join(self.data_dir, 'groundwater_training.csv'))
        
        # 预处理数据
        X, y = self._preprocess_data(data, 'groundwater')
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # 应用采样
        if self.use_oversampling:
            # 统计每个类别的样本数
            class_counts = Counter(y_train)
            self.logger.info(f"原始类别分布: {class_counts}")
            
            # 创建采样pipeline
            sampling_pipeline = self._create_sampling_pipeline(class_counts, is_soil=False)
            
            # 应用采样
            X_train, y_train = sampling_pipeline.fit_resample(X_train, y_train)
            self.logger.info(f"采样后类别分布: {Counter(y_train)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }

    def update_feature_importance(self, importance_dict: Dict[str, float], dataset_type: str):
        """
        更新特征重要性
        
        Args:
            importance_dict: 特征重要性字典
            dataset_type: 数据集类型
        """
        self.feature_importance[dataset_type] = importance_dict 