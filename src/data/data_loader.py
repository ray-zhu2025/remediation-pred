"""
数据加载器类
用于加载和处理训练和测试数据
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class DataLoader:
    """数据加载器类"""
    
    def __init__(self, data_dir: str = 'data/training', test_size: float = 0.2, random_state: int = 42):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_soil_data(self) -> Dict[str, np.ndarray]:
        """
        加载土壤数据
        
        Returns:
            包含训练和测试数据的字典
        """
        # 加载数据
        data = pd.read_csv(os.path.join(self.data_dir, 'soil_training.csv'))
        
        # 分离特征和标签
        y = self.label_encoder.fit_transform(data['修复技术'])
        X = data.drop(['修复技术', '修复面积', '修复土方量', '修复时间', '修复成本'], axis=1)
        
        # 识别数值列和分类列
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # 创建预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ])
        
        # 转换数据
        X = preprocessor.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
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
        
        # 分离特征和标签
        y = self.label_encoder.fit_transform(data['修复技术'])
        X = data.drop('修复技术', axis=1)
        
        # 识别数值列和分类列
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # 创建预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ])
        
        # 转换数据
        X = preprocessor.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        } 