"""
数据处理模块
提供数据加载、预处理和缺失值处理功能
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Tuple, List, Dict, Union

class DataProcessor:
    def __init__(self, n_neighbors: int = 1):
        """
        初始化数据处理器
        
        Args:
            n_neighbors: KNN插值法的邻居数
        """
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        self.zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的数据DataFrame
        """
        return pd.read_csv(file_path, encoding='utf-8')
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            特征矩阵X和目标变量y
        """
        # 找到目标变量的索引
        target_index = int(np.where(df.columns.values == '修复技术')[0])
        
        # 提取特征和目标
        X = df.iloc[:, 2:target_index]
        y = df['修复技术']
        
        # 处理缺失值
        X = self.knn_imputer.fit_transform(X)
        
        return X, y
    
    def prepare_prediction_data(self, 
                              pred_df: pd.DataFrame, 
                              train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备预测数据
        
        Args:
            pred_df: 预测数据DataFrame
            train_df: 训练数据DataFrame
            
        Returns:
            特征矩阵X、标识信息I和设计参数D
        """
        # 找到目标变量的索引
        target_index = int(np.where(train_df.columns.values == '修复技术')[0])
        
        # 提取特征
        X = pred_df.iloc[:, 2:target_index]
        I = pred_df.iloc[:, :2]  # 标识信息
        
        # 处理设计参数
        D = pred_df.iloc[:, target_index:]
        D = self.zero_imputer.fit_transform(D)
        
        # 处理特征缺失值
        X = self.knn_imputer.fit_transform(
            X._append(train_df.iloc[:, 2:target_index], ignore_index=True)
        )[:X.shape[0], :]
        
        return X, I, D
    
    def save_results(self, 
                    results: List[pd.DataFrame], 
                    file_paths: List[str]) -> None:
        """
        保存结果
        
        Args:
            results: 结果DataFrame列表
            file_paths: 保存路径列表
        """
        for result, path in zip(results, file_paths):
            result.to_csv(path, index=False, sep=',', encoding='utf-8-sig')
            
    def calculate_costs_and_time(self, 
                               predictions: np.ndarray, 
                               design_params: np.ndarray,
                               prices: List[float],
                               periods: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算修复成本和周期
        
        Args:
            predictions: 预测结果
            design_params: 设计参数
            prices: 修复成本单价列表
            periods: 修复周期列表
            
        Returns:
            修复成本数组和修复周期数组
        """
        costs = design_params[:, 0] * np.array([prices[i-1] for i in predictions])
        time = np.array([periods[i-1] for i in predictions])
        
        return costs, time 