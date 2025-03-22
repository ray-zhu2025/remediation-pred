"""
基础模型类
包含土壤和地下水模型的共同功能
"""

# 标准库导入
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# 第三方库导入
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 本地应用导入
from src.process.data_processor import DataProcessor
from src.utils.logging import setup_logging

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

class NumpyEncoder(json.JSONEncoder):
    """处理NumPy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class BaseModel:
    """基础模型类"""
    
    def __init__(self, config_path: str, use_hyperopt: bool = False, search_method: str = 'grid'):
        """
        初始化基础模型
        
        Args:
            config_path: 配置文件路径
            use_hyperopt: 是否使用超参数优化
            search_method: 超参数搜索方法，'grid' 或 'random'
        """
        self.data_processor = DataProcessor()
        self.config = self._load_config(config_path)
        self.use_hyperopt = use_hyperopt
        self.search_method = search_method
        self.models = self._initialize_models()
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        self.label_encoders = {}
        self.logger = setup_logging()
        self.feature_names = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.best_params = {}
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_param_grid(self) -> Dict:
        """获取超参数搜索空间"""
        return {
            'DecisionTreeClassifier': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'ComplementNB': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    
    def _initialize_models(self) -> List[Union[GridSearchCV, RandomizedSearchCV, object]]:
        """初始化机器学习模型"""
        base_models = [
            ('DecisionTreeClassifier', tree.DecisionTreeClassifier(random_state=30)),
            ('ComplementNB', ComplementNB()),
            ('RandomForestClassifier', RandomForestClassifier(random_state=42))
        ]
        
        if self.use_hyperopt:
            param_grid = self._get_param_grid()
            optimized_models = []
            
            for name, model in base_models:
                if self.search_method == 'grid':
                    search = GridSearchCV(
                        model, 
                        param_grid[name],
                        cv=5,
                        scoring='f1_weighted',
                        n_jobs=-1,
                        verbose=1
                    )
                else:  # random search
                    search = RandomizedSearchCV(
                        model,
                        param_grid[name],
                        n_iter=20,
                        cv=5,
                        scoring='f1_weighted',
                        n_jobs=-1,
                        random_state=42,
                        verbose=1
                    )
                optimized_models.append(search)
            return optimized_models
        else:
            return [
                tree.DecisionTreeClassifier(criterion="entropy", random_state=30),
                ComplementNB(),
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            ]
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据，将分类变量转换为数值
        
        Args:
            df: 输入数据框
            
        Returns:
            处理后的数据框
        """
        df_processed = df.copy()
        
        # 识别分类列
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # 对每个分类列进行编码
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_processed[column] = self.label_encoders[column].fit_transform(df_processed[column])
            else:
                df_processed[column] = self.label_encoders[column].transform(df_processed[column])
        
        # 保存特征名称
        self.feature_names = df_processed.columns.tolist()
        
        return df_processed
    
    def train(self, train_data_path: str) -> None:
        """
        训练模型
        
        Args:
            train_data_path: 训练数据路径
        """
        # 加载和处理训练数据
        train_df = self.data_processor.load_data(train_data_path)
        train_df = self._preprocess_data(train_df)
        X, y = self.data_processor.prepare_training_data(train_df)
        
        # 划分训练集、验证集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # 保存数据集
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        
        # 标准化特征
        X_train_scaled_standard = self.scalers['standard'].fit_transform(X_train)
        X_train_scaled_minmax = self.scalers['minmax'].fit_transform(X_train)
        X_val_scaled_standard = self.scalers['standard'].transform(X_val)
        X_val_scaled_minmax = self.scalers['minmax'].transform(X_val)
        
        # 训练所有模型
        for i, model in enumerate(self.models):
            if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
                # 使用超参数优化的模型
                if isinstance(self.models[i], ComplementNB):
                    model.fit(X_train_scaled_minmax, y_train)
                    val_score = model.score(X_val_scaled_minmax, y_val)
                else:
                    model.fit(X_train_scaled_standard, y_train)
                    val_score = model.score(X_val_scaled_standard, y_val)
                
                # 保存最佳参数和验证集得分
                self.best_params[type(model.estimator).__name__] = model.best_params_
                self.logger.info(f"{type(model.estimator).__name__} 最佳参数: {model.best_params_}")
                self.logger.info(f"{type(model.estimator).__name__} 验证集得分: {val_score:.4f}")
            else:
                # 使用默认参数的模型
                if isinstance(model, ComplementNB):
                    model.fit(X_train_scaled_minmax, y_train)
                    val_score = model.score(X_val_scaled_minmax, y_val)
                else:
                    model.fit(X_train_scaled_standard, y_train)
                    val_score = model.score(X_val_scaled_standard, y_val)
                self.logger.info(f"{type(model).__name__} 验证集得分: {val_score:.4f}")
        
        self.logger.info(f"{self.__class__.__name__} 训练完成")
        
    def predict(self, pred_data_path: str, output_dir: str) -> None:
        """
        进行预测
        
        Args:
            pred_data_path: 预测数据路径
            output_dir: 输出目录
        """
        # 加载和处理预测数据
        pred_df = self.data_processor.load_data(pred_data_path)
        pred_df = self._preprocess_data(pred_df)
        
        # 确保预测数据包含相同的特征
        if self.feature_names is None:
            raise ValueError("未指定特征名")
            
        # 选择相同的特征
        pred_df = pred_df[self.feature_names]
        
        # 准备预测数据
        X_pred, I_pred, D_pred = self.data_processor.prepare_prediction_data(
            pred_df, self.data_processor.load_data(self.config['train_data_path'])
        )
        
        # 标准化特征
        X_scaled_standard = self.scalers['standard'].transform(X_pred)
        X_scaled_minmax = self.scalers['minmax'].transform(X_pred)
        
        # 使用所有模型进行预测
        predictions = []
        for model in self.models:
            if isinstance(model, ComplementNB):
                pred = model.predict(X_scaled_minmax)
            else:
                pred = model.predict(X_scaled_standard)
            predictions.append(pred)
        
        # 计算修复成本和周期
        results = []
        for model_pred in predictions:
            model_results = []
            for i, pred in enumerate(model_pred):
                costs, time = self.data_processor.calculate_costs_and_time(
                    np.array([pred]), D_pred[i:i+1],
                    self.config['prices'],
                    self.config['periods']
                )
                
                # 整理结果
                result = pd.DataFrame(np.column_stack((
                    I_pred[i:i+1], np.array([pred]), X_pred[i:i+1, 2], D_pred[i:i+1], costs, time
                )))
                model_results.append(result)
            
            # 合并该模型的所有预测结果
            if model_results:
                combined_result = pd.concat(model_results, ignore_index=True)
                results.append(combined_result)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存预测结果到主输出目录
        output_paths = [
            os.path.join(output_dir, f'prediction_{type(model).__name__}.csv')
            for model in self.models
        ]
        self.data_processor.save_results(results, output_paths)
    
    def _calculate_binary_metrics(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> Dict:
        """
        计算二分类评估指标
        
        Args:
            y_true_binary: 真实标签的二分类表示
            y_pred_binary: 预测标签的二分类表示
            
        Returns:
            包含各项评估指标的字典
        """
        # 计算基本指标
        tech_accuracy = accuracy_score(y_true_binary, y_pred_binary)
        tech_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        tech_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        tech_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # 计算混淆矩阵元素
        tn = np.sum((~y_true_binary) & (~y_pred_binary))
        fp = np.sum((~y_true_binary) & y_pred_binary)
        fn = np.sum(y_true_binary & (~y_pred_binary))
        tp = np.sum(y_true_binary & y_pred_binary)
        
        # 计算特异度
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            '正确预测为该技术': tp,
            '正确预测为其他技术': tn,
            '错误预测为该技术': fp,
            '错误预测为其他技术': fn,
            '准确率': tech_accuracy,
            '精确率': tech_precision,
            '召回率': tech_recall,
            'F1分数': tech_f1,
            '特异度': specificity
        }
    
    def _save_confusion_matrix_plot(self, cm: np.ndarray, unique_labels: np.ndarray, 
                                  model_name: str, eval_output_dir: str) -> None:
        """
        保存混淆矩阵图
        
        Args:
            cm: 混淆矩阵
            unique_labels: 唯一的标签值
            model_name: 模型名称
            eval_output_dir: 评估结果输出目录
        """
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="white")
        
        # 绘制混淆矩阵热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_labels,
                   yticklabels=unique_labels)
        
        # 设置标题和标签
        plt.title(f'{model_name}修复技术预测混淆矩阵', fontsize=14, pad=20)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存混淆矩阵图
        plt.savefig(os.path.join(eval_output_dir, f'confusion_matrix_{model_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_tech_metrics(self, label: int, metrics: Dict) -> None:
        """
        打印修复技术的评估指标
        
        Args:
            label: 修复技术标签
            metrics: 评估指标字典
        """
        print(f"\n修复技术 {label}:")
        print(f"样本统计:")
        print(f"  - 总样本量: {metrics['总样本量']}")
        print(f"  - 正确预测为该技术: {metrics['正确预测为该技术']}")
        print(f"  - 正确预测为其他技术: {metrics['正确预测为其他技术']}")
        print(f"  - 错误预测为该技术: {metrics['错误预测为该技术']}")
        print(f"  - 错误预测为其他技术: {metrics['错误预测为其他技术']}")
        print(f"评估指标:")
        print(f"  - 准确率: {metrics['准确率']:.4f}")
        print(f"  - 精确率: {metrics['精确率']:.4f}")
        print(f"  - 召回率: {metrics['召回率']:.4f}")
        print(f"  - F1分数: {metrics['F1分数']:.4f}")
        print(f"  - 特异度: {metrics['特异度']:.4f}")
    
    def evaluate(self, output_dir: str) -> None:
        """
        评估模型性能
        
        Args:
            output_dir: 输出目录
        """
        try:
            if self.test_data is None:
                raise ValueError("模型尚未训练，请先训练模型")
                
            X_test, y_test = self.test_data
            
            # 标准化特征
            X_test_scaled_standard = self.scalers['standard'].transform(X_test)
            X_test_scaled_minmax = self.scalers['minmax'].transform(X_test)
            
            # 创建评估输出目录
            eval_output_dir = os.path.join(output_dir, 'evaluation')
            os.makedirs(eval_output_dir, exist_ok=True)
            
            # 评估所有模型
            all_results = []
            remediation_metrics = {}
            
            for model in self.models:
                # 获取实际的模型（如果是搜索器，则获取最佳估计器）
                if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
                    actual_model = model.best_estimator_
                    model_name = type(actual_model).__name__
                else:
                    actual_model = model
                    model_name = type(model).__name__
                
                # 根据模型类型选择特征缩放方法
                X_test_scaled = X_test_scaled_minmax if isinstance(actual_model, ComplementNB) else X_test_scaled_standard
                
                # 预测
                y_pred = actual_model.predict(X_test_scaled)
                
                # 计算整体评估指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                total_samples = len(y_test)
                
                # 记录评估结果
                self.logger.info(f"{self.__class__.__name__} {model_name} 测试集评估结果:")
                self.logger.info(f"总样本量: {total_samples}")
                self.logger.info(f"准确率: {accuracy:.4f}")
                self.logger.info(f"精确率: {precision:.4f}")
                self.logger.info(f"召回率: {recall:.4f}")
                self.logger.info(f"F1分数: {f1:.4f}")
                
                # 保存整体结果
                all_results.append({
                    '模型': model_name,
                    '总样本量': total_samples,
                    '准确率': accuracy,
                    '精确率': precision,
                    '召回率': recall,
                    'F1分数': f1
                })
                
                # 计算混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                
                # 获取唯一的类别标签
                unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                
                # 计算每个修复技术的样本量
                tech_samples = {label: np.sum(y_test == label) for label in unique_labels}
                
                # 计算每个修复技术的评估指标
                for label in unique_labels:
                    y_true_binary = (y_test == label)
                    y_pred_binary = (y_pred == label)
                    
                    # 计算该修复技术的指标
                    metrics = self._calculate_binary_metrics(y_true_binary, y_pred_binary)
                    metrics.update({
                        '模型': model_name,
                        '修复技术': label,
                        '总样本量': total_samples
                    })
                    
                    # 存储该修复技术的评估指标
                    if label not in remediation_metrics:
                        remediation_metrics[label] = []
                    remediation_metrics[label].append(metrics)
                
                # 保存混淆矩阵图
                self._save_confusion_matrix_plot(cm, unique_labels, model_name, eval_output_dir)
                
                # 生成分类报告
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # 保存评估结果
                evaluation_results = {
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report,
                    'remediation_metrics': {
                        str(label): metrics for label, metrics in remediation_metrics.items()
                    }
                }
                
                # 将评估结果保存为JSON文件
                with open(os.path.join(eval_output_dir, f'evaluation_results_{model_name}.json'), 
                         'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                
                # 打印评估报告
                print(f"\n{model_name}分类评估报告:")
                print(f"各修复技术样本量:")
                for label, count in sorted(tech_samples.items()):
                    print(f"修复技术 {label}: {count} 个样本")
                print("\n分类报告:")
                print(classification_report(y_test, y_pred))
                print("\n各修复技术评估指标:")
                
                # 打印每个修复技术的评估指标
                for label in sorted(remediation_metrics.keys()):
                    self._print_tech_metrics(label, remediation_metrics[label][-1])
                
                print(f"\n混淆矩阵图已保存至: {os.path.join(eval_output_dir, f'confusion_matrix_{model_name}.png')}")
                print(f"评估结果已保存至: {os.path.join(eval_output_dir, f'evaluation_results_{model_name}.json')}")
                print("\n" + "="*50)
            
            # 保存所有模型的评估结果比较
            report_path = os.path.join(eval_output_dir, 'models_comparison.csv')
            report_df = pd.DataFrame(all_results)
            report_df.to_csv(report_path, sep=',', encoding='utf-8-sig', index=False)
            print(f"\n模型比较结果已保存至: {report_path}")
            
            # 如果使用了超参数优化，保存最佳参数
            if self.use_hyperopt:
                best_params_path = os.path.join(eval_output_dir, 'best_parameters.json')
                with open(best_params_path, 'w', encoding='utf-8') as f:
                    json.dump(self.best_params, f, ensure_ascii=False, indent=4)
                print(f"\n最佳超参数已保存至: {best_params_path}")
            
        except Exception as e:
            self.logger.error(f"评估{self.__class__.__name__}时发生错误: {str(e)}")
            raise 