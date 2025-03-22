"""
数据分析模块
分析训练数据的特征分布、缺失值、相关性等
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib as mpl
import platform
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

class DataAnalyzer:
    """数据分析器类"""
    
    def __init__(self, output_dir: str = "output/analysis"):
        """
        初始化数据分析器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.logger = self._setup_logging()
        
        # 设置全局图表样式
        self._setup_plot_style()
        
    def _setup_plot_style(self):
        """设置全局图表样式"""
        # 设置seaborn样式
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 设置全局颜色方案
        self.colors = sns.color_palette("husl", 8)
        
        # 设置中文字体
        self._setup_chinese_font()
        
        # 设置全局图表参数
        plt.rcParams.update({
            'figure.figsize': (15, 8),  # 默认图表大小
            'figure.dpi': 300,          # 图表DPI
            'axes.titlesize': 16,       # 标题字体大小
            'axes.labelsize': 14,       # 轴标签字体大小
            'xtick.labelsize': 12,      # x轴刻度标签大小
            'ytick.labelsize': 12,      # y轴刻度标签大小
            'axes.titleweight': 'bold', # 标题字体粗细
            'axes.grid': True,          # 显示网格
            'grid.alpha': 0.3,          # 网格透明度
            'grid.linestyle': '--'      # 网格线样式
        })
        
    def _setup_chinese_font(self):
        """设置中文字体"""
        # 获取系统信息
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = [
                'PingFang HK',  # 苹方
                'STHeiti',      # 华文黑体
                'STKaiti',      # 华文楷体
                'STSong',       # 华文宋体
                'STFangsong',   # 华文仿宋
                'Arial Unicode MS'  # 作为后备字体
            ]
        elif system == 'Windows':
            plt.rcParams['font.sans-serif'] = [
                'Microsoft YaHei',  # 微软雅黑
                'SimHei',           # 黑体
                'SimSun',           # 宋体
                'KaiTi'             # 楷体
            ]
        else:  # Linux
            plt.rcParams['font.sans-serif'] = [
                'WenQuanYi Micro Hei',  # 文泉驿微米黑
                'Noto Sans CJK SC',     # Noto Sans CJK
                'Droid Sans Fallback'    # Droid Sans
            ]
        
        # 设置负号显示
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置全局字体大小
        mpl.rcParams['font.size'] = 12
        
        # 验证字体是否可用
        self._verify_fonts()
        
    def _verify_fonts(self):
        """验证字体是否可用"""
        # 获取系统所有字体
        system_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 检查每个中文字体是否可用
        available_fonts = []
        for font in plt.rcParams['font.sans-serif']:
            if font in system_fonts:
                available_fonts.append(font)
                self.logger.info(f"字体 {font} 可用")
            else:
                self.logger.warning(f"字体 {font} 不可用")
        
        if not available_fonts:
            self.logger.error("没有可用的中文字体！")
            raise RuntimeError("没有可用的中文字体，请安装中文字体后重试")
        
        # 更新字体列表，只使用可用的字体
        plt.rcParams['font.sans-serif'] = available_fonts
        self.logger.info(f"使用以下字体: {', '.join(available_fonts)}")
        
    def _setup_logging(self):
        """设置日志记录器"""
        import logging
        from datetime import datetime
        
        # 创建日志目录
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"data_analysis_{timestamp}.log")
        
        # 配置日志记录器
        logger = logging.getLogger("data_analyzer")
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def analyze_dataset(self, data_path: str, dataset_type: str) -> None:
        """
        分析数据集
        
        Args:
            data_path: 数据文件路径
            dataset_type: 数据集类型 ('soil' 或 'groundwater')
        """
        try:
            # 创建输出目录
            dataset_output_dir = os.path.join(self.output_dir, dataset_type)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # 读取数据
            self.logger.info(f"读取{dataset_type}数据集: {data_path}")
            df = pd.read_csv(data_path, encoding='utf-8')
            
            # 1. 基本统计信息
            self._analyze_basic_stats(df, dataset_output_dir)
            
            # 2. 缺失值分析
            self._analyze_missing_values(df, dataset_output_dir)
            
            # 3. 特征分布分析
            self._analyze_feature_distributions(df, dataset_output_dir)
            
            # 4. 相关性分析
            self._analyze_correlations(df, dataset_output_dir)
            
            # 5. 目标变量分析
            self._analyze_target_variable(df, dataset_output_dir)
            
            # 6. 生成分析报告
            self._generate_report(df, dataset_output_dir, dataset_type)
            
            self.logger.info(f"{dataset_type}数据集分析完成")
            
        except Exception as e:
            self.logger.error(f"分析{dataset_type}数据集时发生错误: {str(e)}")
            raise
    
    def _analyze_basic_stats(self, df: pd.DataFrame, output_dir: str) -> None:
        """分析基本统计信息"""
        # 计算基本统计量
        stats = df.describe()
        
        # 保存统计结果
        stats.to_csv(os.path.join(output_dir, 'basic_statistics.csv'))
        
        # 记录信息
        self.logger.info(f"数据集基本信息:")
        self.logger.info(f"样本数量: {len(df)}")
        self.logger.info(f"特征数量: {len(df.columns)}")
        self.logger.info(f"特征列表: {', '.join(df.columns)}")
    
    def _analyze_missing_values(self, df: pd.DataFrame, output_dir: str) -> None:
        """分析缺失值"""
        # 计算缺失值统计
        missing_stats = pd.DataFrame({
            '缺失值数量': df.isnull().sum(),
            '缺失值比例': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        # 保存缺失值统计
        missing_stats.to_csv(os.path.join(output_dir, 'missing_values.csv'))
        
        # 绘制缺失值热力图
        plt.figure()
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('缺失值分布热力图', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'), 
                   bbox_inches='tight')
        plt.close()
        
        # 记录信息
        self.logger.info("缺失值分析:")
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                self.logger.info(f"{col}: {missing_count}个缺失值 ({missing_stats.loc[col, '缺失值比例']}%)")
    
    def _analyze_feature_distributions(self, df: pd.DataFrame, output_dir: str) -> None:
        """分析特征分布"""
        # 创建特征分布图目录
        dist_dir = os.path.join(output_dir, 'feature_distributions')
        os.makedirs(dist_dir, exist_ok=True)
        
        # 分析所有特征
        for col in df.columns:
            # 计算特征分布并按索引排序
            value_counts = df[col].value_counts().sort_index()
            
            # 绘制条形图
            plt.figure()
            
            # 创建条形图，使用 hue 参数来设置颜色
            data = pd.DataFrame({
                'category': value_counts.index,
                'count': value_counts.values,
                'hue': value_counts.index
            })
            
            # 使用自定义颜色方案
            n_colors = len(value_counts)
            if n_colors <= 8:
                palette = self.colors[:n_colors]
            else:
                palette = sns.color_palette("husl", n_colors)
            
            # 创建条形图
            bars = sns.barplot(data=data, x='category', y='count', hue='hue',
                             palette=palette, alpha=0.8, legend=False)
            
            # 设置标题和标签
            plt.title(f'{col}分布')
            plt.xlabel(col)
            plt.ylabel('频数')
            
            # 优化x轴标签
            plt.xticks(rotation=45, ha='right')
            
            # 获取y轴的最大值
            y_max = max(value_counts.values)
            
            # 添加数值标签
            for i, v in enumerate(value_counts.values):
                plt.text(i, v, str(v), 
                        ha='center', va='bottom',
                        color='#2c3e50', fontweight='bold')
            
            # 调整y轴上限，为标签留出空间
            plt.ylim(0, y_max * 1.1)
            
            # 优化布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(os.path.join(dist_dir, f'{col}_distribution.png'), 
                       bbox_inches='tight')
            plt.close()
            
            # 保存统计信息
            value_counts.to_csv(os.path.join(dist_dir, f'{col}_statistics.csv'))
    
    def _analyze_correlations(self, df: pd.DataFrame, output_dir: str) -> None:
        """分析特征相关性"""
        # 创建数据副本
        df_processed = df.copy()
        
        # 1. 处理分类变量
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df_processed[column] = pd.factorize(df_processed[column])[0]
        
        # 2. 处理缺失值
        # 对于缺失率超过50%的特征，暂时不参与相关性分析
        missing_threshold = 0.5
        high_missing_features = df_processed.columns[df_processed.isnull().mean() > missing_threshold].tolist()
        df_processed = df_processed.drop(columns=high_missing_features)
        
        # 使用KNN填充其他缺失值
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df_processed = pd.DataFrame(imputer.fit_transform(df_processed), columns=df_processed.columns)
        
        # 3. 处理异常值
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            Q1 = df_processed[column].quantile(0.25)
            Q3 = df_processed[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_processed[column] = df_processed[column].clip(lower_bound, upper_bound)
        
        # 4. 标准化特征
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
        
        # 5. 计算相关性矩阵
        corr_matrix = df_processed.corr()
        
        # 保存相关性矩阵
        corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
        
        # 绘制相关性热力图
        plt.figure(figsize=(15, 12))  # 相关性矩阵需要更大的尺寸
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', annot_kws={'size': 8})
        plt.title('特征相关性热力图', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), 
                   bbox_inches='tight')
        plt.close()
        
        # 记录高相关性特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:  # 相关系数阈值
                    high_corr_pairs.append({
                        '特征1': corr_matrix.columns[i],
                        '特征2': corr_matrix.columns[j],
                        '相关系数': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            self.logger.info("高相关性特征对:")
            for pair in high_corr_pairs:
                self.logger.info(f"{pair['特征1']} 和 {pair['特征2']}: {pair['相关系数']:.3f}")
            
            # 记录被排除的高缺失率特征
            if high_missing_features:
                self.logger.info("\n由于缺失率过高而被排除的特征:")
                for feature in high_missing_features:
                    missing_rate = df[feature].isnull().mean() * 100
                    self.logger.info(f"{feature}: {missing_rate:.2f}% 缺失")
    
    def _analyze_target_variable(self, df: pd.DataFrame, output_dir: str) -> None:
        """分析目标变量"""
        if '修复技术' in df.columns:
            # 计算目标变量分布
            target_counts = df['修复技术'].value_counts()
            target_percentages = (target_counts / len(df) * 100).round(2)
            
            # 保存目标变量统计
            target_stats = pd.DataFrame({
                '数量': target_counts,
                '百分比': target_percentages
            })
            target_stats.to_csv(os.path.join(output_dir, 'target_variable_statistics.csv'))
            
            # 创建数据框
            data = pd.DataFrame({
                'category': target_counts.index,
                'count': target_counts.values,
                'hue': target_counts.index
            })
            
            # 使用自定义颜色方案
            n_colors = len(target_counts)
            if n_colors <= 8:
                palette = self.colors[:n_colors]
            else:
                palette = sns.color_palette("husl", n_colors)
            
            # 绘制目标变量分布图
            plt.figure()
            bars = sns.barplot(data=data, x='category', y='count', hue='hue',
                             palette=palette, alpha=0.8, legend=False)
            plt.title('修复技术分布')
            plt.xlabel('修复技术')
            plt.ylabel('数量')
            plt.xticks(rotation=45, ha='right')
            
            # 添加数值标签
            for i, v in enumerate(target_counts.values):
                plt.text(i, v, str(v), ha='center', va='bottom',
                        color='#2c3e50', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'target_variable_distribution.png'), 
                       bbox_inches='tight')
            plt.close()
            
            # 记录信息
            self.logger.info("目标变量分布:")
            for tech, count in target_counts.items():
                self.logger.info(f"{tech}: {count}个样本 ({target_percentages[tech]}%)")
    
    def _generate_report(self, df: pd.DataFrame, output_dir: str, dataset_type: str) -> None:
        """生成分析报告"""
        report = {
            '数据集信息': {
                '样本数量': len(df),
                '特征数量': len(df.columns),
                '特征列表': df.columns.tolist(),
                '分析时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            '缺失值统计': {
                col: {
                    '缺失值数量': int(df[col].isnull().sum()),
                    '缺失值比例': float((df[col].isnull().sum() / len(df) * 100).round(2))
                }
                for col in df.columns
                if df[col].isnull().sum() > 0
            }
        }
        
        # 添加目标变量统计（如果存在）
        if '修复技术' in df.columns:
            target_counts = df['修复技术'].value_counts()
            report['目标变量统计'] = {
                '修复技术': {
                    tech: {
                        '数量': int(count),
                        '百分比': round(count / len(df) * 100, 2)
                    }
                    for tech, count in target_counts.items()
                }
            }
        
        # 保存报告
        report_path = os.path.join(output_dir, 'analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        self.logger.info(f"分析报告已保存至: {report_path}")

def main():
    """主函数"""
    # 获取项目根目录
    project_root = str(Path(__file__).parent.parent.parent)
    
    # 设置基础目录
    data_dir = os.path.join(project_root, "data", "training")
    output_dir = os.path.join(project_root, "output", "analysis")
    
    # 创建分析器实例
    analyzer = DataAnalyzer(output_dir)
    
    # 分析土壤数据
    soil_data_file = os.path.join(data_dir, "soil_training.csv")
    if os.path.exists(soil_data_file):
        analyzer.analyze_dataset(soil_data_file, "soil")
    else:
        print(f"[WARNING] 土壤训练数据文件不存在: {soil_data_file}")
    
    # 分析地下水数据
    groundwater_data_file = os.path.join(data_dir, "groundwater_training.csv")
    if os.path.exists(groundwater_data_file):
        analyzer.analyze_dataset(groundwater_data_file, "groundwater")
    else:
        print(f"[WARNING] 地下水训练数据文件不存在: {groundwater_data_file}")
    
    print("[INFO] 数据分析完成！")

if __name__ == "__main__":
    main() 