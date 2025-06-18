"""
代码冗余分析工具
用于自动检测和统计项目中的代码冗余问题
"""
import os
import re
import ast
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RedundancyAnalyzer:
    """代码冗余分析器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        self.redundancy_results = defaultdict(list)
        
    def analyze_project(self) -> Dict:
        """分析整个项目的代码冗余"""
        self.logger.info("开始分析项目代码冗余...")
        
        # 获取所有Python文件
        python_files = self._get_python_files()
        self.logger.info(f"找到 {len(python_files)} 个Python文件")
        
        # 分析各种冗余问题
        results = {
            'duplicate_classes': self._analyze_duplicate_classes(python_files),
            'duplicate_methods': self._analyze_duplicate_methods(python_files),
            'duplicate_imports': self._analyze_duplicate_imports(python_files),
            'duplicate_strings': self._analyze_duplicate_strings(python_files),
            'duplicate_patterns': self._analyze_duplicate_patterns(python_files),
            'summary': {}
        }
        
        # 生成汇总统计
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _get_python_files(self) -> List[str]:
        """获取项目中的所有Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # 跳过虚拟环境和缓存目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', '.venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_duplicate_classes(self, files: List[str]) -> List[Dict]:
        """分析重复的类定义"""
        class_definitions = {}
        duplicates = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        class_body = ast.unparse(node)
                        
                        if class_name in class_definitions:
                            # 检查是否真的重复
                            existing_body = class_definitions[class_name]['body']
                            if self._calculate_similarity(class_body, existing_body) > 0.8:
                                duplicates.append({
                                    'type': 'duplicate_class',
                                    'class_name': class_name,
                                    'files': [class_definitions[class_name]['file'], file_path],
                                    'similarity': self._calculate_similarity(class_body, existing_body)
                                })
                        else:
                            class_definitions[class_name] = {
                                'file': file_path,
                                'body': class_body
                            }
                            
            except Exception as e:
                self.logger.warning(f"解析文件 {file_path} 时出错: {e}")
        
        return duplicates
    
    def _analyze_duplicate_methods(self, files: List[str]) -> List[Dict]:
        """分析重复的方法定义"""
        method_definitions = {}
        duplicates = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        method_name = node.name
                        method_body = ast.unparse(node)
                        
                        # 创建方法签名
                        method_signature = f"{method_name}_{len(method_body)}"
                        
                        if method_signature in method_definitions:
                            existing_body = method_definitions[method_signature]['body']
                            similarity = self._calculate_similarity(method_body, existing_body)
                            
                            if similarity > 0.7:  # 70%相似度阈值
                                duplicates.append({
                                    'type': 'duplicate_method',
                                    'method_name': method_name,
                                    'files': [method_definitions[method_signature]['file'], file_path],
                                    'similarity': similarity
                                })
                        else:
                            method_definitions[method_signature] = {
                                'file': file_path,
                                'body': method_body
                            }
                            
            except Exception as e:
                self.logger.warning(f"解析文件 {file_path} 时出错: {e}")
        
        return duplicates
    
    def _analyze_duplicate_imports(self, files: List[str]) -> List[Dict]:
        """分析重复的导入语句"""
        import_patterns = defaultdict(list)
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找导入语句
                import_lines = re.findall(r'^from\s+(\S+)\s+import\s+(.+)$', content, re.MULTILINE)
                for module, imports in import_lines:
                    import_pattern = f"from {module} import {imports}"
                    import_patterns[import_pattern].append(file_path)
                    
            except Exception as e:
                self.logger.warning(f"分析导入语句时出错: {e}")
        
        # 找出重复的导入模式
        duplicates = []
        for pattern, files_list in import_patterns.items():
            if len(files_list) > 1:
                duplicates.append({
                    'type': 'duplicate_import',
                    'pattern': pattern,
                    'files': files_list,
                    'count': len(files_list)
                })
        
        return duplicates
    
    def _analyze_duplicate_strings(self, files: List[str]) -> List[Dict]:
        """分析重复的字符串常量"""
        string_patterns = defaultdict(list)
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找字符串常量
                strings = re.findall(r'["\']([^"\']{10,})["\']', content)
                for string in strings:
                    if len(string) > 10:  # 只考虑长度大于10的字符串
                        string_patterns[string].append(file_path)
                        
            except Exception as e:
                self.logger.warning(f"分析字符串常量时出错: {e}")
        
        # 找出重复的字符串
        duplicates = []
        for string, files_list in string_patterns.items():
            if len(files_list) > 1:
                duplicates.append({
                    'type': 'duplicate_string',
                    'string': string[:50] + '...' if len(string) > 50 else string,
                    'files': files_list,
                    'count': len(files_list)
                })
        
        return duplicates
    
    def _analyze_duplicate_patterns(self, files: List[str]) -> List[Dict]:
        """分析重复的代码模式"""
        patterns = [
            {
                'name': 'timestamp_generation',
                'pattern': r'datetime\.now\(\)\.strftime\(["\']%Y%m%d_%H%M%S["\']\)',
                'description': '时间戳生成模式'
            },
            {
                'name': 'directory_creation',
                'pattern': r'os\.makedirs\([^,]+,\s*exist_ok=True\)',
                'description': '目录创建模式'
            },
            {
                'name': 'logger_creation',
                'pattern': r'logging\.getLogger\([^)]+\)',
                'description': '日志器创建模式'
            },
            {
                'name': 'accuracy_score_import',
                'pattern': r'from sklearn\.metrics import.*accuracy_score',
                'description': 'sklearn指标导入模式'
            }
        ]
        
        duplicates = []
        for pattern_info in patterns:
            pattern_files = []
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    matches = re.findall(pattern_info['pattern'], content)
                    if matches:
                        pattern_files.append({
                            'file': file_path,
                            'matches': len(matches)
                        })
                        
                except Exception as e:
                    self.logger.warning(f"分析模式时出错: {e}")
            
            if len(pattern_files) > 1:
                duplicates.append({
                    'type': 'duplicate_pattern',
                    'pattern_name': pattern_info['name'],
                    'description': pattern_info['description'],
                    'files': pattern_files,
                    'total_matches': sum(f['matches'] for f in pattern_files)
                })
        
        return duplicates
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if text1 == text2:
            return 1.0
        
        # 简单的相似度计算（基于公共字符）
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_summary(self, results: Dict) -> Dict:
        """生成汇总统计"""
        summary = {
            'total_duplicates': 0,
            'duplicate_types': {},
            'files_affected': set(),
            'severity_levels': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        for duplicate_type, duplicates in results.items():
            if duplicate_type == 'summary':
                continue
                
            summary['duplicate_types'][duplicate_type] = len(duplicates)
            summary['total_duplicates'] += len(duplicates)
            
            # 统计受影响的文件
            for duplicate in duplicates:
                if 'files' in duplicate:
                    files = duplicate['files']
                    if isinstance(files, list):
                        for f in files:
                            if isinstance(f, str):
                                summary['files_affected'].add(f)
                            elif isinstance(f, dict) and 'file' in f:
                                summary['files_affected'].add(f['file'])
            
            # 评估严重程度
            if duplicate_type in ['duplicate_classes', 'duplicate_methods']:
                summary['severity_levels']['high'] += len(duplicates)
            elif duplicate_type in ['duplicate_imports', 'duplicate_patterns']:
                summary['severity_levels']['medium'] += len(duplicates)
            else:
                summary['severity_levels']['low'] += len(duplicates)
        
        summary['files_affected'] = len(summary['files_affected'])
        
        return summary
    
    def generate_report(self, results: Dict) -> str:
        """生成分析报告"""
        report = []
        report.append("# 代码冗余分析报告")
        report.append("")
        
        # 汇总统计
        summary = results['summary']
        report.append("## 汇总统计")
        report.append(f"- 总重复问题数: {summary['total_duplicates']}")
        report.append(f"- 受影响文件数: {summary['files_affected']}")
        report.append(f"- 高严重程度: {summary['severity_levels']['high']}")
        report.append(f"- 中严重程度: {summary['severity_levels']['medium']}")
        report.append(f"- 低严重程度: {summary['severity_levels']['low']}")
        report.append("")
        
        # 详细分析
        for duplicate_type, duplicates in results.items():
            if duplicate_type == 'summary':
                continue
                
            if duplicates:
                report.append(f"## {duplicate_type.replace('_', ' ').title()}")
                report.append(f"发现 {len(duplicates)} 个重复问题")
                report.append("")
                
                for i, duplicate in enumerate(duplicates[:5], 1):  # 只显示前5个
                    report.append(f"### {i}. {duplicate.get('description', duplicate.get('type', 'Unknown'))}")
                    
                    if 'files' in duplicate:
                        report.append("**涉及文件:**")
                        for file in duplicate['files']:
                            report.append(f"- {file}")
                    
                    if 'similarity' in duplicate:
                        report.append(f"**相似度:** {duplicate['similarity']:.2f}")
                    
                    if 'count' in duplicate:
                        report.append(f"**重复次数:** {duplicate['count']}")
                    
                    report.append("")
                
                if len(duplicates) > 5:
                    report.append(f"... 还有 {len(duplicates) - 5} 个重复问题")
                report.append("")
        
        return "\n".join(report)

def main():
    """主函数"""
    analyzer = RedundancyAnalyzer()
    results = analyzer.analyze_project()
    
    # 生成报告
    report = analyzer.generate_report(results)
    
    # 保存报告
    with open('docs/redundancy_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("代码冗余分析完成！")
    print(f"报告已保存到: docs/redundancy_analysis_report.md")
    
    # 打印汇总
    summary = results['summary']
    print(f"\n汇总统计:")
    print(f"- 总重复问题数: {summary['total_duplicates']}")
    print(f"- 受影响文件数: {summary['files_affected']}")
    print(f"- 高严重程度: {summary['severity_levels']['high']}")
    print(f"- 中严重程度: {summary['severity_levels']['medium']}")
    print(f"- 低严重程度: {summary['severity_levels']['low']}")

if __name__ == '__main__':
    main() 