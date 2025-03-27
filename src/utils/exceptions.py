"""
自定义异常类
"""

class ModelError(Exception):
    """模型相关错误"""
    pass

class DataError(Exception):
    """数据处理相关错误"""
    pass

class ConfigError(Exception):
    """配置相关错误"""
    pass

class ValidationError(Exception):
    """数据验证相关错误"""
    pass

class TrainingError(Exception):
    """模型训练相关错误"""
    pass

class EvaluationError(Exception):
    """模型评估相关错误"""
    pass 