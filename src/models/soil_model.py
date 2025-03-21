"""
土壤修复技术决策模型
使用多种机器学习模型预测土壤修复技术
"""

# 本地应用导入
from src.models.base_model import BaseModel

class SoilModel(BaseModel):
    """土壤污染修复决策模型"""
    
    def __init__(self, config_path: str = "src/config/soil/parameters.json"):
        """
        初始化土壤模型
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)

def main():
    """主函数"""
    # 创建模型实例
    model = SoilModel()
    
    # 训练模型
    model.train("data/training/soil_training.csv")
    
    # 进行预测
    model.predict(
        "data/prediction/soil_prediction.csv",
        "output/soil"
    )
    
    # 评估模型
    model.evaluate("output/soil")

if __name__ == "__main__":
    main() 