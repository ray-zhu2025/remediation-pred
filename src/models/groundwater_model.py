"""
地下水修复技术决策模型
使用多种机器学习模型预测地下水修复技术
"""

# 本地应用导入
from src.models.base_model import BaseModel

class GroundwaterModel(BaseModel):
    """地下水污染修复决策模型"""
    
    def __init__(self, config_path: str = "src/config/groundwater/parameters.json"):
        """
        初始化地下水模型
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)

def main():
    """主函数"""
    # 创建模型实例
    model = GroundwaterModel()
    
    # 训练模型
    model.train("data/training/groundwater_training.csv")
    
    # 进行预测
    model.predict(
        "data/prediction/groundwater_prediction.csv",
        "output/groundwater"
    )
    
    # 评估模型
    model.evaluate("output/groundwater")

if __name__ == "__main__":
    main() 