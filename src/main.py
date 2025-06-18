"""
主程序入口
"""
import os
import json
import argparse
from datetime import datetime
from src.config.version_config import VersionConfig
from src.process.data_processor import DataProcessor
from src.models import DomainModel, DomainTabPFNModel
from src.utils.logging_utils import setup_logging
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='运行模型')
    parser.add_argument('--version', type=str, required=True, help='模型版本')
    parser.add_argument('--model_type', type=str, default='autogluon', 
                       choices=list(VersionConfig.MODEL_TYPES.keys()), help='模型类型')
    args = parser.parse_args()

    VersionConfig.set_version(args.version)
    logger = setup_logging('main')
    logger.info(f"开始运行版本 {args.version}, 模型类型: {args.model_type}")

    try:
        config = VersionConfig.get_version_config()
        model_config = VersionConfig.MODEL_TYPES[args.model_type]

        processor = DataProcessor()
        soil_data = processor.load_soil_data()
        groundwater_data = processor.load_groundwater_data()

        # 训练土壤模型
        if args.model_type == 'autogluon':
            soil_model = DomainModel(domain_type='soil', version=args.version)
            groundwater_model = DomainModel(domain_type='groundwater', version=args.version)
        else:
            soil_model = DomainTabPFNModel(domain_type='soil', version=args.version)
            groundwater_model = DomainTabPFNModel(domain_type='groundwater', version=args.version)

        soil_model.train(
            soil_data['X_train'], 
            soil_data['y_train'],
            soil_data['X_test'],
            soil_data['y_test']
        )
        soil_metrics = soil_model.evaluate(soil_data['X_test'], soil_data['y_test'])

        groundwater_model.train(
            groundwater_data['X_train'],
            groundwater_data['y_train'],
            groundwater_data['X_test'],
            groundwater_data['y_test']
        )
        groundwater_metrics = groundwater_model.evaluate(groundwater_data['X_test'], groundwater_data['y_test'])

        logger.info("\n模型评估结果:")
        logger.info(f"土壤模型准确率: {soil_metrics['accuracy']:.4f}")
        logger.info(f"地下水模型准确率: {groundwater_metrics['accuracy']:.4f}")

        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        soil_save_path = os.path.join('models', 'soil', f'v{args.version}', timestamp)
        groundwater_save_path = os.path.join('models', 'groundwater', f'v{args.version}', timestamp)
        os.makedirs(os.path.dirname(soil_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(groundwater_save_path), exist_ok=True)

        if model_config['save_model']:
            soil_model.save(soil_save_path)
            groundwater_model.save(groundwater_save_path)
            logger.info("模型已保存到 models 目录")
        else:
            logger.info("当前模型类型无需保存模型文件")

        metrics_file = os.path.join('output', 'metrics', f'metrics_v{args.version}_{args.model_type}.json')
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

        if isinstance(soil_metrics.get('confusion_matrix'), np.ndarray):
            soil_metrics['confusion_matrix'] = soil_metrics['confusion_matrix'].tolist()
        if isinstance(groundwater_metrics.get('confusion_matrix'), np.ndarray):
            groundwater_metrics['confusion_matrix'] = groundwater_metrics['confusion_matrix'].tolist()

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'soil_metrics': soil_metrics,
                'groundwater_metrics': groundwater_metrics,
                'version': args.version,
                'model_type': args.model_type,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"评估指标已保存到: {metrics_file}")

    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 