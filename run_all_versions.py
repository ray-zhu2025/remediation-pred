import subprocess
import json
import os
from datetime import datetime

def run_version(version):
    print(f"Running version {version}...")
    subprocess.run(["./run.sh", version], check=True)
    
def collect_metrics():
    metrics = {}
    metrics_dir = "output/metrics"
    for fname in os.listdir(metrics_dir):
        if fname.startswith("model_metrics_v") and fname.endswith(".json"):
            # 提取版本号
            parts = fname.split("_")
            version = parts[2] if len(parts) > 2 else "unknown"
            with open(os.path.join(metrics_dir, fname), "r") as f:
                data = json.load(f)
            metrics[version] = data
    return metrics

def main():
    versions = [
        "1.0.0", "1.1.0", "1.1.1", "1.1.2", 
        "1.1.3", "1.1.4", "1.1.5", "1.1.6"
    ]
    
    for version in versions:
        run_version(version)
    
    metrics = collect_metrics()
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"output/metrics_summary_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics Summary:")
    for version, data in metrics.items():
        print(f"\nVersion {version}:")
        print(f"  Soil Model Accuracy: {data['soil']['accuracy']:.4f}")
        print(f"  Groundwater Model Accuracy: {data['groundwater']['accuracy']:.4f}")

if __name__ == "__main__":
    main() 