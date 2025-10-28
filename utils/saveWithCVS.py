import csv
import os
from typing import List


def save_accuracy_to_csv(
        acc_tests: List[float],
        defense_method: str,
        attack_method: str,
        dataset: str,
        save_dir: str = "./save/alphaScan"
) -> None:
    """
    保存全局模型测试准确率到 CSV 文件

    参数:
    - acc_tests: 测试准确率列表（例如 [0.12, 0.35, ...]）
    - defense_method: 防御方法名称（如 "krum" 或 "bulyan"）
    - attack_method: 攻击方法名称（如 "data_poisoning" 或 "model_poisoning"）
    - save_dir: 保存目录（默认为 "./save/"）
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名（防御方法_攻击方法_accuracy.csv）
    filename = f"{defense_method}_{attack_method}_{dataset}_accuracy.csv"
    csv_path = os.path.join(save_dir, filename)

    # 写入CSV文件
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Accuracy'])  # 表头

        # 写入每一轮准确率（保留4位小数）
        for epoch, acc in enumerate(acc_tests):
            writer.writerow([epoch, f"{acc:.4f}"])

    print(f"[Success] Accuracy saved to: {csv_path}")
