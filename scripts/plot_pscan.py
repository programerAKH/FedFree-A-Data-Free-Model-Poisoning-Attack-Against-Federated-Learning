import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv("pScan_results.csv")

# 防御方法顺序
defense_order = ["FedAvg", "Multi-Krum", "Trimmed-mean", "Bulyan", "CC", "RFA", "RoFL"]

# 要画的两个数据集
# 数据集的映射表与顺序
dataset_display = {
    "cifar": "CIFAR-10",
    "f-mnist": "Fashion-MNIST",
    "mnist": "MNIST"
}
datasets = list(dataset_display.keys())


# 创建包含三个子图的图形
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)  # 调整图形大小

# 存储所有线条的句柄和标签
all_handles = []
all_labels = []

for i, ds in enumerate(datasets):
    ax = axes[i]
    sub = df[df["dataset"] == ds]

    for defense in defense_order:
        dsub = sub[sub["defense"] == defense].sort_values("p")
        if dsub.empty:
            continue
        line, = ax.plot(
            dsub["p"],
            dsub["delta_auc_pp"],
            marker="o",
            label=defense
        )
        # 只在第一次遇到该防御方法时记录句柄和标签
        if defense not in all_labels:
            all_handles.append(line)
            all_labels.append(defense)

    # 添加子图标签 (a), (b), (c)
    ax.text(
        0.5, -0.15,
        f"({chr(97 + i)}) {dataset_display.get(ds, ds)} — ΔAUC vs Malicious Client Ratio",
        transform=ax.transAxes, ha='center', fontsize=12
    )

    ax.set_xlabel("Number of malicious clients (p)")
    ax.set_ylabel("ΔAUC@100 (pp)")
    ax.grid(True, linestyle="--", alpha=0.6)

# 添加共享图例 - 横着放在上方并居中
fig.legend(
    handles=all_handles,
    labels=all_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=7,
    frameon=False,
    fontsize=9  # 稍微减小字体大小以适应水平排列
)

plt.tight_layout()
plt.savefig("pscan_combined.png", bbox_inches="tight")
plt.close()

print("[OK] 已生成 pscan_combined.png")