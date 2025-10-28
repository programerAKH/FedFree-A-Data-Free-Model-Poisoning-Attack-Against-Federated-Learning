import os
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"E:\PyCharm Community Edition 2024.2.3\py_projects\save\shampoo")
OUT_CSV = "simonly_stats_combined.csv"

datasets = {
    "cifar": "CIFAR-10 (AlexNet)",
    "cifar100": "CIFAR-100 (ResNet-34)",
    "f-mnist": "F-MNIST (CNN)",
    "mnist": "MNIST (FCNet)"
}

alphas = {
    "0.1": ["alpha_0.1", "alpha0.1"],
    "0.5": ["alpha_0.5", "alpha0.5"],
    "1.0": ["alpha_1.0", "alpha1.0"],
    "iid": ["alpha_iid", "iid"]
}

rows = []

for ds_key, ds_name in datasets.items():
    for alpha, patterns in alphas.items():
        folder = ROOT / ds_key
        files = [f for f in folder.glob("*.csv") if any(p in f.name for p in patterns)]
        if not files:
            print(f"[WARN] Missing file for {ds_key}, alpha={alpha}")
            continue

        path = files[0]
        df = pd.read_csv(path)

        # ---- 跳过第 0 轮 ----
        df = df[df["round"] > 0]

        cos_vals = df["cosine"].dropna().to_numpy()
        norm_vals = df["norm_ratio"].dropna().to_numpy()

        cos_mean, cos_std = np.mean(cos_vals), np.std(cos_vals)
        norm_mean, norm_std = np.mean(norm_vals), np.std(norm_vals)

        # 余弦相似度保留 3 位小数，范数比保留 2 位小数
        rows.append({
            "dataset": ds_name,
            "alpha": alpha,
            "cosine": f"{cos_mean:.3f} ± {cos_std:.3f}",
            "norm": f"{norm_mean:.2f} ± {norm_std:.2f}"
        })

# 整理输出
df_out = pd.DataFrame(rows)
df_out = df_out.pivot(index="dataset", columns="alpha", values=["cosine","norm"])
df_out.to_csv(OUT_CSV)
print(f"[OK] Saved combined stats -> {OUT_CSV}")
