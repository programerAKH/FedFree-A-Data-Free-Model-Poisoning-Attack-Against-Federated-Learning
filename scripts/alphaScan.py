import os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"E:\PyCharm Community Edition 2024.2.3\py_projects\save\alphaScan")
OUT_CSV = "alphaScan_results.csv"

def load_curve(path: Path):
    df = pd.read_csv(path)
    # 自动找 epoch 列和 acc 列
    lower = {c.lower(): c for c in df.columns}
    epoch_col = next((lower[k] for k in lower if k in {"epoch","round","iter","step"}), df.columns[0])
    acc_col   = next((lower[k] for k in lower if k in {"accuracy","acc","test_acc","val_acc"}), df.columns[-1])
    ep = pd.to_numeric(df[epoch_col], errors="coerce").astype(int)
    acc = pd.to_numeric(df[acc_col], errors="coerce").astype(float)
    # 如果在 [0,1] 区间，转成百分比
    if acc.min() >= 0 and acc.max() <= 1.0:
        acc = acc * 100
    out = pd.DataFrame({"epoch": ep, "acc": acc})
    return out.dropna().drop_duplicates("epoch").set_index("epoch").sort_index()

def compute_delta_auc(no_df, atk_df, start=0, end=99):
    # 对齐 0-99
    idx = np.arange(start, end+1)
    no  = no_df.reindex(idx).interpolate().ffill().bfill()["acc"].to_numpy()
    atk = atk_df.reindex(idx).interpolate().ffill().bfill()["acc"].to_numpy()
    auc_no = no.mean()
    auc_atk = atk.mean()
    delta = (no - atk).mean()
    return auc_no, auc_atk, delta

def main():
    rows = []
    for alpha_dir in ROOT.iterdir():
        if not alpha_dir.is_dir():
            continue
        alpha = alpha_dir.name  # 例如 "0.1", "0.3", "iid"
        for atk_file in alpha_dir.glob("*_FedSDF_*_accuracy.csv"):
            fname = atk_file.name
            parts = fname.split("_")  # e.g. FedAvg_FedSDF_cifar_accuracy.csv
            defense, attack, dataset, _ = parts
            no_file = alpha_dir / f"{defense}_NoAttack_{dataset}_accuracy.csv"
            if not no_file.exists():
                print(f"[WARN] Missing baseline for {fname}")
                continue
            # 读取曲线
            df_no = load_curve(no_file)
            df_atk = load_curve(atk_file)
            auc_no, auc_atk, delta = compute_delta_auc(df_no, df_atk)
            rows.append({
                "alpha": alpha,
                "dataset": dataset,
                "defense": defense,
                "auc_noattack": round(auc_no, 4),
                "auc_fedsdf": round(auc_atk, 4),
                "delta_auc": round(delta, 4)
            })
    # 保存
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved results -> {OUT_CSV}")

if __name__ == "__main__":
    main()
