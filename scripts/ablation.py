import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"E:\PyCharm Community Edition 2024.2.3\py_projects\save\ablation")
OUT_CSV = "ablation_results.csv"

def load_curve(path: Path):
    """读取CSV，返回 epoch->acc 的 DataFrame"""
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    epoch_col = next((lower[k] for k in lower if k in {"epoch","round","iter","step"}), df.columns[0])
    acc_col   = next((lower[k] for k in lower if k in {"accuracy","acc","test_acc","val_acc"}), df.columns[-1])
    ep  = pd.to_numeric(df[epoch_col], errors="coerce").astype(int)
    acc = pd.to_numeric(df[acc_col], errors="coerce").astype(float)
    if acc.min() >= 0 and acc.max() <= 1.0:  # 如果在[0,1]区间，转为百分比
        acc *= 100
    return pd.DataFrame({"epoch": ep, "acc": acc}).dropna().drop_duplicates("epoch").set_index("epoch").sort_index()

def compute_delta_auc(no_df, atk_df, start=0, end=99):
    """计算 ΔAUC"""
    idx = np.arange(start, end+1)
    no  = no_df.reindex(idx).interpolate().ffill().bfill()["acc"].to_numpy()
    atk = atk_df.reindex(idx).interpolate().ffill().bfill()["acc"].to_numpy()
    return no.mean(), atk.mean(), (no - atk).mean()

def main():
    rows = []

    # 找 baseline
    baselines = {}
    for f in ROOT.glob("*NoAttack*csv"):
        parts = f.name.split("_")
        defense, dataset = parts[0], parts[2]
        baselines[(defense, dataset)] = load_curve(f)

    # 遍历消融变体
    for f in ROOT.glob("*csv"):
        if "NoAttack" in f.name:
            continue
        parts = f.name.split("_")
        defense, dataset, variant = parts[0], parts[2], parts[3]
        key = (defense, dataset)
        if key not in baselines:
            print(f"[WARN] Missing baseline for {defense}-{dataset}")
            continue
        auc_no, auc_atk, delta = compute_delta_auc(baselines[key], load_curve(f))
        rows.append({
            "dataset": dataset,
            "defense": defense,
            "variant": variant,
            "auc_noattack": round(auc_no, 4),
            "auc_attack": round(auc_atk, 4),
            "delta_auc_pp": round(delta, 4),
        })

    df = pd.DataFrame(rows).sort_values(["dataset","defense","variant"])
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved -> {OUT_CSV}")

if __name__ == "__main__":
    main()
