import os
import numpy as np
import pandas as pd
from pathlib import Path

# 路径请改成你自己的
ROOT = Path(r"E:\PyCharm Community Edition 2024.2.3\py_projects\save\pScan")
OUT_CSV = "pScan_results.csv"

def load_curve(path: Path):
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    epoch_col = next((lower[k] for k in lower if k in {"epoch","round","iter","step"}), df.columns[0])
    acc_col   = next((lower[k] for k in lower if k in {"accuracy","acc","test_acc","val_acc"}), df.columns[-1])
    ep = pd.to_numeric(df[epoch_col], errors="coerce").astype(int)
    acc = pd.to_numeric(df[acc_col], errors="coerce").astype(float)
    if acc.min() >= 0 and acc.max() <= 1.0:
        acc = acc * 100
    return pd.DataFrame({"epoch": ep, "acc": acc}).dropna().drop_duplicates("epoch").set_index("epoch").sort_index()

def compute_delta_auc(no_df, atk_df, start=0, end=99):
    idx = np.arange(start, end+1)
    no  = no_df.reindex(idx).interpolate().ffill().bfill()["acc"].to_numpy()
    atk = atk_df.reindex(idx).interpolate().ffill().bfill()["acc"].to_numpy()
    return no.mean(), atk.mean(), (no - atk).mean()

def main():
    rows = []
    # 加载 baseline (NoAttack)
    baselines = {}
    for f in ROOT.glob("*NoAttack*csv"):
        defense, attack, dataset, _ = f.name.split("_")
        baselines[(defense, dataset)] = load_curve(f)

    # 遍历子目录 p=3,5,7,10
    for sub in ["3", "5", "7", "10"]:
        dpath = ROOT / sub
        for f in dpath.glob("*FedSDF*csv"):
            defense, attack, dataset, _ = f.name.split("_")
            key = (defense, dataset)
            if key not in baselines:
                print(f"[WARN] Baseline missing for {defense}-{dataset}")
                continue
            auc_no, auc_atk, delta = compute_delta_auc(baselines[key], load_curve(f))
            rows.append({
                "p": sub,
                "dataset": dataset,
                "defense": defense,
                "auc_noattack": round(auc_no, 4),
                "auc_attack": round(auc_atk, 4),
                "delta_auc_pp": round(delta, 4),
            })

    df = pd.DataFrame(rows).sort_values(["p","dataset","defense"])
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved -> {OUT_CSV}")

if __name__ == "__main__":
    main()
