#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python delta_auc_from_csv.py --root ./save --out-csv delta_auc.csv --heatmap delta_auc_heatmap.png --strict-100
python delta_auc_from_csv.py --root ./save --out-csv delta_auc_first40.csv --heatmap delta_auc_first40.png --start 0 --end 39 --strict-100 --fill-na
python delta_auc_from_csv.py --root ./save --out-csv delta_auc_last40.csv  --heatmap delta_auc_last40.png  --start 60 --end 99 --strict-100 --fill-na
"""
"""
Compute ΔAUC@100 for epochs 0..99 and (optionally) draw a heatmap.

File naming under --root (default ./save):
  <Defense>_<Attack>_<Dataset>_accuracy.csv
e.g., Bulyan_Fang_cifar_accuracy.csv, Bulyan_NoAttack_cifar_accuracy.csv

Each CSV should contain two columns: Epoch & Accuracy（大小写/空格不敏感）。
Accuracy in [0,1] will be scaled to percentage points (pp) by *100; [0,100] kept as-is.

Strict definition:
  ΔAUC@100 = (1/100) * Σ_{t=0..99} [ Acc_no(t) − Acc_atk(t) ]
Use --strict-100 to enforce exactly that; if any epoch missing, either error or
use --fill-na to forward/back-fill to complete 0..99.

Outputs:
  1) CSV: dataset, defense, attack, epochs_used, auc_noattack, auc_attack, delta_auc_pp
  2) Optional PNG heatmap (rows = <dataset — defense>, cols = attacks)

Usage:
  python delta_auc_from_csv.py --root ./save --out-csv delta_auc.csv \
         --heatmap delta_auc_heatmap.png --strict-100
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ Config ------------
# 固定列顺序：FedGhost 倒数第二，FedSDF 最后一列
ATTACK_ORDER = ["LIE","MinMAX","MinSum","Fang","I-FMPA","PoisonedFL","FedGhost","FedSDF"]

# 识别“无攻击”基线的别名（文件名中的 Attack 段）
BASELINE_ALIASES = {
    "noattack","no_attack","no-attack","clean","baseline","none","normal",
    "acc","accuracy","noatk"
}

# 识别列名（不区分大小写）
EPOCH_ALIASES = {"epoch","round","iter","iteration","step"}
ACC_ALIASES   = {"accuracy","acc","test_acc","val_acc","val-acc","test-acc"}

# ------------ Helpers ------------
def parse_filename(fname: str):
    """Parse '<Defense>_<Attack>_<Dataset>_accuracy.csv' -> (defense, attack, dataset)."""
    m = re.match(r"^(.+?)_(.+?)_(.+?)_accuracy\.csv$", fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)

def load_curve(filepath: Path) -> pd.DataFrame:
    """Return DataFrame with columns: epoch(int), acc(float in pp)."""
    df = pd.read_csv(filepath)
    # 清洗列名
    clean = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=clean)
    lower = {c.lower(): c for c in df.columns}

    # 选 epoch 列
    epoch_col = None
    for k in EPOCH_ALIASES:
        if k in lower:
            epoch_col = lower[k]; break
    if epoch_col is None:
        # 兜底：找最像单调计数的数值列
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(5, int(0.5*len(s))) and np.all(np.diff(s.dropna()) >= 0):
                epoch_col = c; break
    if epoch_col is None:
        raise ValueError(f"No epoch-like column in {filepath.name}. Columns={list(df.columns)}")

    # 选 accuracy 列
    acc_col = None
    for k in ACC_ALIASES:
        if k in lower:
            acc_col = lower[k]; break
    if acc_col is None:
        # 兜底：除 epoch 外的第一数值列
        for c in df.columns:
            if c == epoch_col: continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(5, int(0.5*len(s))):
                acc_col = c; break
    if acc_col is None:
        raise ValueError(f"No accuracy-like column in {filepath.name}. Columns={list(df.columns)}")

    ep = pd.to_numeric(df[epoch_col], errors="coerce").astype(float).round().astype(int)
    acc = pd.to_numeric(df[acc_col], errors="coerce").astype(float)

    # [0,1] -> *100 转 pp
    finite = acc[np.isfinite(acc)]
    if finite.size and 0.0 <= float(finite.min()) and float(finite.max()) <= 1.0:
        acc = acc * 100.0

    out = pd.DataFrame({"epoch": ep, "acc": acc})
    out = out.dropna().drop_duplicates(subset=["epoch"]).sort_values("epoch")
    return out

def is_baseline_name(name: str) -> bool:
    return name.lower() in BASELINE_ALIASES

def compute_delta_auc_by_epoch(no_df: pd.DataFrame,
                               atk_df: pd.DataFrame,
                               start: int = 0,
                               end:   int = 99,
                               strict_100: bool = False,
                               fill_na: bool = False):
    """
    Align by epoch then compute:
      ΔAUC = (1/N) * Σ_{t=start..end} [Acc_no(t) − Acc_atk(t)]
    If strict_100=True: require full coverage [start..end] (N=end-start+1).
      - If fill_na=True: ffill/bfill to complete the range.
      - Else: raise error if missing epochs.
    Otherwise N = size of inner-joined epochs.
    """
    # 限定范围
    no  = no_df[(no_df.epoch >= start) & (no_df.epoch <= end)].copy()
    atk = atk_df[(atk_df.epoch >= start) & (atk_df.epoch <= end)].copy()
    total = end - start + 1

    # 补齐
    if strict_100 and fill_na:
        full = pd.DataFrame({"epoch": np.arange(start, end+1)})
        no  = full.merge(no,  on="epoch", how="left").sort_values("epoch")
        atk = full.merge(atk, on="epoch", how="left").sort_values("epoch")
        no["acc"]  = no["acc"].ffill().bfill()
        atk["acc"] = atk["acc"].ffill().bfill()

    # 对齐
    joined = pd.merge(no, atk, on="epoch", how="inner", suffixes=("_no","_atk")).sort_values("epoch")

    if strict_100:
        if len(joined) != total:
            raise ValueError(f"strict-100: epochs missing ({len(joined)}/{total}). Use --fill-na if acceptable.")
        N = total
    else:
        N = len(joined)
        if N == 0:
            return np.nan, np.nan, np.nan, 0

    acc_no = joined["acc_no"].to_numpy(float)
    acc_at = joined["acc_atk"].to_numpy(float)
    diff   = acc_no - acc_at

    delta  = float(np.nansum(diff)   / N)
    auc_no = float(np.nansum(acc_no) / N)
    auc_at = float(np.nansum(acc_at) / N)
    return auc_no, auc_at, delta, N

# ------------ Main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./save", help="Folder with *_accuracy.csv files")
    ap.add_argument("--out-csv", type=str, default="delta_auc.csv", help="Output CSV path")
    ap.add_argument("--heatmap", type=str, default="", help="Optional PNG path for ΔAUC heatmap")
    ap.add_argument("--start", type=int, default=0, help="Epoch start (inclusive)")
    ap.add_argument("--end",   type=int, default=99, help="Epoch end (inclusive)")
    ap.add_argument("--strict-100", action="store_true",
                    help="Use exact 1/(end-start+1) normalization and require full coverage")
    ap.add_argument("--fill-na", action="store_true",
                    help="When --strict-100, fill missing epochs by ffill/bfill to complete the range")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted([p for p in root.glob("*_accuracy.csv") if p.is_file()])
    if not files:
        print(f"[ERR] No CSVs found in: {root}", file=sys.stderr); sys.exit(1)

    # 读入所有曲线
    curves = {}
    for p in files:
        key = parse_filename(p.name)
        if key is None:
            warnings.warn(f"Skip unrecognized file name: {p.name}"); continue
        try:
            curves[key] = load_curve(p)
        except Exception as e:
            warnings.warn(f"Skip {p.name}: {e}"); continue

    # 基线（defense, dataset）-> df
    baselines = {(d, ds): s for (d, a, ds), s in curves.items() if is_baseline_name(a)}
    if not baselines:
        print("[ERR] No baseline '*_NoAttack_*_accuracy.csv' (or acc/accuracy/clean/etc.).", file=sys.stderr)
        sys.exit(1)

    # 计算 ΔAUC
    rows = []
    for (defense, attack, dataset), atk_df in curves.items():
        if is_baseline_name(attack):
            continue
        key = (defense, dataset)
        if key not in baselines:
            warnings.warn(f"Missing baseline for (defense={defense}, dataset={dataset}); skip.")
            continue
        try:
            auc_no, auc_atk, delta, N = compute_delta_auc_by_epoch(
                baselines[key], atk_df,
                start=args.start, end=args.end,
                strict_100=args.strict_100, fill_na=args.fill_na
            )
        except Exception as e:
            warnings.warn(f"Skip {defense}_{attack}_{dataset}: {e}")
            continue

        rows.append({
            "dataset": dataset,
            "defense": defense,
            "attack": attack,
            "epochs_used": N,
            "auc_noattack": round(auc_no, 4) if np.isfinite(auc_no) else np.nan,
            "auc_attack":  round(auc_atk, 4) if np.isfinite(auc_atk) else np.nan,
            "delta_auc_pp": round(delta, 4) if np.isfinite(delta) else np.nan,
        })

    if not rows:
        print("[ERR] No ΔAUC rows computed.", file=sys.stderr); sys.exit(1)

    out_csv = Path(args.out_csv)
    pd.DataFrame(rows).sort_values(["dataset","defense","attack"]).to_csv(out_csv, index=False)
    print(f"[OK] Saved ΔAUC CSV -> {out_csv.resolve()}")

    # ===== Heatmap =====
    if args.heatmap:
        df = pd.DataFrame(rows)

        # ---- 规范显示名称并设置排序 ----
        def norm_dataset(x: str) -> str:
            xl = str(x).lower().replace("_","-")
            if "cifar" in xl: return "cifar"
            if "f-mnist" in xl or "fashion" in xl: return "f-mnist"
            if "mnist" in xl: return "mnist"
            return str(x)

        def norm_defense(x: str) -> str:
            xl = str(x).lower().replace("_","-")
            mapping = {
                "fedavg": "FedAvg",
                "multi-krum": "Multi-Krum", "multikrum": "Multi-Krum", "multi krum": "Multi-Krum",
                "trimmed-mean": "Trimmed-mean", "trimmedmean": "Trimmed-mean", "trimmed_mean": "Trimmed-mean",
                "bulyan": "Bulyan",
                "cc": "CC",
                "rfa": "RFA",
                "rofl": "RoFL",
            }
            return mapping.get(xl, x if isinstance(x, str) else str(x))

        def norm_attack(x: str) -> str:
            xl = str(x).strip()
            low = xl.lower().replace("_","-")
            mapping = {
                "lie": "LIE",
                "minmax": "MinMAX", "min-max": "MinMAX", "min_max": "MinMAX",
                "minsum": "MinSum",  "min-sum": "MinSum",  "min_sum": "MinSum",
                "fang": "Fang",
                "i-fmpa": "I-FMPA", "ifmpa": "I-FMPA", "i_fmpa": "I-FMPA",
                "fedghost": "FedGhost", "fed-ghost": "FedGhost",
                "poisonedfl": "PoisonedFL", "poisoned-fl": "PoisonedFL", "pfl": "PoisonedFL",
                "fedsdf": "FedSDF", "fed-sdf": "FedSDF",
            }
            base = {"noattack","no-attack","acc","accuracy","clean","baseline","none","noatk"}
            if low in base: return "NoAttack"
            return mapping.get(low, xl)

        df["dataset_norm"] = df["dataset"].apply(norm_dataset)
        df["defense_norm"] = df["defense"].apply(norm_defense)
        df["attack_norm"]  = df["attack"].apply(norm_attack)

        # 行顺序：cifar→mnist→f-mnist；各自内部固定防御顺序
        dataset_order = ["cifar","mnist","f-mnist"]
        defense_order = ["FedAvg","Multi-Krum","Trimmed-mean","Bulyan","CC","RFA","RoFL"]
        df["dataset_norm"] = pd.Categorical(df["dataset_norm"], categories=dataset_order, ordered=True)
        df["defense_norm"] = pd.Categorical(df["defense_norm"], categories=defense_order, ordered=True)
        df = df.sort_values(["dataset_norm","defense_norm"])

        # 列顺序：按 ATTACK_ORDER；确保 FedGhost 倒数第二、FedSDF 最后
        attacks_present = list(dict.fromkeys(df["attack_norm"]))
        col_order = [a for a in ATTACK_ORDER if a in attacks_present]

        # 行标签：<dataset — defense>
        df["row_label"] = df["dataset_norm"].astype(str) + " — " + df["defense_norm"].astype(str)

        pivot = df.pivot_table(index="row_label", columns="attack_norm",
                               values="delta_auc_pp", aggfunc="mean") \
                 .reindex(index=df["row_label"].unique(), columns=col_order)

        if pivot.empty:
            print("[WARN] Heatmap pivot empty; skip plotting.")
            return

        # ---- 绘图 ----
        fig, ax = plt.subplots(figsize=(12, 8), dpi=220)
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))

        mat = pivot.values
        vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1.0
        for i in range(mat.shape[0]):
            row = mat[i, :]
            if np.all(np.isnan(row)):
                continue
            maxv = np.nanmax(row)
            winners = np.where(np.isclose(row, maxv, equal_nan=False))[0]
            for j, v in enumerate(row):
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}"
                bold = (j in winners)
                if bold: txt = "★ " + txt
                color = "white" if (vmax and v > 0.6 * vmax) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, fontweight=("bold" if bold else "normal"),
                        color=color)

        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        total = args.end - args.start + 1
        cbar.set_label(f"ΔAUC@{total}(pp)")

        plt.title(f"ΔAUC@{total} Heatmap")
        plt.tight_layout()
        out_png = Path(args.heatmap)
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved heatmap -> {out_png.resolve()}")

if __name__ == "__main__":
    main()