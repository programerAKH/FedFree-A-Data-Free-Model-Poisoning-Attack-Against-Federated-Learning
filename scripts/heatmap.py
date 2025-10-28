#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成9张热力图：3个时间段（100轮、前40轮、后40轮） × 3个数据集（cifar、f-mnist、mnist）
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
ATTACK_ORDER = ["LIE", "MinMAX", "MinSum", "Fang", "I-FMPA", "PoisonedFL", "FedGhost", "FedFree"]

# 识别"无攻击"基线的别名（文件名中的 Attack 段）
BASELINE_ALIASES = {
    "noattack", "no_attack", "no-attack", "clean", "baseline", "none", "normal",
    "acc", "accuracy", "noatk"
}

# 识别列名（不区分大小写）
EPOCH_ALIASES = {"epoch", "round", "iter", "iteration", "step"}
ACC_ALIASES = {"accuracy", "acc", "test_acc", "val_acc", "val-acc", "test-acc"}


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
            epoch_col = lower[k];
            break
    if epoch_col is None:
        # 兜底：找最像单调计数的数值列
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(5, int(0.5 * len(s))) and np.all(np.diff(s.dropna()) >= 0):
                epoch_col = c;
                break
    if epoch_col is None:
        raise ValueError(f"No epoch-like column in {filepath.name}. Columns={list(df.columns)}")

    # 选 accuracy 列
    acc_col = None
    for k in ACC_ALIASES:
        if k in lower:
            acc_col = lower[k];
            break
    if acc_col is None:
        # 兜底：除 epoch 外的第一数值列
        for c in df.columns:
            if c == epoch_col: continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(5, int(0.5 * len(s))):
                acc_col = c;
                break
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
                               end: int = 99,
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
    no = no_df[(no_df.epoch >= start) & (no_df.epoch <= end)].copy()
    atk = atk_df[(atk_df.epoch >= start) & (atk_df.epoch <= end)].copy()
    total = end - start + 1

    # 补齐
    if strict_100 and fill_na:
        full = pd.DataFrame({"epoch": np.arange(start, end + 1)})
        no = full.merge(no, on="epoch", how="left").sort_values("epoch")
        atk = full.merge(atk, on="epoch", how="left").sort_values("epoch")
        no["acc"] = no["acc"].ffill().bfill()
        atk["acc"] = atk["acc"].ffill().bfill()

    # 对齐
    joined = pd.merge(no, atk, on="epoch", how="inner", suffixes=("_no", "_atk")).sort_values("epoch")

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
    diff = acc_no - acc_at

    delta = float(np.nansum(diff) / N)
    auc_no = float(np.nansum(acc_no) / N)
    auc_at = float(np.nansum(acc_at) / N)
    return auc_no, auc_at, delta, N


def generate_individual_heatmaps(rows, start, end, output_dir):
    """为每个数据集单独生成热力图"""
    df = pd.DataFrame(rows)

    # ---- 规范显示名称并设置排序 ----
    def norm_dataset(x: str) -> str:
        xl = str(x).lower().replace("_", "-")
        if "cifar" in xl: return "cifar-10"
        if "f-mnist" in xl or "fashion" in xl: return "fashion-mnist"
        if "mnist" in xl: return "mnist"
        return str(x)

    def norm_defense(x: str) -> str:
        xl = str(x).lower().replace("_", "-")
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
        low = xl.lower().replace("_", "-")
        mapping = {
            "lie": "LIE",
            "minmax": "MinMAX", "min-max": "MinMAX", "min_max": "MinMAX",
            "minsum": "MinSum", "min-sum": "MinSum", "min_sum": "MinSum",
            "fang": "Fang",
            "i-fmpa": "I-FMPA", "ifmpa": "I-FMPA", "i_fmpa": "I-FMPA",
            "fedghost": "FedGhost", "fed-ghost": "FedGhost",
            "poisonedfl": "PoisonedFL", "poisoned-fl": "PoisonedFL", "pfl": "PoisonedFL",
            "fedsdf": "FedFree", "fed-sdf": "FedFree",
        }
        base = {"noattack", "no-attack", "acc", "accuracy", "clean", "baseline", "none", "noatk"}
        if low in base: return "NoAttack"
        return mapping.get(low, xl)

    df["dataset_norm"] = df["dataset"].apply(norm_dataset)
    df["defense_norm"] = df["defense"].apply(norm_defense)
    df["attack_norm"] = df["attack"].apply(norm_attack)

    # 数据集顺序：cifar→f-mnist→mnist
    dataset_order = ["cifar-10", "fashion-mnist", "mnist"]
    defense_order = ["FedAvg", "Multi-Krum", "Trimmed-mean", "Bulyan", "CC", "RFA", "RoFL"]
    df["dataset_norm"] = pd.Categorical(df["dataset_norm"], categories=dataset_order, ordered=True)
    df["defense_norm"] = pd.Categorical(df["defense_norm"], categories=defense_order, ordered=True)
    df = df.sort_values(["dataset_norm", "defense_norm"])

    # 列顺序：按 ATTACK_ORDER
    attacks_present = list(dict.fromkeys(df["attack_norm"]))
    col_order = [a for a in ATTACK_ORDER if a in attacks_present]

    # 为每个数据集生成单独的热力图
    for dataset in dataset_order:
        dataset_df = df[df["dataset_norm"] == dataset]
        if dataset_df.empty:
            print(f"[WARN] No data for dataset {dataset}; skip.")
            continue

        # 创建数据透视表
        pivot = dataset_df.pivot_table(index="defense_norm", columns="attack_norm",
                                       values="delta_auc_pp", aggfunc="mean") \
            .reindex(index=defense_order, columns=col_order)

        if pivot.empty:
            print(f"[WARN] Heatmap pivot empty for {dataset}; skip plotting.")
            continue

        # ---- 绘图 ----
        n_rows = len(pivot.index)
        n_cols = len(pivot.columns)

        # 根据矩阵大小动态调整图形尺寸
        fig_width = max(10, n_cols * 1.2)
        fig_height = max(6, n_rows * 0.7)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

        # 使用 Reds 颜色映射，确保最小值从0开始
        vmin = 0
        vmax = max(1.0, np.nanmax(pivot.values) if np.isfinite(pivot.values).any() else 1.0)
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)

        # 设置坐标轴标签
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(list(pivot.index), fontsize=11)

        # 添加网格线
        ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.3, alpha=0.5)
        ax.tick_params(which="minor", size=0)

        # 在单元格中显示数值
        mat = pivot.values
        for i in range(n_rows):
            for j in range(n_cols):
                v = mat[i, j]
                if np.isnan(v):
                    continue

                # 保留2位小数
                txt = f"{v:.2f}"

                # 标记每行的最大值
                row = mat[i, :]
                row_max = np.nanmax(row) if np.isfinite(row).any() else 0
                is_max = not np.isnan(v) and abs(v - row_max) < 1e-6

                if is_max:
                    txt = "★" + txt

                # 根据背景色调整文字颜色
                color = "white" if v > 0.6 * vmax else "black"

                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight=("bold" if is_max else "normal"),
                        color=color)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        total_epochs = end - start + 1
        cbar.set_label(f"ΔAUC@{total_epochs}(pp)", fontsize=11)

        # 设置标题
        epoch_range = f"Epochs {start}-{end}" if total_epochs < 100 else "All 100 Epochs"
        plt.title(f"{dataset.upper()} - ΔAUC@{total_epochs} Heatmap ({epoch_range})",
                  fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # 保存图片
        output_path = output_dir / f"delta_auc_{dataset}_{start}_{end}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor='white')
        plt.close()

        print(f"[OK] Saved {dataset} heatmap -> {output_path}")


def generate_combined_heatmap(rows, start, end, output_path):
    """生成组合热力图（保留此函数以备需要）"""
    # 此函数保留，但在此版本中我们主要使用单独的热力图
    pass


# ------------ Main ------------
def main():
    # 自动生成9张热力图：3个时间段 × 3个数据集
    configs = [
        {"start": 0, "end": 99, "fill_na": False, "suffix": "100"},
        {"start": 0, "end": 39, "fill_na": True, "suffix": "first40"},
        {"start": 60, "end": 99, "fill_na": True, "suffix": "last40"}
    ]

    # 创建输出目录
    output_dir = Path("./heatmaps")
    output_dir.mkdir(exist_ok=True)

    root = Path(r"E:\PyCharm Community Edition 2024.2.3\py_projects\save")
    files = sorted([p for p in root.glob("*_accuracy.csv") if p.is_file()])
    if not files:
        print(f"[ERR] No CSVs found in: {root}", file=sys.stderr);
        sys.exit(1)

    # 读入所有曲线
    curves = {}
    for p in files:
        key = parse_filename(p.name)
        if key is None:
            warnings.warn(f"Skip unrecognized file name: {p.name}");
            continue
        try:
            curves[key] = load_curve(p)
        except Exception as e:
            warnings.warn(f"Skip {p.name}: {e}");
            continue

    # 基线（defense, dataset）-> df
    baselines = {(d, ds): s for (d, a, ds), s in curves.items() if is_baseline_name(a)}
    if not baselines:
        print("[ERR] No baseline '*_NoAttack_*_accuracy.csv' (or acc/accuracy/clean/etc.).", file=sys.stderr)
        sys.exit(1)

    # 为每个配置生成热力图
    for config in configs:
        start = config["start"]
        end = config["end"]
        fill_na = config["fill_na"]
        suffix = config["suffix"]

        print(f"\n=== Generating heatmaps for epochs {start}-{end} ===")

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
                    start=start, end=end,
                    strict_100=True, fill_na=fill_na
                )
            except Exception as e:
                warnings.warn(f"Skip {defense}_{attack}_{dataset}: {e}")
                continue

            # 保留2位小数
            rows.append({
                "dataset": dataset,
                "defense": defense,
                "attack": attack,
                "epochs_used": N,
                "auc_noattack": round(auc_no, 2) if np.isfinite(auc_no) else np.nan,
                "auc_attack": round(auc_atk, 2) if np.isfinite(auc_atk) else np.nan,
                "delta_auc_pp": round(delta, 2) if np.isfinite(delta) else np.nan,
            })

        if not rows:
            print(f"[ERR] No ΔAUC rows computed for {suffix}.", file=sys.stderr)
            continue

        # 保存CSV
        out_csv = output_dir / f"delta_auc_{suffix}.csv"
        pd.DataFrame(rows).sort_values(["dataset", "defense", "attack"]).to_csv(out_csv, index=False)
        print(f"[OK] Saved ΔAUC CSV -> {out_csv.resolve()}")

        # 为每个数据集生成单独的热力图
        generate_individual_heatmaps(rows, start, end, output_dir)

    print(f"\n=== Summary ===")
    print(f"Generated 9 heatmaps (3 time periods × 3 datasets) in: {output_dir.resolve()}")
    print("File naming: delta_auc_<dataset>_<start>_<end>.png")
    print("Datasets: cifar, f-mnist, mnist")
    print("Time periods: 0-99 (all), 0-39 (first 40), 60-99 (last 40)")


if __name__ == "__main__":
    main()
