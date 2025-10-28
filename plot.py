import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 用法：
# 1) 把本脚本放到 CSV 所在目录运行，或把 base_dir 改成你的数据目录路径。
# 2) CSV 命名：<Defense>_<Variant>_<Dataset>_accuracy.csv
#    例如：
#      Bulyan_FedGhost_cifar_accuracy.csv
#      Bulyan_FedGhost_f-mnist_accuracy.csv
#      Bulyan_FedGhost_mnist_accuracy.csv
#      CC_NoAttack_cifar_accuracy.csv
#      CC_FedSDF_cifar_accuracy.csv
# 3) 运行后会在 base_dir/plots/ 下生成图。
# =========================

VARIANTS = ["PoisonedFL", "FedSDF", "NoAttack"]  # 画图对比的三条曲线（有则画）

def find_accuracy_series(csv_path):
    """
    读取 CSV，鲁棒地获取“accuracy”序列。
    - 优先查找列名包含 'acc' 的列（accuracy / acc / ACC...）
    - 否则选择首个数值列
    索引从 1 开始作为回合数。
    """
    df = pd.read_csv(csv_path)
    # 优先找包含 'acc' 的列
    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    if acc_cols:
        s = pd.to_numeric(df[acc_cols[0]], errors='coerce')
    else:
        # 找到第一个数值列
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            s = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
        else:
            # 兜底：尝试把第一列转成数值
            s = pd.to_numeric(df.iloc[:, 0], errors='coerce')

    s = s.dropna().reset_index(drop=True)
    s.index = s.index + 1  # 回合数从 1 开始
    return s

def make_plots(base_dir="."):
    """
    扫描 base_dir 下的 CSV，匹配 <Defense>_<Variant>_<Dataset>_accuracy.csv
    将同一 (Defense, Dataset) 的多条 Variant 画到同一张图。
    输出到 base_dir/plots/
    """
    pattern = re.compile(
        r'^(?P<defense>[^_]+)_(?P<variant>PoisonedFL|FedSDF|NoAttack)_(?P<dataset>[^_]+)_accuracy\.csv$',
        re.IGNORECASE
    )
    groups = {}  # key: (defense, dataset) -> dict{variant: path}

    # 收集文件
    for fname in os.listdir(base_dir):
        if not fname.lower().endswith(".csv"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        defense = m.group("defense")
        variant = m.group("variant")
        dataset = m.group("dataset")
        key = (defense, dataset)
        groups.setdefault(key, {})
        groups[key][variant] = os.path.join(base_dir, fname)

    # 输出目录
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    made, skipped = [], []

    for (defense, dataset), files in groups.items():
        # 收集本组已有的曲线
        series_map = {}
        for v in VARIANTS:
            if v in files:
                try:
                    series_map[v] = find_accuracy_series(files[v])
                except Exception as e:
                    print(f"读取失败: {files[v]} -> {e}")

        if len(series_map) == 0:
            skipped.append((defense, dataset, "缺少任何可用曲线"))
            continue
        if len(series_map) == 1:
            print(f"⚠️ 仅找到 1 条曲线（{defense} + {dataset}）：{list(series_map.keys())[0]}，仍将绘图。")
        if len(series_map) == 2:
            miss = [v for v in VARIANTS if v not in series_map]
            print(f"ℹ️ 仅有 2 条曲线（{defense} + {dataset}），缺失：{miss}")

        # 画图（单图仅一个坐标轴，不指定颜色，也不使用 seaborn）
        plt.figure()
        # 为了对齐横轴，取最长长度
        max_len = max(len(s) for s in series_map.values())

        # 逐条画
        for v in VARIANTS:
            if v in series_map:
                s = series_map[v]
                plt.plot(s.index, s.values, label=v)

        plt.title(f"{defense} on {dataset} — Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.xlim(1, max_len)
        plt.legend()

        out_path = os.path.join(out_dir, f"{defense}_{dataset}_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        made.append(out_path)

    # 汇总打印
    if made:
        print("✅ 生成的图：")
        for p in made:
            print(" -", p)
    if skipped:
        print("\n⚠️ 以下组合被跳过：")
        for d, ds, reason in skipped:
            print(f" - {d} + {ds}: {reason}")

def plot_trimmed_mean_3datasets(base_dir="save",
                                datasets=("cifar", "f-mnist", "mnist"),
                                attacks=None,
                                out_dir_name="plots/trimmed-mean"):
    """
    只绘制 Trimmed-mean 防御在三个数据集上的对比图（每张图最多 9 条攻击曲线）。

    参数
    ----
    base_dir : str
        CSV 文件所在目录（默认 "save"）。
    datasets : tuple[str]
        需要绘制的 3 个数据集名称（与文件名里的 <Dataset> 部分对应，大小写不敏感；
        常见别名会做归一化：cifar/cifar10 -> cifar；fmnist/fashionmnist -> f-mnist）。
    attacks : list[str] | None
        指定攻击显示顺序（与文件名里的 <Variant> 精确匹配）。为 None 时自动发现；
        自动模式会把 "NoAttack" 放在首位，其余按字母序。最终最多保留 9 个。
    out_dir_name : str
        输出子目录名（默认 "plots"），图会保存到 base_dir/out_dir_name/ 下。

    约定的 CSV 命名：
        Trimmed-mean_<Variant>_<Dataset>_accuracy.csv
      例如：
        Trimmed-mean_PoisonedFL_cifar_accuracy.csv
        Trimmed-mean_NoAttack_f-mnist_accuracy.csv
    """
    import os
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    def _canon(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    def _is_trimmed_mean(defense: str) -> bool:
        # 兼容 Trimmed-mean / Trimmed_mean / Trimmed mean / TrimMean 等写法
        return _canon(defense) in {"trimmedmean", "trimmean", "trimmedmeans"}

    def _norm_dataset(name: str) -> str:
        c = _canon(name)
        if c in {"cifar", "cifar10"}:
            return "cifar"
        if c in {"fmnist", "fashionmnist"}:
            return "f-mnist"
        if c in {"mnist"}:
            return "mnist"
        return name  # 未知名保持原样

    # 目标数据集集合（做规范化后比较）
    target_keys = {_canon(_norm_dataset(ds)) for ds in datasets}

    # 收集：groups[dataset] = { variant: csv_path }
    pattern = re.compile(
        r'^(?P<defense>[^_]+)_(?P<variant>[^_]+)_(?P<dataset>[^_]+)_accuracy\.csv$',
        re.IGNORECASE
    )
    groups = {}

    if not os.path.isdir(base_dir):
        print(f"❌ 目录不存在：{base_dir}")
        return

    for fname in os.listdir(base_dir):
        if not fname.lower().endswith(".csv"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        defense, variant, dataset_raw = m.group("defense"), m.group("variant"), m.group("dataset")
        if not _is_trimmed_mean(defense):
            continue
        dataset = _norm_dataset(dataset_raw)
        if _canon(dataset) not in target_keys:
            continue
        groups.setdefault(dataset, {})[variant] = os.path.join(base_dir, fname)

    if not groups:
        print("⚠️ 未发现 Trimmed-mean 的匹配 CSV。")
        return

    # 组装输出目录
    out_dir = os.path.join(base_dir, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    def _order_variants(variants: list[str]) -> list[str]:
        if attacks:
            ordered = [v for v in attacks if v in variants]
        else:
            no_attack = [v for v in variants if v.lower() == "noattack"]
            others = sorted([v for v in variants if v not in no_attack])
            ordered = no_attack + others
        return ordered[:9]  # 最多 9 条

    def _read_accuracy_series(csv_path: str) -> pd.Series:
        df = pd.read_csv(csv_path)
        acc_cols = [c for c in df.columns if 'acc' in c.lower()]
        if acc_cols:
            s = pd.to_numeric(df[acc_cols[0]], errors='coerce')
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            s = pd.to_numeric(df[numeric_cols[0]] if numeric_cols else df.iloc[:, 0], errors='coerce')
        s = s.dropna().reset_index(drop=True)
        s.index = s.index + 1  # 回合数从 1 开始
        return s

    saved = []
    # 按传入 datasets 的顺序画
    for ds in datasets:
        ds_key = _norm_dataset(ds)
        if ds_key not in groups or not groups[ds_key]:
            print(f"⚠️ 跳过 {ds_key}：未找到该数据集下的 Trimmed-mean CSV。")
            continue

        variant_to_path = groups[ds_key]
        variants = list(variant_to_path.keys())
        ordered = _order_variants(variants)
        if len(ordered) < 9:
            print(f"ℹ️ 数据集 {ds_key} 仅发现 {len(ordered)} 种攻击（少于 9）。")

        # 读取曲线
        series_map = {}
        for v in ordered:
            try:
                series_map[v] = _read_accuracy_series(variant_to_path[v])
            except Exception as e:
                print(f"读取失败: {variant_to_path[v]} -> {e}")

        if not series_map:
            print(f"⚠️ 跳过 {ds_key}：无法读取任何曲线。")
            continue

        # 画图（不指定颜色、不用 seaborn、单轴）
        plt.figure()
        max_len = max(len(s) for s in series_map.values())
        for v in ordered:
            if v in series_map:
                s = series_map[v]
                plt.plot(s.index, s.values, label=v)
        plt.title(f"Trimmed-mean on {ds_key} — Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.xlim(1, max_len)
        plt.legend(ncol=1)

        out_path = os.path.join(out_dir, f"Trimmed-mean_{ds_key}_9attacks.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out_path)

    if saved:
        print("✅ 生成的图：")
        for p in saved:
            print(" -", p)

if __name__ == "__main__":
    plot_trimmed_mean_3datasets(base_dir="save")
    # 如 CSV 不在当前目录，改成你的目录：make_plots("/path/to/csv_dir")
    #make_plots("save")

