# ====== BM25 vs ColBERT 显著性检验（配对 + 多重比较 + 效应量/CI） ======
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, t as t_dist
from typing import List, Dict

# ==== 文件路径（请按需修改） ====
bm25_file = "average_bm25_only/merged/final_results_stage1_bm25only_merged.csv"
colbert_file = "top12345withcolbert/final_results_stage1.csv"

# ==== 配置 ====
metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore"]  # 需要在两侧 CSV 中都具备
alpha = 0.05
use_bootstrap = False         # 如需 bootstrap CI，把这个改为 True
n_boot = 5000                 # bootstrap 轮数
random_seed = 42

np.random.seed(random_seed)

# ==== 工具函数 ====
def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Return Holm-Bonferroni adjusted p-values (step-down)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    cummax = 0.0
    for rank, idx in enumerate(order):
        # step-down: (m - rank) * p_(rank)
        adjusted = (m - rank) * pvals[idx]
        # 保持非降约束
        cummax = max(cummax, adjusted)
        adj[idx] = min(1.0, cummax)
    # 反排回原顺序
    return adj.tolist()

def cohen_dz(diff: np.ndarray) -> float:
    """Cohen's dz for paired design: mean(diff) / std(diff, ddof=1)"""
    diff = np.asarray(diff)
    sd = diff.std(ddof=1)
    return np.nan if sd == 0 else diff.mean() / sd

def ci_mean_diff_t(diff: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """t 分布的均值差 置信区间"""
    n = diff.size
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    if n <= 1 or np.isnan(se):
        return {"ci_low": np.nan, "ci_high": np.nan}
    df = n - 1
    tcrit = t_dist.ppf(0.5 * (1 + conf), df)
    half = tcrit * se
    return {"ci_low": mean - half, "ci_high": mean + half}

def ci_mean_diff_bootstrap(diff: np.ndarray, conf: float = 0.95, n_boot: int = 5000) -> Dict[str, float]:
    """bootstrap 分位数 CI（较稳健，但开销更大）"""
    n = diff.size
    if n == 0:
        return {"ci_low": np.nan, "ci_high": np.nan}
    boots = np.random.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    low = np.quantile(boots, (1 - conf) / 2)
    high = np.quantile(boots, 1 - (1 - conf) / 2)
    return {"ci_low": float(low), "ci_high": float(high)}

# ==== 读取数据 ====
df_bm25 = pd.read_csv(bm25_file)
df_colbert = pd.read_csv(colbert_file)

# ==== 确定 join 键 ====
join_keys = ["qid"] + (["topk"] if ("topk" in df_bm25.columns and "topk" in df_colbert.columns) else [])

# ==== 检查 metric 列是否齐全 ====
missing_b = [m for m in metrics if m not in df_bm25.columns]
missing_c = [m for m in metrics if m not in df_colbert.columns]
if missing_b or missing_c:
    raise ValueError(f"缺少指标列：BM25缺{missing_b}，ColBERT缺{missing_c}")

# ==== inner merge，确保严格配对 ====
df = pd.merge(
    df_bm25[join_keys + metrics],
    df_colbert[join_keys + metrics],
    on=join_keys,
    suffixes=("_bm25", "_colbert")
)

if df.empty:
    raise ValueError("合并后为空，请检查 qid/topk 是否匹配以及输入文件。")

# ==== 逐指标做配对检验 ====
rows = []
for m in metrics:
    x = df[f"{m}_colbert"].to_numpy()
    y = df[f"{m}_bm25"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    diff = x - y
    n = diff.size

    if n < 2:
        # 样本太少，不做检验
        rows.append({
            "metric": m, "n": n,
            "mean_colbert": np.nan, "mean_bm25": np.nan,
            "mean_diff": np.nan, "std_diff": np.nan,
            "t_stat": np.nan, "p_ttest": np.nan,
            "w_stat": np.nan, "p_wilcoxon": np.nan,
            "dz": np.nan, "ci_low": np.nan, "ci_high": np.nan
        })
        continue

    mean_c = float(np.mean(x))
    mean_b = float(np.mean(y))
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    dz = cohen_dz(diff)

    # 配对 t-test
    t_stat, p_t = ttest_rel(x, y)

    # Wilcoxon（若差值全 0 会报错）
    try:
        w_stat, p_w = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", correction=False, mode="auto")
    except ValueError:
        w_stat, p_w = np.nan, 1.0

    # 95% CI
    if use_bootstrap:
        ci = ci_mean_diff_bootstrap(diff, conf=0.95, n_boot=n_boot)
    else:
        ci = ci_mean_diff_t(diff, conf=0.95)

    rows.append({
        "metric": m, "n": n,
        "mean_colbert": mean_c, "mean_bm25": mean_b,
        "mean_diff": mean_diff, "std_diff": std_diff,
        "t_stat": t_stat, "p_ttest": p_t,
        "w_stat": w_stat, "p_wilcoxon": p_w,
        "dz": dz, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]
    })

df_res = pd.DataFrame(rows)

# ==== Holm–Bonferroni 校正 ====
df_res["p_ttest_holm"] = holm_bonferroni(df_res["p_ttest"].fillna(1.0).tolist())
df_res["p_wilcoxon_holm"] = holm_bonferroni(df_res["p_wilcoxon"].fillna(1.0).tolist())

# 排版与输出
df_res = df_res[
    ["metric","n","mean_colbert","mean_bm25","mean_diff","std_diff",
     "dz","ci_low","ci_high","t_stat","p_ttest","p_ttest_holm","w_stat","p_wilcoxon","p_wilcoxon_holm"]
].sort_values("metric")

os.makedirs("figs", exist_ok=True)
df_res.to_csv("rq2_bm25_vs_colbert_stats.csv", index=False)

# ====== 图 1：Mean Difference 热力图 ======
plt.figure(figsize=(6, 3.6))
data_md = df_res.set_index("metric")[["mean_diff"]]
sns.heatmap(data_md, annot=True, fmt=".3f", center=0, cmap="coolwarm",
            cbar_kws={'label': 'ColBERT − BM25'})
plt.title("BM25 vs ColBERT — Mean Difference")
plt.tight_layout()
plt.savefig("figs/rq2_bm25_vs_colbert_mean_diff_heatmap.png", dpi=300)
plt.close()

# ====== 图 2：p 值热力图（Holm 校正后）=====
plt.figure(figsize=(6, 3.6))
data_p = df_res.set_index("metric")[["p_ttest_holm","p_wilcoxon_holm"]]
sns.heatmap(data_p, annot=True, fmt=".1e", cmap="coolwarm_r",
            cbar_kws={'label': 'adjusted p-value'})
plt.title("BM25 vs ColBERT — Holm-adjusted p-values")
plt.tight_layout()
plt.savefig("figs/rq2_bm25_vs_colbert_pvalues_heatmap.png", dpi=300)
plt.close()

# ====== 图 3：效应量 dz 热力图 ======
plt.figure(figsize=(6, 3.6))
data_dz = df_res.set_index("metric")[["dz"]]
sns.heatmap(data_dz, annot=True, fmt=".3f", center=0, cmap="coolwarm",
            cbar_kws={'label': "Cohen's dz"})
plt.title("BM25 vs ColBERT — Effect Size (dz)")
plt.tight_layout()
plt.savefig("figs/rq2_bm25_vs_colbert_effectsize_heatmap.png", dpi=300)
plt.close()

# ====== 分布图（可选）======
for m in metrics:
    if m not in df_res["metric"].values:
        continue
    # 重算差值以便画图（只画配对成功的行）
    x = df[f"{m}_colbert"].to_numpy()
    y = df[f"{m}_bm25"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    diff = (x - y)[mask]
    plt.figure(figsize=(6, 4))
    sns.histplot(diff, bins=30, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"Difference Distribution (ColBERT − BM25) — {m}")
    plt.xlabel("ColBERT − BM25")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"figs/rq2_diff_hist_{m}.png", dpi=300)
    plt.close()

print("\n=== Summary (BM25 vs ColBERT) ===")
print(df_res.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
print("\n已保存：rq2_bm25_vs_colbert_stats.csv 与 figs/*.png")
