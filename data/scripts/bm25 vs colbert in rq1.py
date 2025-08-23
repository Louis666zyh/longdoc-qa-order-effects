# ====== RQ1: BM25 vs ColBERT 显著性检验 ======
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

# ==== 修改成你的文件路径 ====
bm25_file = "average_bm25_only/merged/final_results_stage1_bm25only_merged.csv"
colbert_file = "top12345withcolbert/final_results_stage1.csv"

# 读取结果
df_bm25 = pd.read_csv(bm25_file)
df_colbert = pd.read_csv(colbert_file)

# inner merge 按照 (qid, topk) 对齐
df = pd.merge(df_bm25, df_colbert, on=["qid","topk"], suffixes=("_bm25", "_colbert"))

metrics = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore"]

results = []
for m in metrics:
    diff = df[f"{m}_colbert"] - df[f"{m}_bm25"]

    mean_diff = diff.mean()
    std_diff = diff.std()

    # 配对 t-test
    t_stat, p_t = ttest_rel(df[f"{m}_colbert"], df[f"{m}_bm25"])

    # Wilcoxon
    try:
        w_stat, p_w = wilcoxon(df[f"{m}_colbert"], df[f"{m}_bm25"])
    except ValueError:
        w_stat, p_w = None, 1.0

    results.append({
        "metric": m,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat, "p_ttest": p_t,
        "w_stat": w_stat, "p_wilcoxon": p_w
    })

# 保存结果
df_res = pd.DataFrame(results)
df_res.to_csv("rq1_bm25_vs_colbert_significance.csv", index=False, float_format="%.3e")

# 热力图可视化 mean_diff
plt.figure(figsize=(6,4))
sns.heatmap(df_res.set_index("metric")[["mean_diff"]], annot=True, cmap="RdBu_r", center=0, fmt=".3f")
plt.title("RQ1: BM25 vs ColBERT (Mean Difference)")
plt.xlabel("Mean Difference (ColBERT - BM25)")
plt.tight_layout()
plt.savefig("rq1_bm25_vs_colbert_heatmap.png", dpi=300)
plt.show()
