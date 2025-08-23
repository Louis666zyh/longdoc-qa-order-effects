# -*- coding: utf-8 -*-
# Paired t-tests with fixed file paths (简洁结果版)
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import ttest_rel
import os

# ===== 固定路径 =====
FILE_COLBERT   = "top12345withcolbert/final_results_stage1.csv"
FILE_BM25ONLY  = "average_bm25_only/merged/final_results_stage1_bm25only_merged.csv"
FILE_BASELINE  = "baseline/final_results_stage1-baseline.csv"
OUT_DIR = "ttest_out"

METRICS = ["rouge1","rouge2","rougeL","bleu","bertscore"]
ALPHA   = 0.05

# ---------- 工具 ----------
def _to_num(df):
    for c in METRICS:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def norm_strategy_val(x: str):
    s = str(x).strip().lower()
    if "base" in s:   return "baseline"
    if "shuffl" in s: return "shuffled"
    if "rev" in s:    return "reversed"
    if "seq" in s:    return "sequential"
    return np.nan

def load_stage1(path):
    df = pd.read_csv(path)
    _to_num(df)
    if "qid" not in df.columns and "cid" in df.columns:
        df = df.rename(columns={"cid":"qid"})
    assert "qid" in df.columns, f"{path} 缺少 qid"
    df["strategy_name"] = df["strategy"].apply(norm_strategy_val)
    if "topk" in df.columns:
        df["topk_int"] = pd.to_numeric(df["topk"], errors="coerce").astype("Int64")
    else:
        ext = df["strategy"].astype(str).str.extract(r"top\s*[-_ ]?(\d+)")
        df["topk_int"] = pd.to_numeric(ext[0], errors="coerce").astype("Int64")
    return df

def load_baseline(path):
    df = pd.read_csv(path)
    _to_num(df)
    if "qid" not in df.columns and "cid" in df.columns:
        df = df.rename(columns={"cid":"qid"})
    assert "qid" in df.columns, f"{path} 缺少 qid"
    if "strategy" in df.columns:
        df = df[df["strategy"].apply(norm_strategy_val).eq("baseline")]
    return df

def join_on_qid(A, B, la="_A", lb="_B"):
    A = A.set_index("qid").add_suffix(la)
    B = B.set_index("qid").add_suffix(lb)
    return A.join(B, how="inner").reset_index()

def do_tests(M, retrieval, meta):
    rows=[]
    for m in METRICS:
        xa = pd.to_numeric(M[f"{m}_A"], errors="coerce")
        yb = pd.to_numeric(M[f"{m}_B"], errors="coerce")
        mask = (~xa.isna()) & (~yb.isna()); xa, yb = xa[mask], yb[mask]
        if len(xa) < 5: continue
        t,p = ttest_rel(xa, yb, nan_policy="omit")
        rows.append(dict(retrieval=retrieval, metric=m, N_pairs=len(xa),
                         p_raw=float(p), **meta))
    return rows

def holm_bonferroni(pvals, alpha=0.05):
    p = np.array(pvals, float); order = np.argsort(p); m=len(p)
    adj = np.empty_like(p); run = 0.0
    for r, idx in enumerate(order):
        val = p[idx]*(m-r); run = max(run, val); adj[idx] = min(1.0, run)
    return adj.tolist(), (adj<alpha).tolist()

# ---------- 三类检验 ----------
def run_RQ2(df, retrieval):
    out=[]
    topks = sorted([int(k) for k in df["topk_int"].dropna().unique().tolist()])
    for k in topks:
        avail = set(df.loc[df["topk_int"]==k, "strategy_name"].dropna().unique())
        for a,b in [("reversed","sequential"),("reversed","shuffled"),("sequential","shuffled")]:
            if {a,b} <= avail:
                A=df[(df["strategy_name"]==a)&(df["topk_int"]==k)][["qid"]+METRICS]
                B=df[(df["strategy_name"]==b)&(df["topk_int"]==k)][["qid"]+METRICS]
                M=join_on_qid(A,B)
                if not M.empty:
                    out += do_tests(M, retrieval, dict(tag="RQ2_order_effect", topk=k, compare=f"{a} - {b}"))
    out=pd.DataFrame(out)
    if out.empty: return out
    out["p_adj"]=np.nan; out["significant@0.05"]=False
    for (r,tk,met),g in out.groupby(["retrieval","topk","metric"]):
        idx=g.index; p_adj,sig=holm_bonferroni(g["p_raw"].tolist(), ALPHA)
        out.loc[idx,"p_adj"]=p_adj; out.loc[idx,"significant@0.05"]=sig
    return out[["retrieval","tag","topk","compare","metric","N_pairs","p_raw","p_adj","significant@0.05"]]

def run_RQ1(df, retrieval):
    out=[]; S=df[df["strategy_name"]=="sequential"].copy()
    topks = sorted([int(k) for k in S["topk_int"].dropna().unique().tolist()])
    for k1,k2 in combinations(topks,2):
        A=S[S["topk_int"]==k1][["qid"]+METRICS]; B=S[S["topk_int"]==k2][["qid"]+METRICS]
        M=join_on_qid(A,B)
        if not M.empty:
            out += do_tests(M, retrieval, dict(tag="RQ1_context_length",
                                               topk=f"{k1} vs {k2}", compare=f"top{k1} - top{k2}"))
    out=pd.DataFrame(out)
    if out.empty: return out
    out["p_adj"]=np.nan; out["significant@0.05"]=False
    for (r,tk,met),g in out.groupby(["retrieval","topk","metric"]):
        idx=g.index; p_adj,sig=holm_bonferroni(g["p_raw"].tolist(), ALPHA)
        out.loc[idx,"p_adj"]=p_adj; out.loc[idx,"significant@0.05"]=sig
    return out[["retrieval","tag","topk","compare","metric","N_pairs","p_raw","p_adj","significant@0.05"]]

def run_BASELINE(df, df_base, retrieval):
    out=[]
    topks = sorted([int(k) for k in df["topk_int"].dropna().unique().tolist()])
    for k in topks:
        A=df[(df["strategy_name"]=="sequential")&(df["topk_int"]==k)][["qid"]+METRICS]
        B=df_base[["qid"]+METRICS]; M=join_on_qid(A,B)
        if not M.empty:
            out += do_tests(M, retrieval, dict(tag="Baseline_vs_WithContext",
                                               topk=k, compare=f"sequential_top{k} - baseline"))
    out=pd.DataFrame(out)
    if out.empty: return out
    out["p_adj"]=np.nan; out["significant@0.05"]=False
    for (r,tk,met),g in out.groupby(["retrieval","topk","metric"]):
        idx=g.index; p_adj,sig=holm_bonferroni(g["p_raw"].tolist(), ALPHA)
        out.loc[idx,"p_adj"]=p_adj; out.loc[idx,"significant@0.05"]=sig
    return out[["retrieval","tag","topk","compare","metric","N_pairs","p_raw","p_adj","significant@0.05"]]

# ---------- 运行 ----------
df_col = load_stage1(FILE_COLBERT)
df_m25 = load_stage1(FILE_BM25ONLY)
df_base = load_baseline(FILE_BASELINE)

res_rq2_col  = run_RQ2(df_col,  "BM25+ColBERT")
res_rq2_m25  = run_RQ2(df_m25,  "BM25only")
res_rq1_col  = run_RQ1(df_col,  "BM25+ColBERT")
res_rq1_m25  = run_RQ1(df_m25,  "BM25only")
res_base_col = run_BASELINE(df_col,  df_base, "BM25+ColBERT")
res_base_m25 = run_BASELINE(df_m25,  df_base, "BM25only")

os.makedirs(OUT_DIR, exist_ok=True)
res_rq2_col.to_csv(os.path.join(OUT_DIR,"ttest_RQ2_colbert.csv"), index=False)
res_rq2_m25.to_csv(os.path.join(OUT_DIR,"ttest_RQ2_bm25only.csv"), index=False)
res_rq1_col.to_csv(os.path.join(OUT_DIR,"ttest_RQ1_colbert.csv"), index=False)
res_rq1_m25.to_csv(os.path.join(OUT_DIR,"ttest_RQ1_bm25only.csv"), index=False)
res_base_col.to_csv(os.path.join(OUT_DIR,"ttest_BASELINE_colbert.csv"), index=False)
res_base_m25.to_csv(os.path.join(OUT_DIR,"ttest_BASELINE_bm25only.csv"), index=False)

all_parts = [res_rq2_col,res_rq2_m25,res_rq1_col,res_rq1_m25,res_base_col,res_base_m25]
ttest_all = pd.concat([x for x in all_parts if not x.empty], ignore_index=True) if any([not x.empty for x in all_parts]) else pd.DataFrame()
ttest_all.to_csv(os.path.join(OUT_DIR,"ttest_ALL.csv"), index=False)
print("Saved under", OUT_DIR)

# -*- coding: utf-8 -*-
# 显著性结果热力图绘制

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 输入文件路径 =====
files = {
    "RQ1_bm25only": "ttest_out/ttest_RQ1_bm25only.csv",
    "RQ1_colbert": "ttest_out/ttest_RQ1_colbert.csv",
    "RQ2_bm25only": "ttest_out/ttest_RQ2_bm25only.csv",
    "RQ2_colbert": "ttest_out/ttest_RQ2_colbert.csv",
    "BASELINE_bm25only": "ttest_out/ttest_BASELINE_bm25only.csv",
    "BASELINE_colbert": "ttest_out/ttest_BASELINE_colbert.csv",
}

# ===== 热力图绘制函数 =====
def plot_heatmap(df, title, outpath):
    df = df.copy()
    # 转换显著性为 0/1
    df["sig"] = df["significant@0.05"].map({True:1, False:0})
    # 透视表：行=(topk, compare)，列=metric
    mat = df.pivot_table(index=["topk","compare"], columns="metric",
                         values="sig", aggfunc="first")

    plt.figure(figsize=(8, max(4, 0.5*len(mat))))
    sns.heatmap(mat, annot=True, cmap="Reds", cbar=False,
                linewidths=.5, linecolor="gray", fmt="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved] {outpath}")

# ===== 运行绘制 =====
for name, path in files.items():
    if "RQ1" in name or "RQ2" in name:  # baseline 不需要画热力图
        df = pd.read_csv(path)
        title = name.replace("_", " ")
        outpath = f"ttest_out/heatmap_{name}.png"
        plot_heatmap(df, title, outpath)
