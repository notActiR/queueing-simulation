"""
第 3 章: 统计实验 — 30 次独立仿真揭示规律
"""

import streamlit as st
import numpy as np
from simulation.engine import ME2Queue
from simulation.statistics import run_multiple_simulations
from visualization.analysis_charts import (
    build_convergence_plot,
    build_distribution_evolution,
    build_acf_plot,
    build_convergence_paths,
)

st.set_page_config(page_title="统计实验", page_icon="", layout="wide")

st.title("  第三章：统计实验 — 从随机中寻找规律")

st.markdown(r"""
单次仿真展现了随机过程的**一次实现**, 但统计规律需要大量重复实验才能浮现。

本章进行 **30 次独立仿真** (每次使用不同的随机种子), 对 $\{N_k\}$ 过程进行系统的统计分析。
""")

# ---- 参数 ----
lam = st.session_state.get('lam', 0.7)
mu = st.session_state.get('mu', 1.0)
rho = lam / mu

col1, col2, col3 = st.columns(3)
with col1:
    n_runs = st.slider("独立仿真次数", 10, 100, 30, 5,
                       help="越多估计越精确, 但计算时间更长")
with col2:
    n_customers = st.slider("每次仿真的任务数", 200, 5000, 1000, 100)
with col3:
    base_seed = st.number_input("基础种子", 0, 99999, 42)

if rho >= 1.0:
    st.error("⚠️ ρ ≥ 1.0, 系统不稳定。请先在「系统设定」页面调整参数。")
    st.stop()

# ---- 运行统计分析 ----
@st.cache_data(ttl=300, show_spinner=False)
def run_stats_cached(lam, mu, n_runs, n_customers, base_seed):
    return run_multiple_simulations(lam, mu, n_runs, n_customers, base_seed)

with st.spinner(f"  正在运行 {n_runs} 次独立仿真, 请稍候..."):
    stats = run_stats_cached(lam, mu, n_runs, n_customers, base_seed)

# ---- 3.1 均值-方差收敛图 ----
st.markdown("---")
st.markdown("## 3.1 均值-方差收敛图")

st.markdown(r"""
$E[N_k]$ 和 $\mathrm{{Var}}[N_k]$ 随 $k$ 的变化反映了系统从**瞬态**到**稳态**的收敛过程。

- 小 $k$ 时: 系统刚从空状态开始, 统计量受初始条件影响
- 大 $k$ 时: 统计量应趋于理论稳态值
""")

fig_convergence = build_convergence_plot(stats)
st.plotly_chart(fig_convergence, use_container_width=True)

# 核心数值
col1, col2, col3 = st.columns(3)
with col1:
    last_k = stats.k_values[-1]
    st.metric(f"E[N{last_k}]", f"{stats.means[-1]:.3f}",
              delta=f"理论 {stats.E_X_theory:.3f}",
              delta_color="off")
with col2:
    st.metric(f"Var[N{last_k}]", f"{stats.variances[-1]:.3f}")
with col3:
    ci_width = stats.ci_upper[-1] - stats.ci_lower[-1]
    st.metric(f"95% CI 宽度 (k={last_k})", f"{ci_width:.3f}",
              help="置信区间越窄, 估计越精确")

# ---- 3.2 分布演变图 ----
st.markdown("---")
st.markdown("## 3.2 分布演变图")

st.markdown(r"""
从瞬态到稳态, $\{N_k\}$ 的**分布形态**经历了怎样的变化?

下图用箱线图展示选定 $k$ 值下, 30 个样本的分布。红色虚线为理论稳态值。
""")

fig_dist = build_distribution_evolution(stats)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown(r"""
**观察窗**:
- $k=1$: 第一个任务离开时, 系统几乎总是空的 ($N_1$ 通常 = 0 或很小)
- $k=1000$: 分布趋于稳定, 中位数和方差接近理论预期
- 箱线图的展开反映了 $N_k$ 的变异性: 即使稳态下, 系统人数也有显著波动
""")

# ---- 3.3 自相关分析 ----
st.markdown("---")
st.markdown("## 3.3 自相关分析")

st.markdown(r"""
$\{N_k\}$ 是**马尔可夫链**, 理论上当前状态只依赖前一个状态。自相关函数 (ACF) 可以验证这个性质:

如果 ACF 快速衰减 (呈指数衰减模式), 说明序列的"记忆"很短, 符合马尔可夫性。
""")

fig_acf = build_acf_plot(stats, max_lag=30)
st.plotly_chart(fig_acf, use_container_width=True)

st.markdown(rf"""
**自相关分析要点:**

- $\rho^{{\mathrm{{lag}}}}$ (红色虚线) 给出了一个理论衰减参考
- 如果 ACF 在 lag=1 后迅速降到置信带内, 则体现了接近马尔可夫链的性质
- 实际上, $N_k$ 是 $N_{{k-1}}$ 加上 $k$ 到 $k+1$ 之间的新到达数减去 1 (第 $k$ 个离开), 这是典型的随机游走结构
""")

# ---- 3.4 收敛路径图 ----
st.markdown("---")
st.markdown("## 3.4 收敛路径图")

fig_paths = build_convergence_paths(stats)
st.plotly_chart(fig_paths, use_container_width=True)

st.markdown(r"""
每条**灰色细线**代表一次独立仿真。蓝色粗线是 30 条路径的均值。

可以看到:
- 初期 (小 $k$): 各条路径差异明显
- 后期 (大 $k$): 路径在理论值附近聚集, 但仍有波动
- 30 次仿真的均值已经很接近理论值
""")

st.markdown("---")
st.caption("下一步: 前往「深度探索」对比不同 ρ 下系统的行为差异!")
