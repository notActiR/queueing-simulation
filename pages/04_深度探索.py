"""
第 4 章: 深度探索 — ρ 改变一切
"""

import streamlit as st
import numpy as np
from simulation.engine import ME2Queue
from visualization.analysis_charts import build_rho_comparison

st.set_page_config(page_title="深度探索", page_icon="", layout="wide")

st.title("  第四章：深度探索 — ρ 改变一切")

st.markdown(r"""
排队系统最重要的参数是**系统负载 $\rho = \lambda/\mu$**。

本章将对比两种极端场景，让你直观感受 $\rho$ 对系统性能的巨大影响。
""")

# ---- 预设参数 ----
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("###   低负载场景 (ρ ≈ 0.3)")
    st.markdown("""
    服务器大部分时间空闲, 任务几乎不需要等待。
    这是"理想"但低效的系统。
    """)
    rho_low = st.slider("ρ_low", 0.1, 0.45, 0.3, 0.05, key='rho_low')

with col2:
    st.markdown("###   高负载场景 (ρ ≈ 0.85)")
    st.markdown("""
    服务器接近满负荷运行, 队列经常积压。
    这是"高效"但风险很大的系统。
    """)
    rho_high = st.slider("ρ_high", 0.55, 0.95, 0.85, 0.05, key='rho_high')

mu_val = st.slider("服务率 μ (两边相同)", 0.5, 5.0, 1.0, 0.1)

lam_low = rho_low * mu_val
lam_high = rho_high * mu_val

n_customers = st.slider("仿真任务数", 100, 1000, 300, 50)
seed = st.number_input("随机种子 (两边相同)", 0, 99999, 42)

# ---- 运行对比仿真 ----
col1, col2 = st.columns(2)

with col1:
    with st.spinner("运行低负载仿真..."):
        sim_low = ME2Queue(lam_low, mu_val, seed=seed)
        result_low = sim_low.run(n_customers)

    E_X_low = result_low.E_X_theory
    Cs2 = 0.5
    theory_W_low = rho_low * (1 + Cs2) / (2 * mu_val * (1 - rho_low))

    st.metric("理论 E[X]", f"{E_X_low:.2f}" if E_X_low < 100 else f"{E_X_low:.1f}")
    st.metric("仿真平均 Nₖ", f"{np.mean(result_low.Nk):.2f}")
    st.metric("理论 E[W]", f"{theory_W_low:.3f}")
    st.metric("仿真平均 Wₖ", f"{np.mean(result_low.wait_times):.3f}")
    st.metric("零等待比例", f"{np.mean(result_low.wait_times < 1e-10):.1%}")

with col2:
    with st.spinner("运行高负载仿真..."):
        sim_high = ME2Queue(lam_high, mu_val, seed=seed)
        result_high = sim_high.run(n_customers)

    E_X_high = result_high.E_X_theory
    theory_W_high = rho_high * (1 + Cs2) / (2 * mu_val * (1 - rho_high))

    st.metric("理论 E[X]", f"{E_X_high:.2f}" if E_X_high < 100 else f"{E_X_high:.1f}")
    st.metric("仿真平均 Nₖ", f"{np.mean(result_high.Nk):.2f}")
    st.metric("理论 E[W]", f"{theory_W_high:.3f}")
    st.metric("仿真平均 Wₖ", f"{np.mean(result_high.wait_times):.3f}")
    st.metric("零等待比例", f"{np.mean(result_high.wait_times < 1e-10):.1%}")

# ---- 对比图 ----
st.markdown("---")
st.markdown("##   并排对比: $N_k$ 与 $W_k$")

fig_comp = build_rho_comparison(result_low, result_high, max_customers=n_customers)
st.plotly_chart(fig_comp, use_container_width=True)

# ---- 关键发现 ----
st.markdown("---")
st.markdown(r"##   关键发现: $(1-\rho)^{-1}$ 统治一切")

ratio_W = theory_W_high / theory_W_low if theory_W_low > 0 else float('inf')
ratio_rho_factor = (1 - rho_low) / (1 - rho_high)

st.markdown(rf"""
| 指标 | $\rho={rho_low:.2f}$ | $\rho={rho_high:.2f}$ | 比值 |
|------|------|------|------|
| 系统负载 $\rho$ | {rho_low:.2f} | {rho_high:.2f} | {rho_high/rho_low:.1f}$\times$ |
| $1/(1-\rho)$ | {1/(1-rho_low):.2f} | {1/(1-rho_high):.2f} | **{1/(1-rho_high) / (1/(1-rho_low)):.1f}$\times$** |
| 理论 $E[X]$ | {E_X_low:.2f} | {E_X_high:.2f} | {E_X_high/E_X_low if E_X_low > 0 else float('inf'):.1f}$\times$ |
| 理论平均等待 $E[W]$ | {theory_W_low:.3f} | {theory_W_high:.3f} | **{ratio_W:.1f}$\times$** |
""")

st.info(rf"""
💡 **核心洞察:**

当 $\rho$ 从 {rho_low:.2f} 增加到 {rho_high:.2f} (仅增加 {(rho_high/rho_low - 1)*100:.0f}%)，
平均等待时间增长了 **{ratio_W:.0f} 倍**。

这是因为理论上 $E[W] \propto \rho/(1-\rho)$。随着 $\rho \to 1$, $1/(1-\rho) \to \infty$, 系统性能急剧恶化。

**系统设计启示:**
- 在保证服务质量的前提下, 不能让 $\rho$ 太高
- $\rho = 0.7$ 左右是常见的工程折衷: 服务器利用率 70%, 等待时间可控
- 对于关键系统, 建议 $\rho \leq 0.5$, 给突发流量留足缓冲
""")

st.markdown("---")
st.caption("最后一步: 前往「总结」页面, 回顾我们学到的一切。")
