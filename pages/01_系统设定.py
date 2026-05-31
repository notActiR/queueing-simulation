"""
第 1 章: 系统设定 — 搭建我们的服务器
"""

import streamlit as st
import numpy as np

st.set_page_config(page_title="系统设定", page_icon="", layout="wide")

# 从 URL 查询参数获取全局设定或使用默认值
lam = st.session_state.get('lam', 0.7)
mu = st.session_state.get('mu', 1.0)

st.title("第一章：系统设定 — 搭建我们的服务器")

st.markdown("""
在开始仿真之前，让我们先理解 M/E₂/1 排队系统的基本构成。

一个计算服务器接收来自用户的任务请求。任务的到达时间是**随机的**（泊松过程），每个任务需要的处理时间也是**随机的**（Erlang-2 分布）。服务器一次只能处理一个任务，其他任务需要排队等待。
""")

# ---- 参数设置 ----
st.markdown("---")
st.markdown("##   调整系统参数")

col1, col2, col3 = st.columns(3)

with col1:
    lam = st.slider("到达率 λ", 0.1, 5.0, 0.7, 0.1,
                    help="单位时间内平均到达的任务数")
with col2:
    mu = st.slider("服务率 μ", 0.5, 8.0, 1.0, 0.1,
                   help="单位时间内服务器能完成的平均任务数")
with col3:
    # 快速预设
    preset = st.selectbox("快速预设 ρ", ["自定义", "低负载 0.3", "中负载 0.5", "中高负载 0.7", "高负载 0.85"])
    if preset == "低负载 0.3":
        lam, mu = 0.3, 1.0
    elif preset == "中负载 0.5":
        lam, mu = 0.5, 1.0
    elif preset == "中高负载 0.7":
        lam, mu = 0.7, 1.0
    elif preset == "高负载 0.85":
        lam, mu = 0.85, 1.0

rho = lam / mu
Cs2 = 0.5  # Erlang-2 的变异系数平方

if rho >= 1.0:
    E_X_theory = float('inf')
else:
    E_X_theory = rho + rho**2 * (1 + Cs2) / (2 * (1 - rho))

# 存储到 session state
st.session_state['lam'] = lam
st.session_state['mu'] = mu
st.session_state['rho'] = rho

# ---- 关键指标 ----
st.markdown("---")
st.markdown("##   关键指标")

metrics_cols = st.columns(4)
with metrics_cols[0]:
    st.metric("系统负载 ρ", f"{rho:.3f}",
              delta="稳定" if rho < 1 else "⚠️ 不稳定",
              delta_color="normal" if rho < 1 else "off")
with metrics_cols[1]:
    st.metric("理论 E[X]", f"{E_X_theory:.3f}" if rho < 1 else "∞",
              help="稳态下系统中的平均任务数")
with metrics_cols[2]:
    if rho < 1:
        theory_W = rho * (1 + Cs2) / (2 * mu * (1 - rho))
        st.metric("理论 E[W]", f"{theory_W:.3f}",
                  help="稳态下的平均等待时间 (Pollaczek-Khinchin 公式)")
    else:
        st.metric("理论 E[W]", "∞")
with metrics_cols[3]:
    avg_service = 1.0 / mu
    st.metric("平均服务时间", f"{avg_service:.3f}",
              help="1/μ")

# ---- 模型解释 ----
st.markdown("---")
st.markdown("##   模型详解")

tab1, tab2, tab3 = st.tabs(["泊松到达 (M)", "Erlang-2 服务 (E₂)", "理论公式"])

with tab1:
    st.markdown(rf"""
    **泊松到达过程**

    - 到达间隔服从指数分布: $\text{{Inter-arrival}} \sim \mathrm{{Exp}}(\lambda = {lam})$
    - 平均到达间隔: $1/\lambda = {1/lam:.3f}$ 时间单位
    - **无记忆性**: 上次到达后过了多久不影响下次到达的时间
    - 在任意长度 $T$ 的时间区间内, 到达数 $\sim \mathrm{{Poisson}}(\lambda T)$
    """)

with tab2:
    st.markdown(rf"""
    **Erlang-2 服务时间分布**

    - 服务时间 = 两个独立 $\mathrm{{Exp}}(2\mu)$ 之和, $2\mu = {2*mu}$
    - 平均服务时间: $1/\mu = {1/mu:.3f}$
    - 变异系数 $C_s^2 = 0.5$ (比指数分布的 $C_s^2=1$ 更"确定")
    - 相当于服务过程分为两个独立阶段, 每阶段耗时 $\mathrm{{Exp}}(2\mu)$
    """)

with tab3:
    st.latex(r"E[X] = \rho + \frac{\rho^2 (1 + C_s^2)}{2(1 - \rho)}")
    st.markdown(rf"""
    代入 $C_s^2 = 0.5$ (Erlang-2):
    """)
    st.latex(r"E[X] = \rho + \frac{0.75 \cdot \rho^2}{1 - \rho}")
    st.markdown(rf"""
    当前参数: $\rho = {rho:.3f}$, $E[X] = {E_X_theory:.3f}$
    """)

# ---- 告警 ----
if rho >= 0.9:
    st.error("⚠️  ρ ≥ 0.9: 系统接近饱和, 队长和等待时间将急剧增长!")
elif rho >= 1.0:
    st.error("  ρ ≥ 1.0: 系统不稳定! 队列将无限增长, 无法达到稳态。请降低 λ 或提高 μ。")

st.markdown("---")
st.caption("下一步: 前往「单次仿真」页面, 观察一次运行中四个随机过程的演化!")
