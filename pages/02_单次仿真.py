"""
第 2 章: 单次仿真 — 观察一次运行中四个随机过程的联动演化
"""

import streamlit as st
import numpy as np
from simulation.engine import ME2Queue
from visualization.four_processes import build_four_process_figure, build_wait_histogram

st.set_page_config(page_title="单次仿真", page_icon="", layout="wide")

st.title("  第二章：单次仿真 — 四过程联动")

st.markdown(r"""
让我们运行一次 M/E$_2$/1 排队仿真, 观察系统中四个随机过程的同步演化。

**四个过程:**
- $\{N_k\}$: 第 $k$ 个任务离开时, 系统中还剩多少任务 (嵌入马尔可夫链)
- $\{X_t\}$: 任意时刻 $t$, 系统中有多少任务
- $\{W_k\}$: 第 $k$ 个任务排了多久的队
- $\{Y_t\}$: 任意时刻 $t$, 系统中的累计服务需求
""")

# ---- 参数 ----
lam = st.session_state.get('lam', 0.7)
mu = st.session_state.get('mu', 1.0)
rho = lam / mu

col1, col2, col3, col4 = st.columns(4)
with col1:
    n_customers = st.slider("仿真任务数", 50, 2000, 300, 50,
                            help="任务数越多, 越能看到稳态行为")
with col2:
    seed = st.number_input("随机种子", 0, 99999, 42, step=1,
                           help="改变种子可以看到不同的随机实现")
with col3:
    max_show = st.slider("显示任务数", 50, min(500, n_customers), min(200, n_customers), 50,
                         help="图表中显示的任务数 (太多会看不清)")
with col4:
    if st.button("  换一个随机种子"):
        seed = np.random.randint(0, 99999)
        st.rerun()

# ---- 运行仿真 ----
if rho >= 1.0:
    st.error("⚠️ ρ ≥ 1.0, 系统不稳定。请先在「系统设定」页面调整参数。")
    st.stop()

sim = ME2Queue(lam, mu, seed=seed)
result = sim.run(n_customers)

# ---- 四过程面板 ----
st.markdown("---")
st.markdown("##   四个随机过程的联动展示")

fig = build_four_process_figure(result, max_customers=max_show)
st.plotly_chart(fig, use_container_width=True)

# ---- 关键数据卡片 ----
st.markdown("---")
st.markdown("##   关键数据")

k0 = result.get_first_empty_k()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    avg_N = np.mean(result.Nk)
    st.metric("平均队长 E[Nₖ]", f"{avg_N:.2f}",
              delta=f"理论 {result.E_X_theory:.2f}" if np.isfinite(result.E_X_theory) else None,
              delta_color="off")
with c2:
    avg_W = np.mean(result.wait_times)
    st.metric("平均等待时间", f"{avg_W:.3f}")
with c3:
    busy_ratio = np.sum(result.service_times) / result.departure_times[-1]
    st.metric("服务器利用率", f"{busy_ratio:.1%}",
              delta=f"理论 {rho:.1%}",
              delta_color="off")
with c4:
    st.metric("首次空闲时刻 k₀", str(k0) if k0 > 0 else "从未空闲",
              help="系统第一次变空时的离开序号")
with c5:
    zero_wait_pct = np.mean(result.wait_times < 1e-10)
    st.metric("零等待比例", f"{zero_wait_pct:.1%}")

# ---- 等待时间直方图 ----
st.markdown("---")
col_left, col_right = st.columns([1, 2])
with col_left:
    st.markdown("### ⏳ 等待时间分布")
    hist_fig = build_wait_histogram(result, max_customers=min(n_customers, 500))
    st.plotly_chart(hist_fig, use_container_width=True)

with col_right:
    st.markdown("###   观察要点")
    k0_str = str(k0) if k0 > 0 else 'N/A'
    st.markdown(rf"""
    - **$\{{N_k\}}$ 的颜色变化**: 绿色(空闲) → 黄色(正常) → 红色(繁忙), 反映了系统负载的波动
    - **首次空闲 $k_0 = {k0_str}$**: 系统第一次回到空闲状态的时刻, 之后 $\{{N_k\}}$ 分布趋于稳定
    - **$\{{X_t\}}$ 的锯齿**: 到达事件导致向上跳, 离开事件导致向下跳
    - **$\{{W_k\}}$ 的趋势**: 移动平均线反映了等待时间的整体水平, 初期可能有瞬态效应
    - **$\{{Y_t\}}$ 的堆叠面积**: 上方面积(红色)是当前系统内的剩余服务需求, 下方面积(绿色)是已完成的服务
    """)

st.markdown("---")
st.caption("  在图上悬停鼠标可以查看详细数据。下一步: 前往「统计实验」看看 30 次独立仿真能揭示什么规律。")
