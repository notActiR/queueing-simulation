"""
M/E₂/1 排队系统仿真与可视化分析 — 叙事性 Web 应用

启动方式:
  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="M/E₂/1 排队系统仿真",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 封面页 (主页) ----
st.title("M/E₂/1 排队系统的仿真与可视化分析")
st.markdown("### 《概率论与随机过程》期中小组探究项目")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ###   探索之旅

    本应用将带你一步步深入理解排队系统的随机行为：

    1.  **系统设定** — 搭建我们的服务器，理解 M/E₂/1 模型
    2.  **单次仿真** — 观察一次运行中四个随机过程的联动演化
    3.  **统计实验** — 30 次独立仿真揭示统计规律与收敛性
    4.  **深度探索** — 低负载 vs 高负载的对比实验
    5.  **总结思考** — 回答理论问题，提炼系统设计启示

    ---

    每个页面都包含**交互控件**，你可以拖动滑块、切换参数，实时观察结果变化。
    """)

with col2:
    st.info(r"""
    **M/E$_2$/1 模型简介**

    - **M**: 泊松到达过程
    - **E$_2$**: Erlang-2 服务时间
    - **1**: 单服务器

    这是排队论中的经典模型，其嵌入马尔可夫链 $\{N_k\}$ 是研究系统瞬态与稳态行为的核心工具。
    """)

st.markdown("---")
st.markdown("###  请在左侧边栏选择章节开始探索")

# 侧边栏
with st.sidebar:
    st.markdown("##   章节导航")
    st.markdown("点击左侧页面标签切换章节")

    st.divider()

    st.markdown("### 全局参数")
    lam = st.slider("到达率 λ", 0.1, 5.0, 0.7, 0.1,
                    help="单位时间到达的任务平均数")
    mu = st.slider("服务率 μ", 0.5, 8.0, 1.0, 0.1,
                   help="单位时间可完成服务的任务平均数")

    rho = lam / mu
    st.metric("系统负载 ρ = λ/μ", f"{rho:.3f}",
              delta="稳态" if rho < 1 else "⚠️ 系统不稳定",
              delta_color="normal" if rho < 1 else "off")

    st.divider()

    st.caption("使用左侧页面标签在各章节间切换")
    st.caption("当前页面: **封面**")
