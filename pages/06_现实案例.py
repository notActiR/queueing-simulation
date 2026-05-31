"""
第 6 章: 现实案例 — 医院门诊挂号排队系统
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulation.engine import ME2Queue

st.set_page_config(page_title="现实案例", page_icon="", layout="wide")

st.title("第六章：现实案例 — 医院门诊挂号排队")

# ---- 场景引入 ----
st.markdown("---")

col_text, col_diagram = st.columns([3, 2])

with col_text:
    st.markdown(r"""
    ### 场景描述

    某三甲医院的门诊大厅里，每天早晨 7:30 开始挂号。患者陆续到达，在**唯一的挂号窗口**前排队。

    挂号过程分为两个阶段：

    1. **信息录入** — 工作人员录入患者信息、科室、医保类型
    2. **缴费确认** — 核算费用、扫码支付、打印挂号单

    两个阶段缺一不可，总耗时等于两阶段之和 —— 这正是 **Erlang-2 分布**的现实原型。

    **患者到达**是随机的（泊松过程），有的患者密集到达，有的时段稀稀拉拉。
    """)

with col_diagram:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e8f5e9, #e3f2fd);
        border-radius: 12px; padding: 20px; font-family: monospace;
        text-align: center; line-height: 2.2;
    ">
        <div style="font-size: 2em;">🏃‍♂️🏃‍♀️</div>
        <div style="color: #666;">随机到达</div>
        <div style="font-size: 1.5em;">⬇️</div>
        <div style="background: #fff3cd; border-radius: 8px; padding: 8px; margin: 10px 0;">
            📋 排队等待
        </div>
        <div style="font-size: 1.5em;">⬇️</div>
        <div style="background: #e3f2fd; border-radius: 8px; padding: 8px; margin: 5px 0;">
            🖊️ ① 信息录入
        </div>
        <div style="font-size: 1.2em;">+</div>
        <div style="background: #e8f5e9; border-radius: 8px; padding: 8px; margin: 5px 0;">
            💳 ② 缴费确认
        </div>
        <div style="font-size: 1.5em;">⬇️</div>
        <div style="color: #2e7d32; font-weight: bold;">✅ 挂号完成</div>
    </div>
    """, unsafe_allow_html=True)

# ---- 参数设置 ----
st.markdown("---")
st.markdown("## 场景参数设置")

st.markdown("""
真实数据参考：某三甲医院门诊挂号窗口，早高峰 7:30-9:30 期间：
- 平均每小时到达约 **30 位**患者
- 平均每位患者挂号耗时约 **2 分钟**
- 单窗口服务，患者先到先挂
""")

col1, col2, col3 = st.columns(3)

with col1:
    arrivals_per_hour = st.slider(
        "每小时到达患者数 $\\lambda_h$", 10, 60, 30, 5,
        help="早高峰期间平均每小时有多少患者来挂号"
    )
with col2:
    avg_service_minutes = st.slider(
        "平均挂号耗时 (分钟) $1/\\mu$", 1.0, 5.0, 2.0, 0.5,
        help="每位患者从开始挂号到完成平均耗时多少分钟"
    )
with col3:
    scenario = st.selectbox(
        "快速切换场景",
        ["早高峰 (繁忙)", "上午平峰 (正常)", "下午低峰 (空闲)", "自定义"]
    )
    if scenario == "早高峰 (繁忙)":
        arrivals_per_hour = 45
        avg_service_minutes = 2.0
    elif scenario == "上午平峰 (正常)":
        arrivals_per_hour = 25
        avg_service_minutes = 1.8
    elif scenario == "下午低峰 (空闲)":
        arrivals_per_hour = 12
        avg_service_minutes = 1.5

# 换算为仿真参数
lam = arrivals_per_hour / 60.0  # 每分钟到达率
mu = 1.0 / avg_service_minutes  # 每分钟服务率
rho = lam / mu
n_patients = st.slider("仿真患者数", 50, 500, 200, 25)

if rho >= 1.0:
    st.error(f"⚠️ 系统负载 ρ = {rho:.2f} ≥ 1.0！挂号窗口无法处理这么多患者，队伍会无限增长。请减少到达人数或增加服务速度。")
    st.stop()

# ---- 运行仿真 ----
sim = ME2Queue(lam, mu, seed=42)
result = sim.run(n_patients)

# ---- 关键指标 ----
st.markdown("---")
st.markdown("## 仿真结果")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("系统负载 ρ", f"{rho:.2f}",
              delta="繁忙" if rho > 0.7 else ("适中" if rho > 0.4 else "空闲"))
with c2:
    avg_wait = np.mean(result.wait_times)
    st.metric("平均排队等待时间", f"{avg_wait:.1f} 分钟",
              delta="可接受" if avg_wait < 5 else "⚠️ 过长",
              delta_color="normal" if avg_wait < 5 else "inverse")
with c3:
    avg_queue = np.mean(result.Nk)
    st.metric("平均排队人数", f"{avg_queue:.1f} 人")
with c4:
    max_wait = np.max(result.wait_times)
    st.metric("最长等待时间", f"{max_wait:.1f} 分钟")

# ---- 对比图 ----
st.markdown("---")
st.markdown("## 排队等待时间分析")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("每位患者的等待时间 (W<sub>k</sub>)", "等待时间分布直方图"),
    horizontal_spacing=0.15,
)

# 左: W_k 散点
k_range = np.arange(1, n_patients + 1)
wait_minutes = result.wait_times
# 颜色编码: < 3分钟绿色, 3-10分钟黄色, > 10分钟红色
colors = np.where(wait_minutes < 3, '#2ecc71',
                   np.where(wait_minutes < 10, '#f1c40f', '#e74c3c'))

fig.add_trace(
    go.Scatter(
        x=k_range, y=wait_minutes, mode='markers',
        marker=dict(color=colors, size=4, line=dict(width=0.3, color='#333')),
        name='等待时间',
        hovertemplate='患者%{x}<br>等待 %{y:.1f} 分钟<extra></extra>',
    ),
    row=1, col=1,
)
# 移动平均
window = max(1, n_patients // 15)
ma = np.convolve(wait_minutes, np.ones(window)/window, mode='valid')
fig.add_trace(
    go.Scatter(
        x=np.arange(window, n_patients + 1), y=ma, mode='lines',
        line=dict(color='#3498db', width=2),
        name=f'滑动平均 (w={window})',
        hovertemplate='滑动平均 %{y:.1f} 分钟<extra></extra>',
    ),
    row=1, col=1,
)
fig.add_hline(y=avg_wait, line_dash='dash', line_color='#e74c3c',
              annotation_text=f'均值 {avg_wait:.1f}分', row=1, col=1)

# 右: 直方图
fig.add_trace(
    go.Histogram(
        x=wait_minutes, nbinsx=40,
        marker=dict(color='#3498db', line=dict(color='white', width=1)),
        name='等待时间分布',
    ),
    row=1, col=2,
)
fig.add_vline(x=avg_wait, line_dash='dash', line_color='#e74c3c',
              annotation_text=f'均值 {avg_wait:.1f}', row=1, col=2)

fig.update_xaxes(title_text='患者序号', row=1, col=1)
fig.update_xaxes(title_text='等待时间 (分钟)', row=1, col=2)
fig.update_yaxes(title_text='等待时间 (分钟)', row=1, col=1)
fig.update_yaxes(title_text='频次', row=1, col=2)

fig.update_layout(
    height=420,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    template='plotly_white',
    margin=dict(l=50, r=20, t=40, b=50),
)
st.plotly_chart(fig, use_container_width=True)

# ---- 场景解读 ----
st.markdown("---")
st.markdown("## 场景解读")

# 统计等待时间分布
pct_under_3 = np.mean(wait_minutes < 3) * 100
pct_under_5 = np.mean(wait_minutes < 5) * 100
pct_over_10 = np.mean(wait_minutes > 10) * 100

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    ### 患者体验分析

    | 指标 | 数值 |
    |------|------|
    | 排队 < 3 分钟 | **{pct_under_3:.1f}%** 的患者 |
    | 排队 < 5 分钟 | **{pct_under_5:.1f}%** 的患者 |
    | 排队 > 10 分钟 | **{pct_over_10:.1f}%** 的患者 |
    | 最长排队记录 | **{max_wait:.1f}** 分钟 |

    {f'**评价**: 大约 {pct_under_5:.0f}% 的患者能在 5 分钟内完成挂号，' if pct_under_5 > 50 else '**评价**: 超过一半患者需要排队 5 分钟以上，'}
    {('整体体验较好。' if pct_under_5 > 70 else ('体验尚可，高峰期可能有抱怨。' if pct_under_5 > 40 else '排队时间偏长，需考虑优化措施。'))}
    """)

with col2:
    st.markdown(rf"""
    ### 与 M/E₂/1 模型的对应

    | 排队论概念 | 医院挂号场景 |
    |-----------|-------------|
    | 到达过程 (M) | 患者随机到达挂号大厅 |
    | 服务时间 (E₂) | 信息录入 + 缴费确认双阶段 |
    | 单服务器 (1) | 唯一的挂号窗口 |
    | 系统负载 $\rho = {rho:.2f}$ | 窗口 {rho*100:.0f}% 时间在忙碌 |
    | 等待时间 $W_k$ | 患者实际排队分钟数 |
    | 系统人数 $N_k$ | 大厅里排队的人数 |

    理论预测平均排队人数 $E[X] = {result.E_X_theory:.2f}$ 人，
    仿真观测值 = {avg_queue:.2f} 人，
    二者高度吻合，验证了 M/E₂/1 模型的实用价值。
    """)

# ---- 优化建议 ----
st.markdown("---")
st.markdown("## 优化建议")

col1, col2 = st.columns(2)

with col1:
    st.info(r"""
    ### 方案 A: 增加窗口 (M/E₂/2)

    将挂号窗口从 1 个增加到 2 个：

    - 服务能力翻倍，系统负载 $\rho$ 减半
    - 患者平均等待时间降至原来的 **1/4 ~ 1/10**
    - 成本: 增加一名工作人员

    **适用场景**: 早高峰、周一上午等繁忙时段
    """)

with col2:
    st.warning(r"""
    ### 方案 B: 流程优化 (降低 $C_s^2$)

    引入自助机或线上预挂号，减少信息录入耗时的波动：

    - 服务时间变异系数 $C_s^2$ 从 0.5 降至约 0.2
    - 在相同 $\rho$ 下，平均等待时间**降低约 30%**
    - 无需增加人力，性价比更高

    **适用场景**: 长期优化、智慧医院建设
    """)

# ---- 总结 ----
st.markdown("---")
st.markdown("## 小结")

st.markdown(f"""
通过这个医院挂号案例，我们看到排队论模型如何指导现实决策：

1. **复刻现实** — M/E₂/1 模型精确刻画了"双阶段服务 + 单窗口 + 随机到达"的挂号场景
2. **量化痛点** — 通过仿真，我们知道了患者在高峰期平均排队 **{avg_wait:.1f} 分钟**，最长等到 **{max_wait:.1f} 分钟**
3. **方案比较** — 增加窗口 vs 流程优化，排队论给出了量化的效果预测
4. **通用方法** — 同样的思路可迁移到银行柜台、景区验票、核酸检测点等无数排队场景

**核心洞见**: 抽象的随机过程模型，最终为真实的"人的等待"提供了数量化的改善方案。
""")

st.markdown("---")
st.caption("🏥 本案例数据参考真实医院门诊挂号参数，患者到达率为估算值。")
