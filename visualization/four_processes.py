"""
四过程联动可视化面板

{N_k}, {X_t}, {W_k}, {Y_t} 的 2×2 联动展示
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulation.engine import SimulationResult


def color_by_load(Nk: np.ndarray) -> np.ndarray:
    """
    根据系统负载状态分配颜色
    绿色: Nk = 0 (空闲)
    黄色: 1 <= Nk <= 3 (正常负载)
    红色: Nk > 3 (高负载)
    """
    colors = np.empty(len(Nk), dtype=object)
    colors[Nk == 0] = '#2ecc71'         # 绿色
    colors[(Nk >= 1) & (Nk <= 3)] = '#f1c40f'  # 黄色
    colors[Nk > 3] = '#e74c3c'           # 红色
    return colors


def build_four_process_figure(result: SimulationResult, max_customers: int = 200) -> go.Figure:
    """
    构建四过程联动 2×2 可视化面板

    参数:
      result: 仿真结果
      max_customers: 显示的最大任务数 (太多会看不清)
    """
    n = min(result.num_customers, max_customers)
    t_max = result.departure_times[n - 1]
    t_grid = np.linspace(0, t_max, max(2, n * 4))

    # 准备各过程数据
    k_range = np.arange(1, n + 1)
    Nk_vals = result.Nk[:n]
    colors = color_by_load(Nk_vals)

    Xt_vals = result.get_Xt(t_grid)
    Yt_data = result.get_Yt(t_grid)
    events = result.get_event_events(t_max)

    # 首次空闲时刻
    k0 = result.get_first_empty_k()

    # 移动平均 (窗口大小自适应)
    window = max(1, n // 20)
    wait_ma = np.convolve(result.wait_times[:n], np.ones(window) / window, mode='valid')

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '{N<sub>k</sub>}: 离开时的系统任务数',
            '{X<sub>t</sub>}: 系统任务数随时间变化',
            '{W<sub>k</sub>}: 任务等待时间',
            '{Y<sub>t</sub>}: 累计服务需求'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ---- 左上: {N_k} ----
    fig.add_trace(
        go.Scatter(
            x=k_range, y=Nk_vals, mode='markers',
            marker=dict(color=colors, size=5, line=dict(width=0.5, color='#333')),
            name='N<sub>k</sub>',
            hovertemplate='k=%{x}<br>N<sub>k</sub>=%{y}<extra></extra>',
        ),
        row=1, col=1,
    )
    # 标注首次空闲
    if 1 <= k0 <= n:
        fig.add_vline(x=k0, line_dash='dash', line_color='#2ecc71',
                       annotation_text=f'k₀={k0}', row=1, col=1)

    # ---- 右上: {X_t} ----
    fig.add_trace(
        go.Scatter(
            x=t_grid, y=Xt_vals, mode='lines',
            line=dict(shape='hv', color='#3498db', width=1.5),
            name='X<sub>t</sub>',
            hovertemplate='t=%{x:.2f}<br>X<sub>t</sub>=%{y}<extra></extra>',
        ),
        row=1, col=2,
    )
    # 理论参考线
    if np.isfinite(result.E_X_theory):
        fig.add_hline(y=result.E_X_theory, line_dash='dash', line_color='#e74c3c',
                       annotation_text=f'E[X]={result.E_X_theory:.2f}', row=1, col=2)

    # 在 {X_t} 上标注到达和离开事件 (用可见性切换减少拥挤)
    if n <= 100:
        for i in range(n):
            # 到达: 向上跳
            fig.add_trace(
                go.Scatter(
                    x=[result.arrival_times[i], result.arrival_times[i]],
                    y=[max(0, Xt_vals[np.searchsorted(t_grid, result.arrival_times[i]) - 1]),
                       Xt_vals[np.searchsorted(t_grid, result.arrival_times[i], side='right') - 1]],
                    mode='lines', line=dict(color='#2ecc71', width=0.5),
                    showlegend=False, hoverinfo='skip',
                ),
                row=1, col=2,
            )
            # 离开: 向下跳
            fig.add_trace(
                go.Scatter(
                    x=[result.departure_times[i], result.departure_times[i]],
                    y=[Xt_vals[np.searchsorted(t_grid, result.departure_times[i], side='right') - 1],
                       max(0, Xt_vals[np.searchsorted(t_grid, result.departure_times[i]) - 1])],
                    mode='lines', line=dict(color='#e74c3c', width=0.5),
                    showlegend=False, hoverinfo='skip',
                ),
                row=1, col=2,
            )

    # ---- 左下: {W_k} ----
    fig.add_trace(
        go.Scatter(
            x=k_range, y=result.wait_times[:n], mode='markers',
            marker=dict(color='#9b59b6', size=3, opacity=0.6),
            name='W<sub>k</sub>',
            hovertemplate='k=%{x}<br>W<sub>k</sub>=%{y:.3f}<extra></extra>',
        ),
        row=2, col=1,
    )
    # 移动平均线
    ma_k = np.arange(window, n + 1)
    fig.add_trace(
        go.Scatter(
            x=ma_k, y=wait_ma, mode='lines',
            line=dict(color='#e67e22', width=2),
            name=f'移动平均 (w={window})',
            hovertemplate='k=%{x}<br>MA=%{y:.3f}<extra></extra>',
        ),
        row=2, col=1,
    )

    # ---- 右下: {Y_t} ----
    fig.add_trace(
        go.Scatter(
            x=t_grid, y=Yt_data['completed'], mode='none',
            fill='tozeroy', fillgradient=dict(type='vertical', colorscale=[
                [0, 'rgba(46,204,113,0.3)'], [1, 'rgba(46,204,113,0.8)']
            ]),
            name='已完成服务',
            hovertemplate='t=%{x:.2f}<br>已完成=%{y:.2f}<extra></extra>',
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid, y=Yt_data['total'], mode='none',
            fill='tonexty',
            fillgradient=dict(type='vertical', colorscale=[
                [0, 'rgba(231,76,60,0.3)'], [1, 'rgba(231,76,60,0.8)']
            ]),
            name='剩余服务需求',
            hovertemplate='t=%{x:.2f}<br>累计总需求=%{y:.2f}<extra></extra>',
        ),
        row=2, col=2,
    )

    # 布局
    fig.update_xaxes(title_text='离开序号 k', row=1, col=1)
    fig.update_xaxes(title_text='时间 t', row=1, col=2)
    fig.update_xaxes(title_text='任务序号 k', row=2, col=1)
    fig.update_xaxes(title_text='时间 t', row=2, col=2)

    fig.update_yaxes(title_text='N<sub>k</sub>', row=1, col=1)
    fig.update_yaxes(title_text='X<sub>t</sub>', row=1, col=2)
    fig.update_yaxes(title_text='W<sub>k</sub>', row=2, col=1)
    fig.update_yaxes(title_text='服务需求', row=2, col=2)

    fig.update_layout(
        height=750,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified',
        margin=dict(l=50, r=20, t=90, b=50),
        template='plotly_white',
    )

    return fig


def build_wait_histogram(result: SimulationResult, max_customers: int = 500) -> go.Figure:
    """等待时间分布直方图"""
    wait_times = result.wait_times[:max_customers]
    wait_times = wait_times[wait_times > 0]  # 排除零等待

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=wait_times, nbinsx=50,
        marker=dict(color='#9b59b6', line=dict(color='white', width=1)),
        name='等待时间分布',
    ))
    # 均值线
    if len(wait_times) > 0:
        mean_w = np.mean(wait_times)
        fig.add_vline(x=mean_w, line_dash='dash', line_color='#e74c3c',
                       annotation_text=f'均值={mean_w:.2f}')
    fig.update_layout(
        title='等待时间分布',
        xaxis_title='等待时间',
        yaxis_title='频次',
        template='plotly_white',
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
