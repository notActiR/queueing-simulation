"""
收敛性分析可视化图表

包括:
  1. 均值-方差收敛图 (含 95% CI 和理论参考线)
  2. 分布演变图 (小提琴图/箱线图)
  3. 自相关分析图
  4. 收敛路径图 (所有样本的逐 k 路径)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulation.statistics import MultiRunStats, compute_acf, DISTRIBUTION_K_VALUES, STUDY_K_VALUES
from typing import Optional


def build_convergence_plot(stats: MultiRunStats) -> go.Figure:
    """
    均值-方差收敛图: E[N_k] 和 Var[N_k] 随 k 的变化
    含 95% CI 误差棒和理论稳态参考线
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('E[N<sub>k</sub>] 收敛曲线', 'Var[N<sub>k</sub>] 收敛曲线'),
        horizontal_spacing=0.12,
    )

    k_vals = np.array(stats.k_values)  # [1,5,10,50,100,200,500,1000]

    # 左: 均值
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=stats.means, mode='lines+markers',
            marker=dict(size=8, color='#3498db'),
            line=dict(color='#3498db', width=2),
            name='E[N<sub>k</sub>] (仿真)',
            hovertemplate='k=%{x}<br>E[N<sub>k</sub>]=%{y:.3f}<extra></extra>',
        ),
        row=1, col=1,
    )
    # 95% CI 误差棒
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=stats.ci_upper, mode='lines',
            line=dict(color='#3498db', width=0.5, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=stats.ci_lower, mode='lines',
            line=dict(color='#3498db', width=0.5, dash='dot'),
            fill='tonexty', fillcolor='rgba(52,152,219,0.15)',
            name='95% CI',
            hovertemplate='k=%{x}<br>CI: [%{y:.3f}]<extra></extra>',
        ),
        row=1, col=1,
    )
    # 理论参考线
    if np.isfinite(stats.E_X_theory):
        fig.add_hline(
            y=stats.E_X_theory, line_dash='dash', line_color='#e74c3c',
            annotation_text=f'理论 E[X]={stats.E_X_theory:.3f}',
            row=1, col=1,
        )

    # 右: 方差
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=stats.variances, mode='lines+markers',
            marker=dict(size=8, color='#e67e22'),
            line=dict(color='#e67e22', width=2),
            name='Var[N<sub>k</sub>] (仿真)',
            hovertemplate='k=%{x}<br>Var[N<sub>k</sub>]=%{y:.3f}<extra></extra>',
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text='k (离开序号)', type='log', row=1, col=1)
    fig.update_xaxes(title_text='k (离开序号)', type='log', row=1, col=2)
    fig.update_yaxes(title_text='E[N<sub>k</sub>]', row=1, col=1)
    fig.update_yaxes(title_text='Var[N<sub>k</sub>]', row=1, col=2)

    fig.update_layout(
        height=450,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def build_distribution_evolution(stats: MultiRunStats) -> go.Figure:
    """
    分布演变图: 对选定 k 值绘制小提琴图/箱线图
    使用 Plotly 的 violin 或 box 展示分布形态变化
    """
    dist_k_vals = [k for k in DISTRIBUTION_K_VALUES if k <= stats.n_customers]

    fig = make_subplots(
        rows=1, cols=len(dist_k_vals),
        subplot_titles=[f'k={k}' for k in dist_k_vals],
    )

    for i, k in enumerate(dist_k_vals):
        samples = stats.per_k_samples[k]
        # 小提琴图 (用 box 替代因为 Plotly 的 violin 在 subplot 中有时有问题)
        fig.add_trace(
            go.Box(
                y=samples,
                name=f'k={k}',
                marker=dict(color='#3498db'),
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate='N<sub>k</sub>=%{y}<extra></extra>',
            ),
            row=1, col=i + 1,
        )
        # 添加参考的理论线
        if np.isfinite(stats.E_X_theory):
            fig.add_hline(
                y=stats.E_X_theory, line_dash='dash', line_color='#e74c3c',
                row=1, col=i + 1,
            )

    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white',
        title='{N<sub>k</sub>} 分布演变 (从瞬态到稳态)',
        margin=dict(l=40, r=20, t=60, b=40),
    )
    # 统一 y 轴范围方便对比
    all_samples = np.concatenate([stats.per_k_samples[k] for k in dist_k_vals])
    y_max = np.percentile(all_samples, 98) * 1.1
    for i in range(len(dist_k_vals)):
        fig.update_yaxes(range=[-0.5, y_max], row=1, col=i + 1)

    return fig


def build_acf_plot(stats: MultiRunStats, max_lag: int = 30) -> go.Figure:
    """
    自相关分析图: k=1000 时 {N_k} 的样本自相关函数
    与理论指数衰减对比
    """
    # 用第一次仿真的 N_k 序列计算 ACF
    # 但 N_k 是每个 departure 的值，我们取前 N 个
    if not stats.all_results:
        return go.Figure()

    result = stats.all_results[0]
    Nk_series = result.Nk  # 全部 N_k 序列
    acf = compute_acf(Nk_series, max_lag)

    fig = go.Figure()

    # ACF 柱状图
    lags = np.arange(len(acf))
    fig.add_trace(go.Bar(
        x=lags, y=acf,
        marker=dict(color='#3498db', line=dict(width=0)),
        name='样本 ACF',
    ))

    # 理论指数衰减参考 (基于 ρ 估算)
    rho = stats.rho
    # 对于 M/G/1, N_k 的自相关大致按 ρ^lag 衰减
    exp_decay = rho ** lags
    fig.add_trace(go.Scatter(
        x=lags, y=exp_decay, mode='lines',
        line=dict(color='#e74c3c', dash='dash', width=2),
        name=f'ρ<sup>lag</sup> ({rho:.2f}<sup>lag</sup>)',
    ))

    # 95% 置信边界
    n = len(Nk_series)
    conf_bound = 1.96 / np.sqrt(n)
    fig.add_hline(y=conf_bound, line_dash='dot', line_color='gray',
                   annotation_text='95% 上界')
    fig.add_hline(y=-conf_bound, line_dash='dot', line_color='gray',
                   annotation_text='95% 下界')

    fig.update_layout(
        height=400,
        title=f'N<sub>k</sub> 自相关函数 (N={len(Nk_series)}), ρ={rho:.2f}',
        xaxis_title='滞后 (lag)',
        yaxis_title='自相关',
        template='plotly_white',
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def build_convergence_paths(stats: MultiRunStats) -> go.Figure:
    """
    收敛路径图: 30 条灰色样本均值的累积路径
    """
    n_runs = min(stats.n_runs, len(stats.all_results))
    k_vals = np.array(stats.k_values)

    fig = go.Figure()

    # 每条路径 (半透明灰色)
    for run_idx in range(n_runs):
        values = [stats.per_k_samples[k][run_idx] for k in k_vals]
        fig.add_trace(go.Scatter(
            x=k_vals, y=values, mode='lines+markers',
            line=dict(color='gray', width=0.5),
            marker=dict(size=2),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip',
        ))

    # 均值线 (粗蓝线)
    fig.add_trace(go.Scatter(
        x=k_vals, y=stats.means, mode='lines+markers',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8, color='#3498db'),
        name='E[N<sub>k</sub>] (均值)',
        hovertemplate='k=%{x}<br>E[N<sub>k</sub>]=%{y:.3f}<extra></extra>',
    ))

    # 理论参考线
    if np.isfinite(stats.E_X_theory):
        fig.add_hline(
            y=stats.E_X_theory, line_dash='dash', line_color='#e74c3c',
            annotation_text=f'理论 E[X]={stats.E_X_theory:.3f}',
        )

    fig.update_xaxes(title_text='k (离开序号)', type='log')
    fig.update_yaxes(title_text='N<sub>k</sub>')

    fig.update_layout(
        height=450,
        title=f'{n_runs} 条样本路径的收敛过程',
        template='plotly_white',
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def build_rho_comparison(
    result_low: 'SimulationResult',
    result_high: 'SimulationResult',
    max_customers: int = 300,
) -> go.Figure:
    """
    ρ 对比图: 低负载 vs 高负载下的 W_k 和 N_k 对比
    用于第 4 章"深度探索"
    """
    n_low = min(result_low.num_customers, max_customers)
    n_high = min(result_high.num_customers, max_customers)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'N<sub>k</sub> (ρ={result_low.rho:.2f})',
            f'N<sub>k</sub> (ρ={result_high.rho:.2f})',
            f'W<sub>k</sub> (ρ={result_low.rho:.2f})',
            f'W<sub>k</sub> (ρ={result_high.rho:.2f})',
        ),
        vertical_spacing=0.15,
    )

    # 左上: 低 ρ 的 N_k
    from visualization.four_processes import color_by_load
    colors_low = color_by_load(result_low.Nk[:n_low])
    fig.add_trace(
        go.Scatter(x=np.arange(1, n_low + 1), y=result_low.Nk[:n_low],
                   mode='markers', marker=dict(color=colors_low, size=3),
                   showlegend=False, hoverinfo='skip'),
        row=1, col=1,
    )
    if np.isfinite(result_low.E_X_theory):
        fig.add_hline(y=result_low.E_X_theory, line_dash='dash', line_color='#e74c3c',
                       row=1, col=1, annotation_text=f'E[X]={result_low.E_X_theory:.2f}')

    # 右上: 高 ρ 的 N_k
    colors_high = color_by_load(result_high.Nk[:n_high])
    fig.add_trace(
        go.Scatter(x=np.arange(1, n_high + 1), y=result_high.Nk[:n_high],
                   mode='markers', marker=dict(color=colors_high, size=3),
                   showlegend=False, hoverinfo='skip'),
        row=1, col=2,
    )
    if np.isfinite(result_high.E_X_theory):
        fig.add_hline(y=result_high.E_X_theory, line_dash='dash', line_color='#e74c3c',
                       row=1, col=2, annotation_text=f'E[X]={result_high.E_X_theory:.2f}')

    # 左下: 低 ρ 的 W_k
    fig.add_trace(
        go.Scatter(x=np.arange(1, n_low + 1), y=result_low.wait_times[:n_low],
                   mode='markers', marker=dict(color='#9b59b6', size=3, opacity=0.5),
                   showlegend=False, hoverinfo='skip'),
        row=2, col=1,
    )
    # 右下: 高 ρ 的 W_k
    fig.add_trace(
        go.Scatter(x=np.arange(1, n_high + 1), y=result_high.wait_times[:n_high],
                   mode='markers', marker=dict(color='#9b59b6', size=3, opacity=0.5),
                   showlegend=False, hoverinfo='skip'),
        row=2, col=2,
    )

    fig.update_xaxes(title_text='离开序号 k', row=2, col=1)
    fig.update_xaxes(title_text='离开序号 k', row=2, col=2)
    fig.update_yaxes(title_text='N<sub>k</sub>', row=1, col=1)
    fig.update_yaxes(title_text='N<sub>k</sub>', row=1, col=2)
    fig.update_yaxes(title_text='W<sub>k</sub>', row=2, col=1)
    fig.update_yaxes(title_text='W<sub>k</sub>', row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig
