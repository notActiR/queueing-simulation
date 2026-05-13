"""
导出独立 HTML 交互页面

生成可脱离 Python 环境在浏览器中打开的完整交互页面。
"""

import os
import numpy as np
from string import Template
from simulation.engine import ME2Queue
from simulation.statistics import run_multiple_simulations
from visualization.four_processes import build_four_process_figure
from visualization.analysis_charts import (
    build_convergence_plot,
    build_distribution_evolution,
    build_acf_plot,
)

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>M/E2/1 排队系统仿真与可视化分析</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
    background: #f5f7fa;
    color: #2c3e50;
    line-height: 1.6;
  }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  .hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 60px 40px; border-radius: 16px;
    margin-bottom: 30px; text-align: center;
  }
  .hero h1 { font-size: 2.5em; margin-bottom: 10px; }
  .hero p { font-size: 1.2em; opacity: 0.9; }
  .section {
    background: white; border-radius: 12px; padding: 30px;
    margin-bottom: 30px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  }
  .section h2 {
    font-size: 1.6em; margin-bottom: 15px;
    border-left: 4px solid #667eea; padding-left: 15px;
  }
  .metrics {
    display: flex; gap: 20px; flex-wrap: wrap;
    margin: 20px 0;
  }
  .metric-card {
    background: #f0f4ff; border-radius: 10px; padding: 20px;
    flex: 1; min-width: 150px; text-align: center;
  }
  .metric-card .value { font-size: 2em; font-weight: bold; color: #667eea; }
  .metric-card .label { font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }
  .chart-container { margin: 20px 0; }
  .footer {
    text-align: center; padding: 30px; color: #95a5a6;
    font-size: 0.9em;
  }
  .insight-box {
    background: #fff3cd; border-left: 4px solid #f39c12;
    padding: 15px 20px; border-radius: 6px; margin: 15px 0;
  }
</style>
</head>
<body>
<div class="container">

  <div class="hero">
    <h1>M/E2/1 排队系统的仿真与可视化分析</h1>
    <p>《概率论与随机过程》期中小组探究项目</p>
  </div>

  <div class="section">
    <h2>系统参数</h2>
    <div class="metrics">
      <div class="metric-card">
        <div class="value">$LAM</div>
        <div class="label">到达率</div>
      </div>
      <div class="metric-card">
        <div class="value">$MU</div>
        <div class="label">服务率</div>
      </div>
      <div class="metric-card">
        <div class="value">$RHO</div>
        <div class="label">系统负载</div>
      </div>
      <div class="metric-card">
        <div class="value">E[X] = $EX</div>
        <div class="label">理论稳态队长</div>
      </div>
      <div class="metric-card">
        <div class="value">$AVGW</div>
        <div class="label">仿真平均等待时间</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>四个随机过程的联动可视化</h2>
    <p style="color: #7f8c8d; margin-bottom: 10px;">
      鼠标悬停可在各图之间联动高亮。绿色=空闲, 黄色=正常(1-3), 红色=繁忙(>3)
    </p>
    <div class="chart-container">$CHART4</div>
    <div class="insight-box">
      <strong>观察要点:</strong> N_k 的颜色变化反映了系统负载的波动; X_t 在 E[X]=$EX 参考线附近波动; W_k 初期可能偏高(瞬态), 后期趋于稳定。
    </div>
  </div>

  <div class="section">
    <h2>均值-方差收敛图 (30 次独立仿真)</h2>
    <div class="chart-container">$CHARTCONV</div>
  </div>

  <div class="section">
    <h2>分布演变图</h2>
    <div class="chart-container">$CHARTDIST</div>
  </div>

  <div class="section">
    <h2>自相关分析</h2>
    <div class="chart-container">$CHARTACF</div>
  </div>

  <div class="footer">
    <p>生成参数: $PARAMS</p>
    <p>M/E2/1 Queueing System Simulation (c) 2026</p>
  </div>

</div>
</body>
</html>""")


def export_dashboard_html(output_path: str, lam=0.7, mu=1.0, n_customers=300, seed=42):
    """导出自包含的交互仪表板 HTML 文件"""
    # 运行仿真
    sim = ME2Queue(lam, mu, seed=seed)
    result = sim.run(n_customers)

    # 构建四过程图
    fig4 = build_four_process_figure(result, max_customers=n_customers)

    # 运行统计分析
    stats = run_multiple_simulations(lam, mu, n_runs=30, n_customers=min(1000, n_customers + 200))
    fig_conv = build_convergence_plot(stats)
    fig_dist = build_distribution_evolution(stats)
    fig_acf = build_acf_plot(stats)

    rho = lam / mu
    E_X = result.E_X_theory

    full_html = HTML_TEMPLATE.substitute(
        LAM=f"lambda = {lam}",
        MU=f"mu = {mu}",
        RHO=f"rho = {rho:.3f}",
        EX=f"{E_X:.3f}",
        AVGW=f"{np.mean(result.wait_times):.3f}",
        CHART4=fig4.to_html(full_html=False, include_plotlyjs='cdn',
                            config={'responsive': True, 'displayModeBar': True}),
        CHARTCONV=fig_conv.to_html(full_html=False, include_plotlyjs=False,
                                   config={'responsive': True, 'displayModeBar': True}),
        CHARTDIST=fig_dist.to_html(full_html=False, include_plotlyjs=False,
                                   config={'responsive': True, 'displayModeBar': True}),
        CHARTACF=fig_acf.to_html(full_html=False, include_plotlyjs=False,
                                 config={'responsive': True, 'displayModeBar': True}),
        PARAMS=f"lambda={lam}, mu={mu}, rho={rho:.3f}, n_customers={n_customers}",
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"独立 HTML 页面已导出到: {output_path}")
    return output_path


def export_offline_html(output_path: str, lam=0.7, mu=1.0, n_customers=300, seed=42):
    """导出完全离线的 HTML 文件 (Plotly.js 内嵌, 约 3.5MB)"""
    sim = ME2Queue(lam, mu, seed=seed)
    result = sim.run(n_customers)

    fig4 = build_four_process_figure(result, max_customers=n_customers)
    stats = run_multiple_simulations(lam, mu, n_runs=30, n_customers=min(1000, n_customers + 200))
    fig_conv = build_convergence_plot(stats)
    fig_dist = build_distribution_evolution(stats)
    fig_acf = build_acf_plot(stats)

    rho = lam / mu
    E_X = result.E_X_theory

    # 第一个图 include_plotlyjs=True 内嵌完整 Plotly.js (~3.5MB), 后续用 False
    prefix = fig4.to_html(full_html=False, include_plotlyjs=True,
                          config={'responsive': True, 'displayModeBar': True})
    conv_html = fig_conv.to_html(full_html=False, include_plotlyjs=False,
                                 config={'responsive': True, 'displayModeBar': True})
    dist_html = fig_dist.to_html(full_html=False, include_plotlyjs=False,
                                 config={'responsive': True, 'displayModeBar': True})
    acf_html = fig_acf.to_html(full_html=False, include_plotlyjs=False,
                               config={'responsive': True, 'displayModeBar': True})

    full_html = HTML_TEMPLATE.substitute(
        LAM=f"lambda = {lam}",
        MU=f"mu = {mu}",
        RHO=f"rho = {rho:.3f}",
        EX=f"{E_X:.3f}",
        AVGW=f"{np.mean(result.wait_times):.3f}",
        CHART4=prefix,
        CHARTCONV=conv_html,
        CHARTDIST=dist_html,
        CHARTACF=acf_html,
        PARAMS=f"lambda={lam}, mu={mu}, rho={rho:.3f}, n_customers={n_customers}",
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"离线 HTML 页面已导出到: {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    export_dashboard_html(os.path.join(output_dir, 'dashboard.html'))
    export_offline_html(os.path.join(output_dir, 'dashboard_offline.html'))
