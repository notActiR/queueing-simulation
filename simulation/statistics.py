"""
多轮仿真统计分析模块
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from simulation.engine import ME2Queue, SimulationResult
from typing import List, Optional


# 需要重点研究的 k 值
STUDY_K_VALUES = [1, 5, 10, 50, 100, 200, 500, 1000]
# 分布演变观察点
DISTRIBUTION_K_VALUES = [1, 10, 50, 100, 500, 1000]


@dataclass
class MultiRunStats:
    """多轮仿真统计结果"""
    arrival_rate: float
    service_rate: float
    rho: float
    n_runs: int
    n_customers: int
    k_values: List[int]
    seeds: List[int]

    # Nk_samples[run_idx][k_value] -> 该轮该 k 的 Nk 值
    # 但我们按 k 组织: per_k_samples[k] -> n_runs 个样本
    per_k_samples: dict = field(default_factory=dict)

    # 每个 k 的统计量
    means: np.ndarray = field(default_factory=lambda: np.array([]))
    variances: np.ndarray = field(default_factory=lambda: np.array([]))
    std_errs: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_upper: np.ndarray = field(default_factory=lambda: np.array([]))

    # 理论稳态值
    E_X_theory: float = 0.0
    Var_X_theory: Optional[float] = None

    # 所有轮次的仿真结果 (第一轮用于序列分析)
    all_results: List[SimulationResult] = field(default_factory=list)

    def compute(self):
        """计算所有统计量"""
        self.means = np.zeros(len(self.k_values))
        self.variances = np.zeros(len(self.k_values))
        self.std_errs = np.zeros(len(self.k_values))
        self.ci_lower = np.zeros(len(self.k_values))
        self.ci_upper = np.zeros(len(self.k_values))

        for i, k in enumerate(self.k_values):
            samples = self.per_k_samples[k]
            n = len(samples)
            mean = np.mean(samples)
            var = np.var(samples, ddof=1)  # 样本方差
            std_err = np.sqrt(var / n)

            self.means[i] = mean
            self.variances[i] = var
            self.std_errs[i] = std_err

            # 95% 置信区间 (t 分布, df = n-1)
            t_val = stats.t.ppf(0.975, df=n - 1)
            self.ci_lower[i] = mean - t_val * std_err
            self.ci_upper[i] = mean + t_val * std_err

        # 理论稳态值
        if self.all_results:
            self.E_X_theory = self.all_results[0].E_X_theory


def run_multiple_simulations(
    arrival_rate: float = 0.7,
    service_rate: float = 1.0,
    n_runs: int = 30,
    n_customers: int = 1000,
    base_seed: int = 42,
) -> MultiRunStats:
    """
    运行多轮独立仿真并计算统计量

    参数:
      arrival_rate: λ, 默认 0.7 (ρ = 0.7)
      service_rate: μ, 默认 1.0
      n_runs: 独立运行次数, 默认 30
      n_customers: 每次仿真的任务数, 默认 1000
      base_seed: 基础种子
    """
    k_values = STUDY_K_VALUES
    rho = arrival_rate / service_rate

    # 初始化样本存储
    per_k_samples = {k: np.zeros(n_runs) for k in k_values}
    seeds = [base_seed + i * 100 for i in range(n_runs)]
    all_results = []

    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        sim = ME2Queue(arrival_rate, service_rate, seed=seed)
        result = sim.run(n_customers)
        all_results.append(result)

        for k in k_values:
            if k <= n_customers:
                # Nk 是 0-indexed, k=1 对应 index=0
                per_k_samples[k][run_idx] = result.Nk[k - 1]
            else:
                per_k_samples[k][run_idx] = np.nan

    stats_obj = MultiRunStats(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        rho=rho,
        n_runs=n_runs,
        n_customers=n_customers,
        k_values=list(k_values),
        seeds=seeds,
        per_k_samples=per_k_samples,
        all_results=all_results,
    )
    stats_obj.compute()
    return stats_obj


def compute_acf(data: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """计算样本自相关函数"""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            cov = np.mean((data[lag:] - mean) * (data[:n - lag] - mean))
            acf[lag] = cov / var
    return acf
