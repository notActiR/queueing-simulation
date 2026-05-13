"""
M/E₂/1 排队系统仿真引擎

M:  泊松到达过程（指数分布到达间隔），到达率 λ
E₂: Erlang-2 服务时间（两个独立 Exp(2μ) 之和），服务率 μ
1:  单服务器，FCFS 调度

四个随机过程:
  {N_k}: 第 k 个任务离开时系统中的任务数（嵌入马尔可夫链）
  {X_t}: 时刻 t 系统中的总任务数
  {W_k}: 第 k 个任务的等待时间
  {Y_t}: 时刻 t 系统中累计服务需求
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimulationResult:
    """单次仿真运行结果"""
    # 系统参数
    arrival_rate: float    # λ
    service_rate: float    # μ
    rho: float             # 负载 ρ = λ/μ
    seed: int
    num_customers: int

    # 每个任务的详细数据 (长度 = num_customers, 0-indexed)
    arrival_times: np.ndarray       # 到达时间
    service_times: np.ndarray       # 服务时间
    start_times: np.ndarray         # 开始服务时间
    departure_times: np.ndarray     # 离开时间
    wait_times: np.ndarray          # 等待时间

    # 过程 {N_k}: k-th departure 时的系统人数 (0-indexed, k=0..n-1)
    Nk: np.ndarray

    # 理论稳态值
    E_X_theory: float = 0.0

    def __post_init__(self):
        self.E_X_theory = self._calc_E_X()

    def _calc_E_X(self) -> float:
        """理论稳态平均系统任务数 E[X]"""
        r = self.rho
        if r >= 1.0:
            return np.inf
        Cs2 = 0.5  # Erlang-2 的变异系数平方
        return r + r**2 * (1 + Cs2) / (2 * (1 - r))

    # ---- 导出时间序列 ----

    def get_Xt(self, t_grid: np.ndarray) -> np.ndarray:
        """
        计算各时间点系统总任务数 {X_t}。
        X_t = 截至 t 的到达数 - 截至 t 的离开数
        """
        arrivals = np.searchsorted(self.arrival_times, t_grid, side='right')
        departures = np.searchsorted(self.departure_times, t_grid, side='right')
        return arrivals - departures

    def get_Yt(self, t_grid: np.ndarray) -> dict:
        """
        计算累计服务需求，返回字典:
          'total':     累计进入系统的总服务需求 (monotonic step)
          'completed': 已完成的服务量
          'remaining': 剩余服务需求 = total - completed
        """
        total = np.zeros_like(t_grid)
        completed = np.zeros_like(t_grid)
        busy_until = 0.0
        service_done = 0.0  # how much service has been done up to current time
        arrival_idx = 0
        n = self.num_customers

        for i, t in enumerate(t_grid):
            # 累计到达的服务需求
            while arrival_idx < n and self.arrival_times[arrival_idx] <= t:
                arrival_idx += 1
            total[i] = np.sum(self.service_times[:arrival_idx])

            # 累计已完成的服务
            # 服务器在 [start_time[j], departure_time[j]] 之间以速率 1 工作
            done = 0.0
            for j in range(arrival_idx):
                # j 已经到达
                if self.departure_times[j] <= t:
                    # j 已完成，贡献全部服务时间
                    done += self.service_times[j]
                elif self.start_times[j] <= t:
                    # j 正在服务中
                    done += t - self.start_times[j]
                # else: j 在等待，尚未贡献
            completed[i] = done

        remaining = total - completed
        return {'total': total, 'completed': completed, 'remaining': remaining}

    def get_event_events(self, t_max: float) -> dict:
        """返回 (0, t_max) 范围内的到达/离开事件列表"""
        arrivals = self.arrival_times[self.arrival_times <= t_max]
        departures = self.departure_times[self.departure_times <= t_max]
        # 到达事件: X 增加 1
        # 离开事件: X 减少 1
        return {
            'arrivals': arrivals,
            'departures': departures,
        }

    def get_first_empty_k(self) -> int:
        """返回系统首次变空的离开序号 k0 (1-indexed)"""
        mask = self.Nk == 0
        if np.any(mask):
            return int(np.argmax(mask)) + 1  # +1 转为 1-indexed
        return -1  # 从未变空


class ME2Queue:
    """M/E₂/1 排队系统仿真器"""

    def __init__(self, arrival_rate: float, service_rate: float, seed: int = 42):
        """
        arrival_rate: λ, 到达率
        service_rate: μ, 服务率
        seed: 随机种子
        """
        if arrival_rate <= 0 or service_rate <= 0:
            raise ValueError("到达率和服务率必须为正")
        self.lam = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self, num_customers: int) -> SimulationResult:
        """运行仿真 num_customers 个任务，返回 SimulationResult"""
        n = num_customers
        rng = self.rng

        # ---- 1. 生成到达时间 (泊松过程: 间隔 ~ Exp(λ)) ----
        inter_arrivals = rng.exponential(scale=1.0 / self.lam, size=n)
        arrival_times = np.cumsum(inter_arrivals)

        # ---- 2. 生成服务时间 (Erlang-2: Exp(2μ) + Exp(2μ)) ----
        # shape=2 的 Gamma 分布, scale=1/(2μ)
        service_times = rng.gamma(shape=2, scale=1.0 / (2 * self.mu), size=n)

        # ---- 3. FCFS 调度 (Lindley 递推) ----
        start_times = np.zeros(n)
        departure_times = np.zeros(n)

        for i in range(n):
            if i == 0:
                start_times[i] = arrival_times[i]
            else:
                start_times[i] = max(arrival_times[i], departure_times[i - 1])
            departure_times[i] = start_times[i] + service_times[i]

        wait_times = start_times - arrival_times

        # ---- 4. 计算 {N_k} (离开时刻系统内人数) ----
        Nk = np.zeros(n, dtype=int)
        for k in range(n):
            dep_time = departure_times[k]
            # 在 dep_time 之前到达的任务数
            n_arrived = int(np.searchsorted(arrival_times, dep_time, side='right'))
            # N_k = 已到达数 - 已离开数(=k+1)
            n_departed = k + 1
            Nk[k] = n_arrived - n_departed

        result = SimulationResult(
            arrival_rate=self.lam,
            service_rate=self.mu,
            rho=self.rho,
            seed=self.seed,
            num_customers=n,
            arrival_times=arrival_times,
            service_times=service_times,
            start_times=start_times,
            departure_times=departure_times,
            wait_times=wait_times,
            Nk=Nk,
        )
        # E_X_theory 已在 __post_init__ 中计算
        return result
