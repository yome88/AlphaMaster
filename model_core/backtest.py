"""model_core/backtest.py — MT5 回测评估器（组合级多目标 Reward）

A股适配：
  - 新增 A_SHARE_PERIODS_PER_YEAR = 252（A 股日线每年约 252 个交易日）
  - evaluate_fold 支持 symbol 参数自动判断市场类型和成本率
  - evaluate 支持 symbol 参数
  - _multi_objective 中 N=1 单品种模式新增 a_share 奖励模式（仅做多）

评分框架（5品种组合版）：
  final_score =
      0.35 * portfolio_sortino
    + 0.20 * portfolio_calmar
    + 0.15 * ts_ic_stability
    + 0.10 * symbol_consistency
    + 0.10 * cost_stress
    + 0.10 * turnover_quality
    - complexity_penalty
    - correlation_penalty
"""
import math
import torch
from torch import Tensor

from strategy_manager.signal import compute_target_positions_stateless
from .config import ModelConfig

_H1_PERIODS_PER_YEAR     = 6240
_A_SHARE_PERIODS_PER_YEAR = 252  # A 股日线：约 252 个交易日/年
_SORTINO_CLIP            = 20.0


class MT5Backtest:
    """MT5 组合级回测评估器。"""

    def __init__(
        self,
        cost_rate:        float = 0.0001,
        periods_per_year: int   = _H1_PERIODS_PER_YEAR,
        symbol:           str | None = None,
    ):
        self.cost_rate        = cost_rate
        self.periods_per_year = periods_per_year
        self.symbol           = symbol

    @staticmethod
    def _detect_periods_per_year(symbol: str | None, timeframe: str | None = None) -> int:
        """根据品种/周期自动推断每年 bar 数。"""
        if symbol is None:
            return _H1_PERIODS_PER_YEAR
        try:
            from config import Config, _is_a_share_symbol
            if _is_a_share_symbol(symbol):
                # A 股默认日线
                if timeframe is None or timeframe in ("D1", "d1", "1d", "daily"):
                    return _A_SHARE_PERIODS_PER_YEAR
                if timeframe in ("H1", "h1", "1h"):
                    # A股每天约 4h 交易（09:30-11:30, 13:00-15:00）≈ 240 分钟 = 4h
                    return _A_SHARE_PERIODS_PER_YEAR * 4
                return _A_SHARE_PERIODS_PER_YEAR
        except Exception:
            pass
        return _H1_PERIODS_PER_YEAR

    @staticmethod
    def _detect_cost_rate(symbol: str | None) -> float:
        """根据品种自动推断成本率。"""
        if symbol is None:
            return 0.0001
        try:
            from config import Config, _is_a_share_symbol
            if _is_a_share_symbol(symbol):
                return Config.COST_RATE_A_SHARE
        except Exception:
            pass
        return 0.0001

    # ──────────────────────────────────────────────────────────────────────
    # 基础统计
    # ──────────────────────────────────────────────────────────────────────

    def _sortino(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        flat     = pnl.reshape(-1)
        mean_pnl = flat.mean()
        downside = flat[flat < 0]
        raw_std  = downside.std(unbiased=False) if downside.numel() > 0 \
                   else torch.tensor(0.0, dtype=flat.dtype, device=flat.device)
        full_std       = flat.std(unbiased=False).clamp(min=eps)
        floor          = torch.clamp(full_std * 0.2, min=eps)
        downside_std   = torch.clamp(raw_std, min=floor)
        sortino        = mean_pnl / downside_std * math.sqrt(self.periods_per_year)
        return torch.clamp(sortino, -_SORTINO_CLIP, _SORTINO_CLIP)

    def _calmar(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        flat      = pnl.reshape(-1)
        ann_ret   = flat.mean() * self.periods_per_year
        cum       = torch.cumsum(flat, dim=0)
        peak      = torch.cummax(cum, dim=0).values
        drawdown  = (peak - cum).max()
        drawdown  = torch.clamp(drawdown, min=eps)
        calmar    = ann_ret / drawdown
        return torch.clamp(calmar, -10.0, 10.0)

    # ──────────────────────────────────────────────────────────────────────
    # 组合级评分组件
    # ──────────────────────────────────────────────────────────────────────

    def _ts_ic_stability(self, factors: Tensor, target_ret: Tensor) -> float:
        N, T = factors.shape
        if T < 10:
            return 0.0
        ic_list = []
        for n in range(N):
            x = factors[n, :-1]
            y = target_ret[n, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic.item())
        if not ic_list:
            return 0.0
        ic_mean = sum(ic_list) / len(ic_list)
        ic_std  = (sum((v - ic_mean) ** 2 for v in ic_list) / len(ic_list)) ** 0.5
        stability = ic_mean / (ic_std + 1e-6)
        return float(max(-3.0, min(3.0, stability)))

    def _symbol_consistency(
        self,
        per_symbol_sortino: list[float],
        per_symbol_trade_count: list[int] | None = None,
        eval_bars: int = 0,
    ) -> float:
        N = len(per_symbol_sortino)
        if N == 0:
            return 0.0
        min_trades = max(5, eval_bars // 100) if eval_bars > 0 else 5
        if per_symbol_trade_count is not None:
            n_inactive = sum(1 for c in per_symbol_trade_count if c < min_trades)
            inactive_ratio = n_inactive / N
            if inactive_ratio > 0.4:
                return -3.0
        else:
            n_inactive = 0
            inactive_ratio = 0.0
        if any(s < -2.0 for s in per_symbol_sortino):
            return -2.0
        if per_symbol_trade_count is not None:
            active_sortinos = [
                s for s, c in zip(per_symbol_sortino, per_symbol_trade_count)
                if c >= min_trades
            ]
        else:
            active_sortinos = per_symbol_sortino
        if not active_sortinos:
            return -3.0
        n_positive = sum(1 for s in active_sortinos if s > 0)
        ratio = n_positive / len(active_sortinos)
        if ratio < 0.6:
            score = (ratio - 0.6) / 0.6 * 1.0
        else:
            score = (ratio - 0.6) / 0.4 * 1.0
        if ratio == 1.0:
            score += 0.5
        return float(score)

    def _cost_stress(
        self,
        position:   Tensor,
        target_ret: Tensor,
        stress_mult: float = 2.0,
    ) -> float:
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        stressed_pnl = position * target_ret - turnover * self.cost_rate * stress_mult
        sortino = self._sortino(stressed_pnl)
        return float(torch.clamp(sortino, -5.0, 5.0))

    def _turnover_quality(self, position: Tensor) -> float:
        N, T = position.shape
        pos_2d = position.tolist()
        all_runs, total_trades = [], 0
        for n in range(N):
            runs, cur_len, cur_dir = [], 0, 0
            for p in pos_2d[n]:
                pi = int(p)
                if pi != 0:
                    if pi == cur_dir:
                        cur_len += 1
                    else:
                        if cur_len > 0: runs.append(cur_len)
                        cur_dir, cur_len = pi, 1
                else:
                    if cur_len > 0: runs.append(cur_len)
                    cur_dir, cur_len = 0, 0
            if cur_len > 0: runs.append(cur_len)
            all_runs.extend(runs)
            total_trades += len(runs)
        total_bars    = N * T
        target_trades = total_bars / 12.0
        actual_ratio  = total_trades / max(target_trades, 1.0)
        if actual_ratio <= 0:
            freq_score = -2.0
        elif actual_ratio < 0.05:
            freq_score = -2.0 + actual_ratio / 0.05
        elif actual_ratio < 0.5:
            freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
        elif actual_ratio <= 2.0:
            log_r = math.log(actual_ratio) / math.log(2.0)
            freq_score = 1.0 * math.exp(-0.5 * log_r ** 2)
        elif actual_ratio <= 8.0:
            freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
        else:
            freq_score = -2.0
        hold_bonus = 0.0
        if all_runs:
            avg_hold = sum(all_runs) / len(all_runs)
            hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)
        return float(freq_score + hold_bonus)

    def _beta_neutral_penalty(self, position: Tensor) -> float:
        flat = position.reshape(-1)
        long_ratio = (flat > 0.05).float().mean().item()
        short_ratio = (flat < -0.05).float().mean().item()
        max_ratio = max(long_ratio, short_ratio)
        if max_ratio > 0.85:
            excess = (max_ratio - 0.85) / 0.15
            return -2.0 * excess
        elif max_ratio > 0.70:
            excess = (max_ratio - 0.70) / 0.15
            return -0.5 * excess
        return 0.0

    def _half_consistency_bonus(self, pnl: Tensor) -> float:
        T = pnl.shape[1]
        if T < 20:
            return 0.0
        half = T // 2
        s1 = self._sortino(pnl[:, :half]).item()
        s2 = self._sortino(pnl[:, half:]).item()
        if s1 > 0 and s2 > 0:
            return 0.5
        elif s1 * s2 < 0:
            return -1.0
        return 0.0

    def _exposure_penalty(self, position: Tensor) -> float:
        flat = position.reshape(-1).abs()
        exposure = flat.mean().item()
        if exposure < 0.10:
            return float((exposure / 0.10 - 1.0) * 2.0)
        return 0.0

    def _turnover_penalty(self, turnover: Tensor) -> Tensor:
        mean_to = turnover.mean()
        penalty = torch.clamp(
            (mean_to - 0.2) * 3.0,
            min=0.0,
            max=3.0,
        )
        return -penalty

    # ──────────────────────────────────────────────────────────────────────
    # Walk-Forward 辅助接口
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_fold(
        self,
        factors:     Tensor,
        target_ret:  Tensor,
        train_start: int,
        train_end:   int,
        val_start:   int,
        val_end:     int,
        *,
        symbol: str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """在指定训练/验证切片上计算组合多目标得分。

        A股适配：新增 symbol 参数，用于自动判断成本率和仅做多。
        """
        sym = symbol or self.symbol
        # 自动推断成本率
        cost_rate = self._detect_cost_rate(sym)
        # 自动推断 periods_per_year
        periods = self._detect_periods_per_year(sym)
        # 是否允许做空
        try:
            from config import Config
            allow_short = Config.allow_short(sym)
        except Exception:
            allow_short = True

        position = compute_target_positions_stateless(factors, allow_short=allow_short, symbol=sym)

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * cost_rate

        pnl_train = pnl[:, train_start:train_end]
        pnl_val   = pnl[:, val_start:val_end]

        train_bars = train_end - train_start
        train_score = self._multi_objective(
            factors[:, train_start:train_end],
            target_ret[:, train_start:train_end],
            pnl_train,
            position[:, train_start:train_end],
            eval_bars=train_bars,
            symbol=sym,
        ) + self._turnover_penalty(turnover[:, train_start:train_end])

        val_bars = val_end - val_start
        base_val    = self._multi_objective(
            factors[:, val_start:val_end],
            target_ret[:, val_start:val_end],
            pnl_val,
            position[:, val_start:val_end],
            eval_bars=val_bars,
            symbol=sym,
        )
        oos_sor = self._sortino(pnl_val).item()
        if oos_sor <= 0:
            mult = max(0.1, 0.5 + oos_sor * 0.4)
        else:
            mult = min(1.2, 1.0 + oos_sor * 0.1)
        val_score = base_val * mult

        return train_score, val_score

    def _reversal_bonus(self, factors: Tensor) -> Tensor:
        N = factors.shape[0]
        scores = []
        for n in range(N):
            x = factors[n, :-1]
            y = factors[n, 1:]
            xm = x - x.mean(); ym = y - y.mean()
            sx = (xm**2).mean().sqrt(); sy = (ym**2).mean().sqrt()
            ac1 = (xm*ym).mean() / (sx*sy + 1e-8) if sx > 1e-6 and sy > 1e-6 else torch.tensor(0.0)
            bonus = 1.0 - torch.abs(ac1)
            if ac1 < 0:
                bonus = bonus + 0.5
            bonus = torch.clamp(bonus, -1.0, 2.0)
            scores.append(bonus)
        return torch.stack(scores).mean()

    def _symmetry_check(self, position: Tensor) -> Tensor:
        long_ratio  = (position > 0).float().mean()
        short_ratio = (position < 0).float().mean()
        deviation = torch.abs(long_ratio - 0.5) + torch.abs(short_ratio - 0.5)
        bonus = 1.0 - 2.0 * deviation
        return torch.clamp(bonus, -1.0, 1.0)

    def _multi_objective(
        self,
        factors:    Tensor,
        target_ret: Tensor,
        pnl:        Tensor,
        position:   Tensor,
        eval_bars:  int = 0,
        *,
        symbol: str | None = None,
    ) -> Tensor:
        """收益优先的多目标评分。

        A股适配：
          - 新增 'a_share' 奖励模式（仅做多，不奖励多空对称）
          - 单品种 A 股模式下不检查 beta_neutral（仅做多天然不平衡）
        """
        N = pnl.shape[0]
        ann_ret = pnl.mean() * self.periods_per_year
        port_sortino = self._sortino(pnl)
        port_calmar  = self._calmar(pnl)
        ts_ic        = self._ts_ic_stability(factors, target_ret)
        tq           = self._turnover_quality(position)
        exp_pen      = self._exposure_penalty(position)

        # 判断是否 A 股
        is_a_share = False
        if symbol is not None:
            try:
                from config import _is_a_share_symbol
                is_a_share = _is_a_share_symbol(symbol)
            except Exception:
                pass

        if N == 1:
            beta_pen = self._beta_neutral_penalty(position) if not is_a_share else 0.0
            consist = self._half_consistency_bonus(pnl)

            if ModelConfig.REWARD_MODE == "forex":
                rev_bonus = self._reversal_bonus(factors)
                sym_bonus = self._symmetry_check(position)
                return (
                    0.25 * ann_ret
                    + 0.05 * port_sortino
                    + 0.05 * port_calmar
                    + 0.25 * ts_ic
                    + 0.20 * rev_bonus
                    + 0.15 * sym_bonus
                    + 0.05 * tq
                    + exp_pen
                    + beta_pen
                    + consist
                )

            if ModelConfig.REWARD_MODE == "a_share":
                # A 股仅做多模式：
                #   - 不奖励多空对称（天然只有多/空=空仓）
                #   - 降低反转奖励（A 股趋势性更强，反转策略效果有限）
                #   - 提高 IC 权重（选股质量是核心）
                #   - 提高一致性奖励（防止只在牛市有效）
                rev_bonus = self._reversal_bonus(factors) * 0.5  # 反转奖励减半
                return (
                    0.55 * ann_ret           # 年化收益主目标（仅做多）
                    + 0.10 * port_sortino    # 风险调整辅助
                    + 0.10 * port_calmar     # 回撤控制
                    + 0.15 * ts_ic           # 信号质量（选股核心）
                    + 0.05 * rev_bonus       # 反转奖励（降权）
                    + 0.05 * tq              # 交易频率
                    + exp_pen                # 稀疏惩罚
                    + consist                # 前后一致性（防止 beta 因子）
                )

            if ModelConfig.REWARD_MODE == "ftmo":
                return (
                    0.80 * ann_ret
                    + 0.05 * port_sortino
                    + 0.10 * port_calmar
                    + 0.03 * ts_ic
                    + 0.02 * tq
                    + exp_pen
                    + beta_pen
                    + consist
                )
            return (
                0.60 * ann_ret
                + 0.15 * port_sortino
                + 0.10 * port_calmar
                + 0.10 * ts_ic
                + 0.05 * tq
                + exp_pen
                + beta_pen
                + consist
            )

        per_sym_sortino     = []
        per_sym_trade_count = []
        for n in range(N):
            per_sym_sortino.append(self._sortino(pnl[n]).item())
            pos_n = position[n].abs()
            diff = (pos_n[1:] - pos_n[:-1]).abs()
            trades = int((diff > 0.1).sum().item())
            per_sym_trade_count.append(trades)

        sym_cons = self._symbol_consistency(
            per_sym_sortino, per_sym_trade_count, eval_bars=eval_bars
        )
        cost_s   = self._cost_stress(position, target_ret)
        beta_pen = self._beta_neutral_penalty(position) if not is_a_share else 0.0
        consist  = self._half_consistency_bonus(pnl)

        if ModelConfig.REWARD_MODE == "ftmo":
            return (
                0.75 * ann_ret
                + 0.05 * port_sortino
                + 0.10 * port_calmar
                + 0.02 * ts_ic
                + 0.03 * sym_cons
                + 0.02 * cost_s
                + 0.03 * tq
                + exp_pen
                + beta_pen
                + consist
            )

        return (
            0.60 * ann_ret
            + 0.10 * port_sortino
            + 0.05 * port_calmar
            + 0.10 * ts_ic
            + 0.05 * sym_cons
            + 0.05 * cost_s
            + 0.05 * tq
            + exp_pen
            + beta_pen
            + consist
        )

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口（非 Walk-Forward 模式）
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        factors:    Tensor,
        raw_dict:   dict,
        target_ret: Tensor,
        *,
        symbol: str | None = None,
    ) -> tuple[Tensor, float]:
        """评估一组 Alpha 因子（含 OOS 80/20 门控）。

        A股适配：新增 symbol 参数。
        """
        sym = symbol or self.symbol
        cost_rate = self._detect_cost_rate(sym)
        try:
            from config import Config
            allow_short = Config.allow_short(sym)
        except Exception:
            allow_short = True

        position = compute_target_positions_stateless(factors, allow_short=allow_short, symbol=sym)

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * cost_rate

        T     = factors.shape[1]
        split = int(math.floor(T * 0.8))

        score = self._multi_objective(
            factors[:, :split], target_ret[:, :split],
            pnl[:, :split], position[:, :split],
            eval_bars=split,
            symbol=sym,
        ) + self._turnover_penalty(turnover[:, :split])

        pnl_oos = pnl[:, split:]
        oos_sor = self._sortino(pnl_oos).item()
        if oos_sor <= 0:
            mult = max(0.1, 0.5 + oos_sor * 0.4)
            score = score * mult
        else:
            score = score * min(1.2, 1.0 + oos_sor * 0.1)

        mean_oos = pnl_oos.mean().item()
        return score, mean_oos
