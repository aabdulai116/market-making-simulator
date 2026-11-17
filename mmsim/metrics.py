"""Performance statistics across many simulated trading days."""

import numpy as np

TRADING_DAYS = 252


def summarize(name, results):
    daily = np.array([r.pnl for r in results])
    cumulative = np.cumsum(daily)
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    drawdowns = running_peak - cumulative

    std = daily.std(ddof=1) if len(daily) > 1 else 0.0
    if std > 0:
        sharpe = daily.mean() / std * np.sqrt(TRADING_DAYS)
    else:
        sharpe = float("nan")

    total_volume = sum(r.volume for r in results)
    total_pnl = daily.sum()
    if total_volume > 0:
        pnl_per_lot = total_pnl / total_volume
    else:
        pnl_per_lot = 0.0

    return {
        "strategy": name,
        "days": len(daily),
        "mean_daily_pnl": daily.mean(),
        "std_daily_pnl": std,
        "sharpe_annualized": sharpe,
        "total_pnl": total_pnl,
        "max_drawdown": drawdowns.max(),
        "win_rate": (daily > 0).mean(),
        "fills_per_day": np.mean([r.fills for r in results]),
        "lots_per_day": np.mean([r.volume for r in results]),
        "pnl_per_lot": pnl_per_lot,
        "spread_capture_per_day": np.mean([r.spread_capture for r in results]),
        "markout_per_lot": np.mean([r.markout for r in results]),
        "flatten_cost_per_day": np.mean([r.flatten_cost for r in results]),
        "mean_abs_inventory": np.mean([r.mean_abs_inventory for r in results]),
        "max_abs_inventory": max(r.max_abs_inventory for r in results),
    }
