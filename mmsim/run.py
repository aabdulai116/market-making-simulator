"""Run the full experiment.

1. Tune the inventory skew strategy on a set of training days.
2. Compare it with the fixed spread baseline on separate, unseen test days.
3. Save a results table and charts.

Usage:
    python -m mmsim.run --train-days 40 --test-days 100 --out results
"""

import argparse
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .market import MarketConfig, simulate_day
from .metrics import summarize
from .strategies import FixedSpreadMaker, InventorySkewMaker

TEST_SEED_OFFSET = 100_000   # test days never overlap with training days

GAMMAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
HORIZONS = [30, 60]


def run_days(strategy, config, seeds):
    results = []
    for seed in seeds:
        results.append(simulate_day(strategy, config, seed))
    return results


def tune(config, train_seeds):
    rows = []
    for gamma, horizon in itertools.product(GAMMAS, HORIZONS):
        strategy = InventorySkewMaker(gamma=gamma, horizon=horizon)
        stats = summarize(strategy.name, run_days(strategy, config, train_seeds))
        stats["gamma"] = gamma
        stats["horizon"] = horizon
        rows.append(stats)
        print(
            f"  gamma={gamma:<6} horizon={horizon:<3} "
            f"sharpe={stats['sharpe_annualized']:6.1f} "
            f"mean pnl=${stats['mean_daily_pnl']:8.0f}"
        )
    table = pd.DataFrame(rows)
    best = table.sort_values("sharpe_annualized", ascending=False).iloc[0]
    return table, float(best["gamma"]), int(best["horizon"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-days", type=int, default=40)
    parser.add_argument("--test-days", type=int, default=100)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    config = MarketConfig()
    train_seeds = list(range(args.train_days))
    test_seeds = []
    for day in range(args.test_days):
        test_seeds.append(TEST_SEED_OFFSET + day)

    print(f"Tuning inventory skew maker on {args.train_days} training days")
    grid, gamma, horizon = tune(config, train_seeds)
    grid.to_csv(os.path.join(args.out, "tuning_grid.csv"), index=False)
    print(f"Best on training days: gamma={gamma}, horizon={horizon}\n")

    strategies = [
        FixedSpreadMaker(),
        InventorySkewMaker(gamma=gamma, horizon=horizon),
    ]

    print(f"Testing on {args.test_days} unseen days")
    all_results = {}
    summary_rows = []
    for strategy in strategies:
        results = run_days(strategy, config, test_seeds)
        all_results[strategy.name] = results
        summary_rows.append(summarize(strategy.name, results))

    summary = pd.DataFrame(summary_rows).set_index("strategy")
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(
            {
                "chosen_params": {"gamma": gamma, "horizon": horizon},
                "test": summary.reset_index().to_dict(orient="records"),
            },
            handle,
            indent=2,
            default=float,
        )
    print(summary.T.round(2).to_string())

    _plot_cumulative(all_results, args.out)
    _plot_inventory(all_results, args.out)
    _plot_intraday(all_results, args.out)
    _plot_tuning(grid, args.out)
    print(f"\nCharts and tables saved to {args.out}/")


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
COLORS = {"Fixed spread": "#9a9a9a", "Inventory skew (A-S)": "#2f6db5"}


def _plot_cumulative(all_results, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, results in all_results.items():
        daily = np.array([r.pnl for r in results])
        ax.plot(np.cumsum(daily), label=name, color=COLORS.get(name), lw=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Cumulative P&L over unseen test days")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Dollars")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cumulative_pnl.png"), dpi=150)
    plt.close(fig)


def _plot_inventory(all_results, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(-65, 66, 2)
    for name, results in all_results.items():
        inventory = np.concatenate([r.inventory_path for r in results])
        ax.hist(inventory, bins=bins, alpha=0.6, density=True,
                label=name, color=COLORS.get(name))
    ax.set_title("Where inventory sits during the day")
    ax.set_xlabel("Position (lots of 100 shares)")
    ax.set_ylabel("Share of time")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "inventory_distribution.png"), dpi=150)
    plt.close(fig)


def _plot_intraday(all_results, out):
    names = list(all_results)
    fig, axes = plt.subplots(3, len(names), figsize=(12, 8), sharex=True)
    for col, name in enumerate(names):
        day = all_results[name][0]
        window = slice(0, 600)
        steps = np.arange(len(day.mid_path))[window]

        price_ax = axes[0, col]
        price_ax.plot(steps, day.mid_path[window] / 100, color="black", lw=1, label="Market mid")
        price_ax.plot(steps, day.fundamental_path[window] / 100, color="#d98c1f",
                      lw=1, ls="--", label="Hidden value")
        price_ax.plot(steps, day.bid_path[window] / 100, color="#2e8b57", lw=0.8, label="MM bid")
        price_ax.plot(steps, day.ask_path[window] / 100, color="#c0392b", lw=0.8, label="MM ask")
        price_ax.set_title(name)
        price_ax.set_ylabel("Price ($)")
        if col == 0:
            price_ax.legend(frameon=False, fontsize=8)

        axes[1, col].plot(steps, day.inventory_path[window], color=COLORS.get(name))
        axes[1, col].axhline(0, color="black", lw=0.5)
        axes[1, col].set_ylabel("Inventory (lots)")

        axes[2, col].plot(steps, day.pnl_path[window], color=COLORS.get(name))
        axes[2, col].axhline(0, color="black", lw=0.5)
        axes[2, col].set_ylabel("P&L ($)")
        axes[2, col].set_xlabel("Step (5 seconds each)")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("First 50 minutes of test day 1")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "intraday_example.png"), dpi=150)
    plt.close(fig)


def _plot_tuning(grid, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for horizon in sorted(grid["horizon"].unique()):
        rows = grid[grid["horizon"] == horizon].sort_values("gamma")
        ax.plot(rows["gamma"], rows["sharpe_annualized"], marker="o",
                label=f"horizon = {horizon} steps")
    ax.set_xscale("log")
    ax.set_xticks(GAMMAS)
    ax.set_xticklabels([str(g) for g in GAMMAS])
    ax.minorticks_off()
    ax.set_title("Training days: Sharpe ratio by risk aversion")
    ax.set_xlabel("gamma (risk aversion)")
    ax.set_ylabel("Annualized Sharpe")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tuning_sharpe.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
