"""
=============================================================================
DIGITAL ASSETS IN INSTITUTIONAL PORTFOLIOS
Module 3: On-Chain Risk Indicators as Portfolio Signals

Author : Paolo Maizza — NUS Master in Management
Date   : 2025–2026
Purpose: Test SOPR and MVRV-based risk filters for BTC allocation sizing.

Outputs (saved to outputs/module3):
  - fig10_sopr_signal.png
  - fig11_mvrv_zscore.png
  - fig12_cumulative_returns.png
  - fig13_strategy_comparison.png
  - module3_strategy_comparison.csv
=============================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import module1_portfolio_optimization as m1
except ModuleNotFoundError:
    # Allow imports when running from repo root in ad-hoc analysis shells.
    sys.path.append(str(Path(__file__).resolve().parent))
    import module1_portfolio_optimization as m1


# Reuse Module 1 configuration and style primitives
COLORS = m1.COLORS
set_style = m1.set_style
add_source_note = m1.add_source_note
add_watermark = m1.add_watermark

CONFIG = {
    "start_date": m1.CONFIG["start_date"],
    "end_date": m1.CONFIG["end_date"],
    "trading_days": m1.CONFIG["trading_days"],
    "output_dir": "outputs/module3",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# =============================================================================
# 1) SYNTHETIC MARKET + ON-CHAIN LAYER
# =============================================================================

def gbm_series(
    s0: float,
    mu: float,
    sigma: float,
    n: int,
    seed: int,
) -> np.ndarray:
    """GBM generator aligned with module1 synthetic setup."""
    rng = np.random.default_rng(seed)
    dt = 1 / 365
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n)
    return s0 * np.exp(np.cumsum(log_ret))


def _apply_hump(series: pd.Series, start: str, end: str, amplitude: float) -> pd.Series:
    """Apply a smooth hump (positive or negative) over a date window."""
    mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
    if mask.sum() == 0:
        return series
    x = np.linspace(-np.pi, np.pi, mask.sum())
    profile = (np.cos(x) + 1.0) / 2.0
    series.loc[mask] = series.loc[mask] + amplitude * profile
    return series


def build_synthetic_market_data() -> pd.DataFrame:
    """Build daily synthetic BTC/ETH/SPY/AGG series with module1 GBM params/seeds."""
    idx = pd.date_range(CONFIG["start_date"], CONFIG["end_date"], freq="D")
    n = len(idx)

    data = {
        "BTC": gbm_series(10_000, 0.60, 0.75, n, seed=1),
        "ETH": gbm_series(800, 0.55, 0.85, n, seed=2),
        "SPY": gbm_series(250, 0.12, 0.18, n, seed=3),
        "AGG": gbm_series(105, 0.02, 0.05, n, seed=4),
    }
    prices = pd.DataFrame(data, index=idx)
    prices.index.name = "Date"
    return prices


def _build_sopr_series(idx: pd.DatetimeIndex) -> pd.Series:
    """SOPR around 1.0 with cycle-aware spikes/drops and 7d smoothing."""
    n = len(idx)
    rng = np.random.default_rng(42)

    # Mean-reverting process around 1.0 (institutional SOPR baseline)
    sopr_vals = np.empty(n)
    sopr_vals[0] = 1.0
    for i in range(1, n):
        sopr_vals[i] = sopr_vals[i - 1] + 0.08 * (1.0 - sopr_vals[i - 1]) + rng.normal(0, 0.0028)

    sopr = pd.Series(sopr_vals, index=idx)
    sopr = sopr + 0.006 * np.sin(np.linspace(0, 14 * np.pi, n))

    # Bull euphoria peaks
    sopr = _apply_hump(sopr, "2021-10-01", "2021-11-30", +0.080)
    sopr = _apply_hump(sopr, "2024-02-01", "2024-03-31", +0.075)

    # Bear capitulation drops
    sopr = _apply_hump(sopr, "2018-11-01", "2018-12-31", -0.070)
    sopr = _apply_hump(sopr, "2022-06-01", "2022-06-30", -0.055)
    sopr = _apply_hump(sopr, "2022-11-01", "2022-11-30", -0.060)

    return sopr.clip(0.90, 1.10)


def _build_mvrv_z_series(btc: pd.Series, realized_price: pd.Series) -> pd.Series:
    """MVRV Z-score calibrated to tops/bottoms while tied to BTC momentum."""
    std_btc = float(btc.std())

    # Requested base formula component
    base_formula = (btc - realized_price) / std_btc
    scaling_factor = 55.0

    # Correlate with BTC momentum to create cyclical signal dynamics
    momentum = btc.pct_change(30).fillna(0.0).rolling(14, min_periods=1).mean()

    # Compress long bull-trend drift while retaining cycle structure
    mvrv_z = 3.5 + 3.0 * np.tanh((base_formula * scaling_factor) / 3.0) + momentum * 6.0
    mvrv_z = pd.Series(mvrv_z, index=btc.index)

    # Force cycle-top / cycle-bottom behavior in known windows
    mvrv_z = _apply_hump(mvrv_z, "2021-10-15", "2021-11-30", +2.3)
    mvrv_z = _apply_hump(mvrv_z, "2024-02-01", "2024-03-31", +2.5)
    mvrv_z = _apply_hump(mvrv_z, "2018-11-01", "2018-12-31", -5.5)
    mvrv_z = _apply_hump(mvrv_z, "2022-11-01", "2022-11-30", -8.5)

    return mvrv_z.clip(-0.5, 9.0)


def build_synthetic_onchain(prices: pd.DataFrame) -> pd.DataFrame:
    """Generate realistic synthetic SOPR, Realized Price, and MVRV Z-Score."""
    btc = prices["BTC"]

    sopr = _build_sopr_series(prices.index)
    sopr_7d = sopr.rolling(7, min_periods=1).mean()

    # Realized price proxy requested in the spec
    realized_price = btc.ewm(span=180, adjust=False).mean() * 0.72

    mvrv_z = _build_mvrv_z_series(btc, realized_price)

    onchain = pd.DataFrame(
        {
            "SOPR": sopr,
            "SOPR_7D": sopr_7d,
            "Realized_Price": realized_price,
            "MVRV_Z": mvrv_z,
        },
        index=prices.index,
    )
    onchain.index.name = "Date"
    return onchain


# =============================================================================
# 2) STRATEGY ENGINE
# =============================================================================

def strategy_weights_sopr(onchain: pd.DataFrame) -> pd.Series:
    """Strategy B target BTC weights with hysteresis on SOPR 7d signal."""
    w = pd.Series(index=onchain.index, dtype=float)
    reduced = False
    for dt, val in onchain["SOPR_7D"].items():
        if not reduced and val > 1.05:
            reduced = True
        elif reduced and val < 0.98:
            reduced = False
        w.loc[dt] = 0.01 if reduced else 0.04
    return w


def strategy_weights_mvrv(onchain: pd.DataFrame) -> pd.Series:
    """Strategy C target BTC weights with hysteresis on MVRV Z-score."""
    w = pd.Series(index=onchain.index, dtype=float)
    reduced = False
    for dt, val in onchain["MVRV_Z"].items():
        if not reduced and val > 7.0:
            reduced = True
        elif reduced and val < 5.0:
            reduced = False
        w.loc[dt] = 0.00 if reduced else 0.04
    return w


def _target_weights(btc_w: float, eth_w: float = 0.0) -> Dict[str, float]:
    """Match module1 build_crypto_portfolio funding logic."""
    crypto_total = btc_w + eth_w
    scale = 1.0 - crypto_total
    return {
        "SPY": 0.60 * scale,
        "AGG": 0.40 * scale,
        "BTC": btc_w,
        "ETH": eth_w,
    }


def _contiguous_true_spans(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if mask.empty:
        return spans
    current_start = None
    prev_dt = None
    for dt, flag in mask.items():
        if flag and current_start is None:
            current_start = dt
        if not flag and current_start is not None:
            spans.append((current_start, prev_dt if prev_dt is not None else dt))
            current_start = None
        prev_dt = dt
    if current_start is not None:
        spans.append((current_start, mask.index[-1]))
    return spans


def simulate_portfolio(
    prices: pd.DataFrame,
    btc_target_weight: pd.Series,
    quarterly_rebalance_dates: Sequence[pd.Timestamp],
) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """
    Holdings-based backtest:
      - Rebalance quarterly
      - Rebalance additionally on signal weight-change dates
      - Funding logic mirrors module1 60/40 + crypto overlay construction
    """
    assets = ["SPY", "AGG", "BTC", "ETH"]
    idx = prices.index

    btc_target_weight = btc_target_weight.reindex(idx).ffill().bfill()

    signal_change = btc_target_weight.diff().abs().fillna(0) > 1e-12
    signal_dates = idx[signal_change]
    event_dates = set(quarterly_rebalance_dates).union(set(signal_dates))

    port_value = pd.Series(index=idx, dtype=float)
    btc_alloc_actual = pd.Series(index=idx, dtype=float)

    first_dt = idx[0]
    w0 = _target_weights(float(btc_target_weight.loc[first_dt]), eth_w=0.0)
    shares = {a: w0[a] / float(prices.loc[first_dt, a]) for a in assets}
    port_value.loc[first_dt] = 1.0
    btc_alloc_actual.loc[first_dt] = w0["BTC"]

    for dt in idx[1:]:
        value = float(sum(shares[a] * float(prices.loc[dt, a]) for a in assets))
        port_value.loc[dt] = value
        btc_alloc_actual.loc[dt] = (shares["BTC"] * float(prices.loc[dt, "BTC"])) / value if value > 0 else 0.0

        if dt in event_dates:
            target = _target_weights(float(btc_target_weight.loc[dt]), eth_w=0.0)
            for a in assets:
                px = float(prices.loc[dt, a])
                shares[a] = (value * target[a] / px) if px > 0 else 0.0
            btc_alloc_actual.loc[dt] = target["BTC"]

    port_returns = np.log(port_value / port_value.shift(1)).dropna()
    return port_value, port_returns, signal_dates


def compute_metrics(ret_series: pd.Series, label: str) -> Dict[str, float]:
    return {
        "Strategy": label,
        "Ann. Return": m1.ann_return(ret_series),
        "Ann. Volatility": m1.ann_volatility(ret_series),
        "Sharpe Ratio": m1.sharpe(ret_series),
        "Sortino Ratio": m1.sortino(ret_series),
        "Calmar Ratio": m1.calmar(ret_series),
        "Max Drawdown": m1.max_drawdown(ret_series),
        "VaR 95% (daily)": m1.var_95(ret_series),
    }


# =============================================================================
# 3) CHARTS
# =============================================================================

def plot_fig10_sopr_signal(prices: pd.DataFrame, onchain: pd.DataFrame) -> str:
    set_style()
    fig, ax1 = plt.subplots(figsize=(13, 6))

    ax1.plot(prices.index, prices["BTC"], color=COLORS["BTC"], lw=1.6, label="BTC Price")
    ax1.set_yscale("log")
    ax1.set_ylabel("BTC Price (USD, log scale)")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(onchain.index, onchain["SOPR_7D"], color="#3A3A3A", lw=1.3, label="SOPR (7d avg)")
    ax2.axhline(1.05, color="#B22222", ls="--", lw=1.2)
    ax2.axhline(0.98, color="#2E8B57", ls="--", lw=1.2)
    ax2.set_ylabel("SOPR (7d avg)")

    euphoric_mask = onchain["SOPR_7D"] > 1.05
    for start, end in _contiguous_true_spans(euphoric_mask):
        ax1.axvspan(start, end, color="#F7931A", alpha=0.14)

    ax1.set_title("Figure 10 — SOPR Signal: Identifying BTC Euphoria & Capitulation Periods")
    add_source_note(ax1, "Source: Synthetic on-chain data calibrated to BTC cycles | Author: Paolo Maizza")
    add_watermark(fig)

    output = os.path.join(CONFIG["output_dir"], "fig10_sopr_signal.png")
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_fig11_mvrv(onchain: pd.DataFrame) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(13, 5.5))

    y = onchain["MVRV_Z"]
    y_max = max(9.2, float(y.max()) + 0.4)
    y_min = min(-0.8, float(y.min()) - 0.2)

    ax.axhspan(7.0, y_max, color="#B22222", alpha=0.14)
    ax.axhspan(5.0, 7.0, color="#D4AF37", alpha=0.14)
    ax.axhspan(y_min, 0.0, color="#2E8B57", alpha=0.12)

    ax.plot(onchain.index, y, color="#222222", lw=1.4)
    ax.axhline(7.0, color="#B22222", ls="--", lw=1.1)
    ax.axhline(5.0, color="#D4AF37", ls="--", lw=1.1)
    ax.axhline(0.0, color="#2E8B57", ls="--", lw=1.1)

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("MVRV Z-Score")
    ax.set_xlabel("Date")
    ax.set_title("Figure 11 — MVRV Z-Score: Market Value vs Realized Value")

    add_source_note(ax, "Source: Synthetic on-chain data calibrated to BTC cycles | Author: Paolo Maizza")
    add_watermark(fig)

    output = os.path.join(CONFIG["output_dir"], "fig11_mvrv_zscore.png")
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_fig12_cumulative_returns(values: Dict[str, pd.Series]) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    labels = {
        "A": "Strategy A — Static 4% BTC",
        "B": "Strategy B — SOPR Filtered",
        "C": "Strategy C — MVRV Filtered",
    }
    line_colors = {
        "A": "#003087",   # navy
        "B": "#F7931A",   # orange
        "C": "#2E8B57",   # green
    }

    cum = {k: (v / v.iloc[0]) * 100.0 for k, v in values.items()}

    for key in ["A", "B", "C"]:
        ax.plot(cum[key].index, cum[key], lw=1.8, color=line_colors[key], label=labels[key])

    outperform_mask = cum["B"] > cum["A"]
    ax.fill_between(
        cum["B"].index,
        cum["A"].values,
        cum["B"].values,
        where=outperform_mask.values,
        color="#F7931A",
        alpha=0.14,
        label="B outperformance vs A",
    )

    last_dt = cum["A"].index[-1]
    for key in ["A", "B", "C"]:
        last_val = float(cum[key].iloc[-1])
        ax.annotate(
            f"{last_val:.1f}",
            xy=(last_dt, last_val),
            xytext=(6, 0),
            textcoords="offset points",
            color=line_colors[key],
            fontsize=9,
            fontweight="bold",
            va="center",
        )

    ax.set_title("Figure 12 — Cumulative Returns: Static vs On-Chain Filtered Strategies")
    ax.set_ylabel("Portfolio Value (Start = 100)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")

    add_source_note(ax, "Source: Synthetic market + on-chain calibration | Author: Paolo Maizza")
    add_watermark(fig)

    output = os.path.join(CONFIG["output_dir"], "fig12_cumulative_returns.png")
    fig.savefig(output)
    plt.close(fig)
    return output


def _darken(hex_color: str, factor: float = 0.72) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_fig13_strategy_comparison(metrics_df: pd.DataFrame) -> str:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    order = ["A", "B", "C"]
    cmap = {"A": "#003087", "B": "#F7931A", "C": "#2E8B57"}

    metric_specs = [
        ("Sharpe Ratio", "Sharpe Ratio", False),
        ("Sortino Ratio", "Sortino Ratio", False),
        ("Max Drawdown", "Max Drawdown", True),
        ("Ann. Return", "Annual Return", True),
    ]

    for ax, (col, title, is_pct) in zip(axes, metric_specs):
        vals = metrics_df.set_index("Strategy").loc[order, col]
        best_key = vals.idxmax()

        bar_colors = [cmap[s] for s in order]
        best_idx = order.index(best_key)
        bar_colors[best_idx] = _darken(bar_colors[best_idx], factor=0.62)

        bars = ax.bar(order, vals.values, color=bar_colors, width=0.62)
        ax.set_title(title)
        ax.set_xlabel("Strategy")

        for i, bar in enumerate(bars):
            v = float(vals.iloc[i])
            label = f"{v * 100:.1f}%" if is_pct else f"{v:.2f}"
            y = bar.get_height()
            if y >= 0:
                va = "bottom"
                y_text = y + (0.01 if not is_pct else max(0.002, abs(y) * 0.03))
            else:
                va = "top"
                y_text = y - max(0.002, abs(y) * 0.05)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                label,
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )

    fig.suptitle(
        "Figure 13 — Strategy Performance Comparison: On-Chain Filtering Impact",
        fontsize=13,
        fontweight="bold",
    )
    add_source_note(axes[-1], "Source: Backtest on synthetic market + on-chain indicators")
    add_watermark(fig)

    output = os.path.join(CONFIG["output_dir"], "fig13_strategy_comparison.png")
    fig.savefig(output)
    plt.close(fig)
    return output


# =============================================================================
# 4) REPORTING
# =============================================================================

def print_summary(metrics_df: pd.DataFrame, sopr_weights: pd.Series) -> None:
    df = metrics_df.set_index("Strategy")

    best_sharpe_strategy = str(df["Sharpe Ratio"].idxmax())

    sharpe_a = float(df.loc["A", "Sharpe Ratio"])
    sharpe_b = float(df.loc["B", "Sharpe Ratio"])
    sopr_improvement = np.nan
    if abs(sharpe_a) > 1e-12:
        sopr_improvement = (sharpe_b - sharpe_a) / abs(sharpe_a) * 100

    dd_a = abs(float(df.loc["A", "Max Drawdown"]))
    dd_c = abs(float(df.loc["C", "Max Drawdown"]))
    dd_reduction = ((dd_a - dd_c) / dd_a * 100) if dd_a > 1e-12 else np.nan

    sopr_reduced_days = int((sopr_weights <= 0.0100001).sum())

    print("\n" + "=" * 70)
    print(" MODULE 3 — ON-CHAIN SIGNAL BACKTEST SUMMARY")
    print("=" * 70)
    print(f" Key finding: highest Sharpe strategy = Strategy {best_sharpe_strategy}")
    print(f" SOPR filter impact (B vs A Sharpe): {sopr_improvement:+.1f}%")
    print(f" MVRV filter impact (C vs A Max DD reduction): {dd_reduction:+.1f}%")
    print(f" Days with reduced BTC allocation (SOPR signal): {sopr_reduced_days:,}")


def _sopr_signal(v: float) -> str:
    if v < 0.98:
        return "🟢 ACCUMULATE"
    if v <= 1.05:
        return "🟡 NEUTRAL"
    return "🔴 REDUCE"


def _mvrv_signal(v: float) -> str:
    if v < 2.0:
        return "🟢 UNDERVALUED"
    if v <= 6.0:
        return "🟡 FAIR VALUE"
    return "🔴 OVERVALUED"


def _cost_basis_signal(pct_diff: float) -> str:
    if pct_diff > 5.0:
        return "🟢 ABOVE COST"
    if pct_diff >= -5.0:
        return "🟡 NEAR COST"
    return "🔴 BELOW COST"


def print_scorecard(prices: pd.DataFrame, onchain: pd.DataFrame) -> None:
    latest = prices.index[-1]
    sopr_v = float(onchain["SOPR_7D"].iloc[-1])
    mvrv_v = float(onchain["MVRV_Z"].iloc[-1])
    btc_v = float(prices["BTC"].iloc[-1])
    realized_v = float(onchain["Realized_Price"].iloc[-1])
    vs_realized_pct = (btc_v / realized_v - 1.0) * 100 if realized_v > 0 else np.nan

    print("\n┌─────────────────────────────────────────────┐")
    print("│  ON-CHAIN INSTITUTIONAL SCORECARD           │")
    print(f"│  As of: {latest.date()}                       │")
    print("├─────────────┬──────────────┬────────────────┤")
    print("│  Indicator  │  Value       │  Signal        │")
    print("├─────────────┼──────────────┼────────────────┤")
    print(f"│  SOPR (7d)  │  {sopr_v:>6.2f}      │  {_sopr_signal(sopr_v):<14} │")
    print(f"│  MVRV Z     │  {mvrv_v:>6.2f}      │  {_mvrv_signal(mvrv_v):<14} │")
    print(f"│  vs Realized│  {vs_realized_pct:+6.1f}%     │  {_cost_basis_signal(vs_realized_pct):<14} │")
    print("└─────────────┴──────────────┴────────────────┘")


# =============================================================================
# 5) MAIN
# =============================================================================

def main() -> None:
    print("\n=== Module 3 | On-Chain Risk Indicators as Portfolio Signals ===")

    prices = build_synthetic_market_data()
    onchain = build_synthetic_onchain(prices)

    idx = prices.index
    quarter_ends = idx.intersection(pd.date_range(idx.min(), idx.max(), freq="QE"))

    # Strategy A: static 4% BTC
    w_a = pd.Series(0.04, index=idx)

    # Strategy B: SOPR-filtered
    w_b = strategy_weights_sopr(onchain)

    # Strategy C: MVRV-filtered
    w_c = strategy_weights_mvrv(onchain)

    val_a, ret_a, _ = simulate_portfolio(prices, w_a, quarter_ends)
    val_b, ret_b, _ = simulate_portfolio(prices, w_b, quarter_ends)
    val_c, ret_c, _ = simulate_portfolio(prices, w_c, quarter_ends)

    metrics = pd.DataFrame(
        [
            compute_metrics(ret_a, "A"),
            compute_metrics(ret_b, "B"),
            compute_metrics(ret_c, "C"),
        ]
    )

    # Save CSV output
    csv_path = os.path.join(CONFIG["output_dir"], "module3_strategy_comparison.csv")
    metrics.to_csv(csv_path, index=False)

    # Render charts
    fig10 = plot_fig10_sopr_signal(prices, onchain)
    fig11 = plot_fig11_mvrv(onchain)
    fig12 = plot_fig12_cumulative_returns({"A": val_a, "B": val_b, "C": val_c})
    fig13 = plot_fig13_strategy_comparison(metrics)

    # Console report
    print_summary(metrics, w_b)
    print_scorecard(prices, onchain)

    print("\nSaved charts:")
    print(f"  - {fig10}")
    print(f"  - {fig11}")
    print(f"  - {fig12}")
    print(f"  - {fig13}")
    print("\nSaved CSV:")
    print(f"  - {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
