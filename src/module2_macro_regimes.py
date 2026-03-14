"""
=============================================================================
DIGITAL ASSETS IN INSTITUTIONAL PORTFOLIOS
Module 2: Macro Regimes & Rolling Correlations

Author : Paolo Maizza — NUS Master in Management
Date   : 2025–2026
Purpose: Evaluate BTC correlation behavior across macro regimes.

Regimes:
  1. Risk-On   : VIX < 20 and SPY trending up
  2. Risk-Off  : VIX > 25
  3. Inflation : 2021-03-01 to 2022-06-30
  4. Rate Hike : 2022-03-01 to 2023-07-31

Outputs (saved to outputs/module2):
  1. Rolling BTC correlations (30/90/180d) with regime shading
  2. 2x2 regime correlation heatmap grid
  3. BTC timeline colored by regime
  4. CSV exports for daily regimes and rolling correlations
=============================================================================
"""

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAVE_YFINANCE = True
except ImportError:
    HAVE_YFINANCE = False
    print("[INFO] yfinance not found — using synthetic data.")


CONFIG = {
    "start_date": "2018-01-01",
    "end_date": "2026-03-14",
    "windows": [30, 90, 180],
    "output_dir": "outputs/module2",
}

TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SPY": "SPY",
    "AGG": "AGG",
    "GLD": "GLD",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

COLORS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
    "SPY": "#003087",
    "AGG": "#6B8E23",
    "GLD": "#D4AF37",
    "DXY": "#4F4F4F",
    "VIX": "#8B0000",
    "bg": "#FFFFFF",
    "grid": "#EEEEEE",
    "text": "#1A1A1A",
    "subtext": "#555555",
}

REGIME_COLORS = {
    "Risk-On": "#2E8B57",
    "Risk-Off": "#B22222",
    "Inflation": "#D4AF37",
    "Rate Hike": "#1F77B4",
    "Neutral": "#BDBDBD",
}

REGIME_ORDER = ["Rate Hike", "Inflation", "Risk-Off", "Risk-On"]
HEATMAP_ORDER = ["Risk-On", "Risk-Off", "Inflation", "Rate Hike"]

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def set_style() -> None:
    """Institutional matplotlib style aligned with module1."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": COLORS["bg"],
    })


def add_source_note(
    ax: plt.Axes,
    note: str = "Source: Yahoo Finance | Author: Paolo Maizza",
) -> None:
    ax.annotate(
        note,
        xy=(0, -0.14),
        xycoords="axes fraction",
        fontsize=7,
        color=COLORS["subtext"],
        fontstyle="italic",
        ha="left",
    )


def add_watermark(
    fig: plt.Figure,
    text: str = "INDEPENDENT RESEARCH — NOT INVESTMENT ADVICE",
) -> None:
    fig.text(
        0.5,
        0.5,
        text,
        fontsize=30,
        color="lightgrey",
        alpha=0.15,
        ha="center",
        va="center",
        rotation=30,
        fontweight="bold",
        transform=fig.transFigure,
    )


def download_prices() -> pd.DataFrame:
    """
    Download daily close data from Yahoo Finance and align to daily calendar.
    """
    if not HAVE_YFINANCE:
        return _synthetic_prices()

    print("Downloading data from Yahoo Finance...")
    raw = yf.download(
        list(TICKERS.values()),
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        price_col = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[price_col].copy()
    else:
        prices = raw.copy()

    reverse_map = {ticker: name for name, ticker in TICKERS.items()}
    prices = prices.rename(columns=reverse_map)
    prices = prices[[c for c in TICKERS.keys() if c in prices.columns]].sort_index()
    prices.columns.name = None
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"

    full_index = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    prices = prices.reindex(full_index).ffill()
    prices.index.name = "Date"
    prices = prices.dropna()

    missing_assets = [c for c in TICKERS.keys() if c not in prices.columns]
    if missing_assets:
        raise ValueError(f"Missing downloaded assets: {missing_assets}")

    print(
        f"Loaded {len(prices):,} daily rows "
        f"({prices.index[0].date()} to {prices.index[-1].date()})"
    )
    return prices


def _synthetic_prices() -> pd.DataFrame:
    """
    Synthetic daily market data aligned with module1 GBM assumptions.
    Includes mean-reverting synthetic VIX with crisis spikes.
    """
    idx = pd.date_range(CONFIG["start_date"], CONFIG["end_date"], freq="D")
    n = len(idx)

    def gbm(s0: float, mu: float, sigma: float, n_obs: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt = 1 / 365
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n_obs)
        return s0 * np.exp(np.cumsum(log_ret))

    # Mean-reverting VIX around 18 with crisis-like spikes.
    rng_vix = np.random.default_rng(7)
    vix = np.zeros(n, dtype=float)
    vix[0] = 18.0
    kappa = 0.10
    vol = 1.15
    for i in range(1, n):
        shock = rng_vix.normal(0.0, vol)
        vix[i] = vix[i - 1] + kappa * (18.0 - vix[i - 1]) + shock

    vix_series = pd.Series(vix, index=idx)

    def add_spike(series: pd.Series, start: str, end: str, amp: float) -> pd.Series:
        mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
        if mask.sum() == 0:
            return series
        x = np.linspace(-np.pi, np.pi, mask.sum())
        profile = (np.cos(x) + 1.0) / 2.0
        series.loc[mask] = series.loc[mask] + amp * profile
        return series

    vix_series = add_spike(vix_series, "2020-03-01", "2020-03-31", 18.0)
    vix_series = add_spike(vix_series, "2022-11-01", "2022-11-30", 13.0)
    vix_series = vix_series.clip(lower=9.0, upper=65.0)

    prices = pd.DataFrame(
        {
            "BTC": gbm(10_000, 0.60, 0.75, n, 1),
            "ETH": gbm(800, 0.55, 0.85, n, 2),
            "SPY": gbm(250, 0.12, 0.18, n, 3),
            "AGG": gbm(105, 0.02, 0.05, n, 4),
            "GLD": gbm(120, 0.06, 0.14, n, 5),
            "DXY": gbm(90, 0.00, 0.07, n, 6),
            "VIX": vix_series.values,
        },
        index=idx,
    )
    prices.index.name = "Date"

    print(
        f"[SYNTHETIC DATA] Loaded {len(prices):,} daily rows "
        f"({prices.index[0].date()} to {prices.index[-1].date()})"
    )
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def build_regimes(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create regime flags and a primary regime label per day."""
    idx = prices.index
    spy = prices["SPY"]
    vix = prices["VIX"]

    sma50 = spy.rolling(50).mean()
    spy_trending_up = (spy > sma50) & (sma50 > sma50.shift(5))

    regimes = pd.DataFrame(index=idx)
    regimes["Risk-On"] = (vix < 20) & spy_trending_up
    regimes["Risk-Off"] = vix > 25
    regimes["Inflation"] = (idx >= pd.Timestamp("2021-03-01")) & (idx <= pd.Timestamp("2022-06-30"))
    regimes["Rate Hike"] = (idx >= pd.Timestamp("2022-03-01")) & (idx <= pd.Timestamp("2023-07-31"))

    # Precedence resolves overlap: policy regimes first, then volatility regimes.
    primary = pd.Series("Neutral", index=idx, name="Regime", dtype="object")
    for regime in REGIME_ORDER:
        primary.loc[regimes[regime]] = regime

    return regimes, primary


def rolling_btc_correlations(
    returns: pd.DataFrame,
    windows: List[int],
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Compute rolling Pearson correlations of BTC returns vs all other assets.
    """
    assets = [c for c in returns.columns if c != "BTC"]
    corr_by_window: Dict[int, pd.DataFrame] = {}
    long_frames: List[pd.DataFrame] = []

    for window in windows:
        corr = pd.DataFrame(index=returns.index)
        for asset in assets:
            corr[asset] = returns["BTC"].rolling(window).corr(returns[asset])
        corr_by_window[window] = corr

        melted = (
            corr.reset_index()
            .melt(id_vars="Date", var_name="Asset", value_name="Correlation")
            .dropna(subset=["Correlation"])
        )
        melted["Window"] = window
        long_frames.append(melted)

    all_corr = pd.concat(long_frames, ignore_index=True)
    return corr_by_window, all_corr


def _shade_regimes(ax: plt.Axes, regime_series: pd.Series) -> None:
    """Shade contiguous periods by primary regime."""
    series = regime_series.dropna()
    if series.empty:
        return

    start = series.index[0]
    current = series.iloc[0]
    for date, regime in series.iloc[1:].items():
        if regime != current:
            if current in REGIME_COLORS and current != "Neutral":
                ax.axvspan(start, date, color=REGIME_COLORS[current], alpha=0.07, lw=0)
            start = date
            current = regime

    if current in REGIME_COLORS and current != "Neutral":
        ax.axvspan(start, series.index[-1], color=REGIME_COLORS[current], alpha=0.07, lw=0)


def plot_rolling_correlations(
    corr_by_window: Dict[int, pd.DataFrame],
    regime_primary: pd.Series,
) -> str:
    set_style()
    ordered_windows = sorted(corr_by_window.keys())
    fig, axes = plt.subplots(len(ordered_windows), 1, figsize=(14, 10), sharex=True)
    if len(ordered_windows) == 1:
        axes = [axes]

    asset_colors = {k: v for k, v in COLORS.items() if k in ["ETH", "SPY", "AGG", "GLD", "DXY", "VIX"]}

    for ax, window in zip(axes, ordered_windows):
        corr = corr_by_window[window]
        _shade_regimes(ax, regime_primary.reindex(corr.index))
        for asset in corr.columns:
            ax.plot(
                corr.index,
                corr[asset],
                label=asset,
                lw=1.2,
                color=asset_colors.get(asset, "#333333"),
            )
        ax.axhline(0, color="#666666", lw=0.9, ls="--")
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Correlation")
        ax.set_title(f"BTC Rolling Correlations ({window}-day)")

    axes[-1].set_xlabel("Date")

    asset_handles = [
        Line2D([0], [0], color=asset_colors[a], lw=2, label=a) for a in corr_by_window[ordered_windows[0]].columns
    ]
    regime_handles = [
        Patch(facecolor=REGIME_COLORS[r], alpha=0.25, label=r) for r in HEATMAP_ORDER
    ]
    axes[0].legend(
        handles=asset_handles + regime_handles,
        loc="upper left",
        ncol=5,
        frameon=True,
    )

    fig.suptitle(
        "BTC vs Cross-Asset Rolling Correlations with Macro Regime Shading",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    add_source_note(
        axes[-1],
        "Source: Yahoo Finance | Regimes: Risk-On (VIX<20 & SPY uptrend), Risk-Off (VIX>25), Inflation, Rate Hike",
    )
    add_watermark(fig)

    output_path = os.path.join(CONFIG["output_dir"], "module2_rolling_correlations_regimes.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_regime_heatmaps(returns: pd.DataFrame, regime_primary: pd.Series) -> Tuple[str, pd.DataFrame]:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    heatmap_records: List[pd.DataFrame] = []
    columns = ["BTC", "ETH", "SPY", "AGG", "GLD", "DXY", "VIX"]

    for ax, regime in zip(axes, HEATMAP_ORDER):
        mask = regime_primary.reindex(returns.index) == regime
        regime_returns = returns.loc[mask, columns].copy()

        if len(regime_returns) < 3:
            ax.text(0.5, 0.5, f"Insufficient data\n{regime}", ha="center", va="center")
            ax.axis("off")
            continue

        corr_matrix = regime_returns.corr()
        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            cbar=True,
            annot=False,
            linewidths=0.25,
            linecolor="#F0F0F0",
        )
        ax.set_title(f"{regime} Regime Correlations (n={len(regime_returns):,})")

        # Build long table explicitly to avoid pandas axis-name collisions.
        rows = []
        for asset_1 in corr_matrix.index:
            for asset_2 in corr_matrix.columns:
                rows.append(
                    {
                        "Asset_1": asset_1,
                        "Asset_2": asset_2,
                        "Correlation": float(corr_matrix.loc[asset_1, asset_2]),
                        "Regime": regime,
                    }
                )
        stacked = pd.DataFrame(rows)
        heatmap_records.append(stacked)

    fig.suptitle(
        "Cross-Asset Correlation Structure by Macro Regime",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    add_source_note(axes[-1], "Source: Yahoo Finance | Correlations computed on daily log returns")
    add_watermark(fig)

    output_path = os.path.join(CONFIG["output_dir"], "module2_regime_heatmap_grid.png")
    fig.savefig(output_path)
    plt.close(fig)

    heatmap_df = pd.concat(heatmap_records, ignore_index=True) if heatmap_records else pd.DataFrame()
    return output_path, heatmap_df


def plot_btc_regime_timeline(prices: pd.DataFrame, regime_primary: pd.Series) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    btc = prices["BTC"]
    ax.plot(btc.index, btc, color="#808080", lw=1.0, alpha=0.5, label="BTC (context)")

    for regime in HEATMAP_ORDER:
        mask = regime_primary.reindex(btc.index) == regime
        ax.scatter(
            btc.index[mask],
            btc[mask],
            s=8,
            color=REGIME_COLORS[regime],
            alpha=0.9,
            label=regime,
            zorder=3,
        )

    ax.set_title("BTC Price Timeline Colored by Macro Regime")
    ax.set_ylabel("BTC Price (USD)")
    ax.set_xlabel("Date")
    ax.legend(ncol=5, loc="upper left")
    add_source_note(ax)
    add_watermark(fig)

    output_path = os.path.join(CONFIG["output_dir"], "module2_btc_timeline_by_regime.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def export_csvs(
    prices: pd.DataFrame,
    regime_flags: pd.DataFrame,
    regime_primary: pd.Series,
    rolling_long: pd.DataFrame,
    corr_by_window: Dict[int, pd.DataFrame],
    heatmap_df: pd.DataFrame,
) -> List[str]:
    paths: List[str] = []

    daily = pd.DataFrame(index=prices.index)
    daily["BTC"] = prices["BTC"]
    daily["SPY"] = prices["SPY"]
    daily["VIX"] = prices["VIX"]
    daily = daily.join(regime_flags).assign(Regime=regime_primary)
    daily_path = os.path.join(CONFIG["output_dir"], "module2_daily_regimes.csv")
    daily.to_csv(daily_path, index_label="Date")
    paths.append(daily_path)

    rolling_long = rolling_long.copy()
    rolling_long["Regime"] = rolling_long["Date"].map(regime_primary)
    rolling_path = os.path.join(CONFIG["output_dir"], "module2_rolling_correlations_long.csv")
    rolling_long.to_csv(rolling_path, index=False)
    paths.append(rolling_path)

    for window, corr in corr_by_window.items():
        wide = corr.copy()
        wide["Regime"] = regime_primary.reindex(wide.index)
        wide_path = os.path.join(CONFIG["output_dir"], f"module2_rolling_correlations_{window}d.csv")
        wide.to_csv(wide_path, index_label="Date")
        paths.append(wide_path)

    summary = (
        rolling_long.dropna(subset=["Correlation", "Regime"])
        .groupby(["Window", "Regime", "Asset"], as_index=False)["Correlation"]
        .mean()
        .rename(columns={"Correlation": "Mean_Correlation"})
    )
    summary_path = os.path.join(CONFIG["output_dir"], "module2_correlation_summary_by_regime.csv")
    summary.to_csv(summary_path, index=False)
    paths.append(summary_path)

    if not heatmap_df.empty:
        heatmap_path = os.path.join(CONFIG["output_dir"], "module2_regime_heatmap_values.csv")
        heatmap_df.to_csv(heatmap_path, index=False)
        paths.append(heatmap_path)

    return paths


def main() -> None:
    print("\n=== Module 2 | Macro Regimes & Rolling Correlations ===")
    prices = download_prices()
    returns = compute_log_returns(prices)

    regime_flags, regime_primary = build_regimes(prices)
    corr_by_window, rolling_long = rolling_btc_correlations(returns, CONFIG["windows"])

    chart1 = plot_rolling_correlations(corr_by_window, regime_primary)
    chart2, heatmap_df = plot_regime_heatmaps(returns, regime_primary)
    chart3 = plot_btc_regime_timeline(prices, regime_primary)
    csv_paths = export_csvs(
        prices=prices,
        regime_flags=regime_flags,
        regime_primary=regime_primary,
        rolling_long=rolling_long,
        corr_by_window=corr_by_window,
        heatmap_df=heatmap_df,
    )

    print("\nSaved charts:")
    print(f"  - {chart1}")
    print(f"  - {chart2}")
    print(f"  - {chart3}")
    print("\nSaved CSV files:")
    for path in csv_paths:
        print(f"  - {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
