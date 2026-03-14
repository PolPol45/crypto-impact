"""
=============================================================================
DIGITAL ASSETS IN INSTITUTIONAL PORTFOLIOS
Module 1: Mean-Variance Portfolio Optimization & Efficient Frontier

Author : Paolo Maizza — NUS Master in Management
Date   : 2025–2026
Purpose: Quantify the risk-adjusted benefit of adding BTC/ETH to a
         standard 60/40 institutional portfolio (SPY + AGG).

Methodology follows: VanEck (2024), ARK Invest (2025), 21Shares (2024)
Benchmark: 60% SPY / 40% AGG — standard institutional 60/40

Outputs:
  1. Efficient frontier  — with vs without crypto
  2. Sharpe/Sortino/Calmar table  — allocations 0–10% BTC+ETH
  3. Rolling Sharpe chart  — stability of optimal allocation over time
  4. Rebalancing analysis  — monthly / quarterly / annual
  5. Metrics summary CSV  — for the research report

Run:  python module1_portfolio_optimization.py
      pip install yfinance scipy numpy pandas matplotlib seaborn
=============================================================================
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats import norm
import datetime

warnings.filterwarnings("ignore")

# ── Try to import yfinance; fall back to synthetic data if unavailable ────────
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except ImportError:
    HAVE_YFINANCE = False
    print("[INFO] yfinance not found — using synthetic data. "
          "Install with: pip install yfinance")

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

CONFIG = {
    "start_date"   : "2018-01-01",
    "end_date"     : "2026-03-14",
    "rf_annual"    : 0.045,          # 3M T-Bill proxy (2024-25 avg ~4.5%)
    "trading_days" : 365,            # Crypto trades 24/7; use 365 for consistency
    "btc_weight_range": np.arange(0.00, 0.105, 0.01),   # 0% to 10%, step 1%
    "n_mc_portfolios": 5_000,        # Monte Carlo random portfolios for cloud
    "rebalance_freqs": ["ME", "QE", "YE"],  # Monthly / Quarterly / Annual (pandas 2.x)
    "output_dir"   : "outputs/module1",
}

# Asset tickers (yfinance)
TICKERS = {
    "BTC"  : "BTC-USD",
    "ETH"  : "ETH-USD",
    "SPY"  : "SPY",       # S&P 500
    "AGG"  : "AGG",       # US Aggregate Bonds
    "GLD"  : "GLD",       # Gold
    "DXY"  : "DX-Y.NYB",  # US Dollar Index
}

# Institutional colour palette  ──────────────────────────────────────────────
COLORS = {
    "BTC"        : "#F7931A",
    "ETH"        : "#627EEA",
    "SPY"        : "#003087",
    "AGG"        : "#6B8E23",
    "GLD"        : "#D4AF37",
    "60_40"      : "#003087",
    "frontier_no": "#AAAAAA",
    "frontier_yes": "#003087",
    "optimal"    : "#D4321C",
    "bg"         : "#FFFFFF",
    "grid"       : "#EEEEEE",
    "text"       : "#1A1A1A",
    "subtext"    : "#555555",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =============================================================================
# 1.  DATA LAYER
# =============================================================================

def download_prices() -> pd.DataFrame:
    """
    Download daily adjusted closing prices from Yahoo Finance.
    Falls back to synthetic data for offline/demo use.
    """
    if HAVE_YFINANCE:
        print("Downloading market data from Yahoo Finance …")
        raw = yf.download(
            list(TICKERS.values()),
            start=CONFIG["start_date"],
            end=CONFIG["end_date"],
            progress=False,
        )["Close"]
        raw.columns = list(TICKERS.keys())
        raw = raw.ffill().dropna()
        print(f"  Downloaded {len(raw):,} rows | "
              f"{raw.index[0].date()} → {raw.index[-1].date()}")
        return raw
    else:
        return _synthetic_prices()


def _synthetic_prices() -> pd.DataFrame:
    """
    Realistic synthetic price series for offline testing.
    Calibrated to approximate actual 2018–2025 statistics.
    """
    np.random.seed(42)
    dates = pd.date_range(CONFIG["start_date"], CONFIG["end_date"], freq="D")
    n = len(dates)

    def gbm(s0, mu, sigma, n, seed=None):
        rng = np.random.default_rng(seed)
        dt  = 1 / 365
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n)
        return s0 * np.exp(np.cumsum(log_ret))

    # Approximate historical stats
    data = {
        "BTC": gbm(10_000, 0.60, 0.75, n, 1),
        "ETH": gbm(800,    0.55, 0.85, n, 2),
        "SPY": gbm(250,    0.12, 0.18, n, 3),
        "AGG": gbm(105,    0.02, 0.05, n, 4),
        "GLD": gbm(120,    0.06, 0.14, n, 5),
        "DXY": gbm(90,     0.00, 0.07, n, 6),
    }
    df = pd.DataFrame(data, index=dates)
    print(f"[SYNTHETIC DATA] {len(df):,} rows | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns — standard in institutional quantitative analysis."""
    return np.log(prices / prices.shift(1)).dropna()


# =============================================================================
# 2.  PORTFOLIO METRICS ENGINE
# =============================================================================

RF_DAILY = CONFIG["rf_annual"] / CONFIG["trading_days"]


def sharpe(ret_series: pd.Series) -> float:
    excess = ret_series - RF_DAILY
    if ret_series.std() == 0:
        return 0.0
    return float(excess.mean() / ret_series.std() * np.sqrt(CONFIG["trading_days"]))


def sortino(ret_series: pd.Series) -> float:
    """Sortino ratio — penalises only downside volatility."""
    excess  = ret_series - RF_DAILY
    neg_ret = ret_series[ret_series < 0]
    downside_std = neg_ret.std() * np.sqrt(CONFIG["trading_days"])
    if downside_std == 0:
        return 0.0
    return float(excess.mean() * CONFIG["trading_days"] / downside_std)


def calmar(ret_series: pd.Series) -> float:
    """Calmar ratio = annualised return / maximum drawdown."""
    cum = (1 + ret_series).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = abs(drawdown.min())
    ann_ret = ret_series.mean() * CONFIG["trading_days"]
    if max_dd == 0:
        return 0.0
    return float(ann_ret / max_dd)


def max_drawdown(ret_series: pd.Series) -> float:
    cum = (1 + ret_series).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def var_95(ret_series: pd.Series) -> float:
    """Historical VaR at 95% confidence (daily)."""
    return float(np.percentile(ret_series, 5))


def cvar_95(ret_series: pd.Series) -> float:
    """CVaR / Expected Shortfall at 95% confidence."""
    threshold = var_95(ret_series)
    return float(ret_series[ret_series <= threshold].mean())


def ann_return(ret_series: pd.Series) -> float:
    return float(ret_series.mean() * CONFIG["trading_days"])


def ann_volatility(ret_series: pd.Series) -> float:
    return float(ret_series.std() * np.sqrt(CONFIG["trading_days"]))


def full_metrics(ret_series: pd.Series, label: str = "") -> dict:
    return {
        "Label"           : label,
        "Ann. Return"     : ann_return(ret_series),
        "Ann. Volatility" : ann_volatility(ret_series),
        "Sharpe Ratio"    : sharpe(ret_series),
        "Sortino Ratio"   : sortino(ret_series),
        "Calmar Ratio"    : calmar(ret_series),
        "Max Drawdown"    : max_drawdown(ret_series),
        "VaR 95% (daily)" : var_95(ret_series),
        "CVaR 95% (daily)": cvar_95(ret_series),
    }


# =============================================================================
# 3.  PORTFOLIO CONSTRUCTION
# =============================================================================

def portfolio_return(weights: np.ndarray, mean_ret: np.ndarray) -> float:
    return float(np.dot(weights, mean_ret) * CONFIG["trading_days"])


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights) * np.sqrt(CONFIG["trading_days"]))


def neg_sharpe(weights, mean_ret, cov):
    p_ret  = portfolio_return(weights, mean_ret)
    p_vol  = portfolio_volatility(weights, cov)
    rf     = CONFIG["rf_annual"]
    return -(p_ret - rf) / p_vol if p_vol > 0 else 0.0


def build_60_40_returns(returns: pd.DataFrame) -> pd.Series:
    """Standard 60% SPY / 40% AGG benchmark."""
    return 0.60 * returns["SPY"] + 0.40 * returns["AGG"]


def build_crypto_portfolio(returns: pd.DataFrame,
                           btc_w: float,
                           eth_w: float) -> pd.Series:
    """
    60/40 base + crypto overlay.
    BTC + ETH allocation is funded proportionally from SPY and AGG.
    """
    crypto_total = btc_w + eth_w
    scale        = 1 - crypto_total          # residual for 60/40
    w_spy = 0.60 * scale
    w_agg = 0.40 * scale
    return (w_spy  * returns["SPY"]
            + w_agg  * returns["AGG"]
            + btc_w  * returns["BTC"]
            + eth_w  * returns["ETH"])


# =============================================================================
# 4.  EFFICIENT FRONTIER  (mean-variance, Monte Carlo cloud)
# =============================================================================

def efficient_frontier_assets(returns: pd.DataFrame,
                               assets: list[str],
                               n_points: int = 80) -> pd.DataFrame:
    """
    Compute minimum-variance frontier for a given asset universe.
    Returns DataFrame with (volatility, return, sharpe) for each point.
    """
    mu  = returns[assets].mean()
    cov = returns[assets].cov()
    n   = len(assets)

    target_returns = np.linspace(
        mu.min() * CONFIG["trading_days"] * 0.8,
        mu.max() * CONFIG["trading_days"] * 1.1,
        n_points,
    )
    frontier_vols, frontier_rets = [], []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target:
             portfolio_return(w, mu.values) - t},
        ]
        bounds = [(0, 1)] * n
        w0     = np.ones(n) / n
        result = minimize(
            lambda w: portfolio_volatility(w, cov.values),
            w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if result.success:
            frontier_vols.append(result.fun)
            frontier_rets.append(target)

    rf = CONFIG["rf_annual"]
    frontier = pd.DataFrame({
        "vol"   : frontier_vols,
        "ret"   : frontier_rets,
        "sharpe": [(r - rf) / v for r, v in zip(frontier_rets, frontier_vols)],
    })
    return frontier


def monte_carlo_cloud(returns: pd.DataFrame,
                      assets: list[str],
                      n: int = 5_000) -> pd.DataFrame:
    """Random portfolio cloud for visualisation."""
    mu  = returns[assets].mean()
    cov = returns[assets].cov()
    rng = np.random.default_rng(99)

    results = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(len(assets)))
        p_ret = portfolio_return(w, mu.values)
        p_vol = portfolio_volatility(w, cov.values)
        p_shr = (p_ret - CONFIG["rf_annual"]) / p_vol
        results.append({"vol": p_vol, "ret": p_ret, "sharpe": p_shr})
    return pd.DataFrame(results)


def optimal_weights(returns: pd.DataFrame, assets: list[str]) -> np.ndarray:
    """Maximum Sharpe ratio portfolio weights."""
    mu  = returns[assets].mean()
    cov = returns[assets].cov()
    n   = len(assets)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    result = minimize(
        neg_sharpe, np.ones(n) / n, args=(mu.values, cov.values),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return result.x if result.success else np.ones(n) / n


# =============================================================================
# 5.  ALLOCATION TABLE  (0–10% BTC+ETH, step 1%)
# =============================================================================

def allocation_sweep(returns: pd.DataFrame) -> pd.DataFrame:
    """
    For each BTC+ETH total allocation (0–10%), compute all metrics.
    BTC/ETH split follows the VanEck optimal ratio: ~71.4% BTC / 28.6% ETH.
    Also tests: pure BTC, pure ETH, equal split.
    """
    BTC_SHARE = 0.714   # VanEck (2024) optimal mix
    ETH_SHARE = 1 - BTC_SHARE

    rows = []
    for total_crypto in CONFIG["btc_weight_range"]:
        btc_w = total_crypto * BTC_SHARE
        eth_w = total_crypto * ETH_SHARE
        port_ret = build_crypto_portfolio(returns, btc_w, eth_w)
        m = full_metrics(port_ret,
                         label=f"{total_crypto:.0%} crypto ({btc_w:.1%} BTC + {eth_w:.1%} ETH)")
        m["BTC Weight"] = btc_w
        m["ETH Weight"] = eth_w
        m["Crypto Total"] = total_crypto
        rows.append(m)

    df = pd.DataFrame(rows)
    # Mark optimal (max Sharpe)
    best_idx = df["Sharpe Ratio"].idxmax()
    df["Optimal"] = False
    df.loc[best_idx, "Optimal"] = True
    return df


# =============================================================================
# 6.  REBALANCING ANALYSIS
# =============================================================================

def rebalancing_analysis(prices: pd.DataFrame,
                         btc_w: float = 0.04,
                         eth_w: float = 0.02) -> pd.DataFrame:
    """
    Compare buy-and-hold vs rebalanced portfolios at different frequencies.
    Follows 21Shares (2024) methodology.
    """
    crypto_total = btc_w + eth_w
    scale        = 1 - crypto_total
    weights = np.array([0.60 * scale, 0.40 * scale, btc_w, eth_w])
    assets  = ["SPY", "AGG", "BTC", "ETH"]

    rows = []

    # Buy and hold
    port_prices = (prices[assets] / prices[assets].iloc[0] * weights).sum(axis=1)
    port_ret    = np.log(port_prices / port_prices.shift(1)).dropna()
    m = full_metrics(port_ret, "Buy & Hold")
    m["Rebalancing"] = "Buy & Hold"
    rows.append(m)

    # Rebalanced portfolios
    labels = {"ME": "Monthly", "QE": "Quarterly", "YE": "Annual"}
    for freq in CONFIG["rebalance_freqs"]:
        monthly_prices = prices[assets].resample(freq).last()
        cum_ret_list   = []

        for i in range(len(monthly_prices) - 1):
            period_prices = prices[assets].loc[
                monthly_prices.index[i] : monthly_prices.index[i + 1]
            ]
            if len(period_prices) < 2:
                continue
            period_ret  = np.log(period_prices / period_prices.shift(1)).dropna()
            period_port = period_ret @ weights
            cum_ret_list.extend(period_port.values)

        if not cum_ret_list:
            continue
        s = pd.Series(cum_ret_list)
        m = full_metrics(s, labels[freq])
        m["Rebalancing"] = labels[freq]
        rows.append(m)

    return pd.DataFrame(rows)


# =============================================================================
# 7.  ROLLING SHARPE
# =============================================================================

def rolling_sharpe(returns: pd.DataFrame,
                   window: int = 252) -> pd.DataFrame:
    """
    Rolling 1-year Sharpe for 60/40, optimal crypto portfolio, and BTC alone.
    """
    base   = build_60_40_returns(returns)
    crypto = build_crypto_portfolio(returns, btc_w=0.04, eth_w=0.016)
    btc    = returns["BTC"]

    def roll(s):
        return s.rolling(window).apply(sharpe, raw=False)

    return pd.DataFrame({
        "60/40"          : roll(base),
        "60/40 + 4% Crypto": roll(crypto),
        "BTC only"       : roll(btc),
    }).dropna()


# =============================================================================
# 8.  MATPLOTLIB INSTITUTIONAL STYLE
# =============================================================================

def set_style():
    plt.rcParams.update({
        "figure.facecolor"   : COLORS["bg"],
        "axes.facecolor"     : COLORS["bg"],
        "axes.edgecolor"     : "#CCCCCC",
        "axes.linewidth"     : 0.8,
        "axes.grid"          : True,
        "grid.color"         : COLORS["grid"],
        "grid.linestyle"     : "-",
        "grid.linewidth"     : 0.6,
        "grid.alpha"         : 0.8,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "font.family"        : "serif",
        "font.size"          : 10,
        "axes.titlesize"     : 12,
        "axes.titleweight"   : "bold",
        "axes.labelsize"     : 10,
        "xtick.labelsize"    : 9,
        "ytick.labelsize"    : 9,
        "legend.fontsize"    : 9,
        "legend.frameon"     : True,
        "legend.framealpha"  : 0.9,
        "legend.edgecolor"   : "#CCCCCC",
        "figure.dpi"         : 150,
        "savefig.dpi"        : 200,
        "savefig.bbox"       : "tight",
        "savefig.facecolor"  : COLORS["bg"],
    })


def add_source_note(ax, note="Source: Yahoo Finance / CoinGecko | Author: Paolo Maizza"):
    ax.annotate(
        note,
        xy=(0, -0.12), xycoords="axes fraction",
        fontsize=7, color=COLORS["subtext"],
        fontstyle="italic", ha="left",
    )


def add_watermark(fig, text="INDEPENDENT RESEARCH — NOT INVESTMENT ADVICE"):
    fig.text(0.5, 0.5, text, fontsize=30,
             color="lightgrey", alpha=0.15,
             ha="center", va="center", rotation=30,
             fontweight="bold", transform=fig.transFigure)


# =============================================================================
# 9.  CHART 1 — Efficient Frontier
# =============================================================================

def plot_efficient_frontier(returns: pd.DataFrame):
    set_style()
    fig, ax = plt.subplots(figsize=(11, 7))

    # Universe without crypto
    assets_no  = ["SPY", "AGG", "GLD"]
    assets_yes = ["SPY", "AGG", "GLD", "BTC", "ETH"]

    # Monte Carlo cloud
    cloud_no  = monte_carlo_cloud(returns, assets_no,  n=3_000)
    cloud_yes = monte_carlo_cloud(returns, assets_yes, n=4_000)

    # Efficient frontiers
    front_no  = efficient_frontier_assets(returns, assets_no)
    front_yes = efficient_frontier_assets(returns, assets_yes)

    # Optimal portfolios
    w_no  = optimal_weights(returns, assets_no)
    w_yes = optimal_weights(returns, assets_yes)
    mu_no,  cov_no  = returns[assets_no].mean(),  returns[assets_no].cov()
    mu_yes, cov_yes = returns[assets_yes].mean(), returns[assets_yes].cov()

    opt_ret_no  = portfolio_return(w_no,  mu_no.values)
    opt_vol_no  = portfolio_volatility(w_no,  cov_no.values)
    opt_ret_yes = portfolio_return(w_yes, mu_yes.values)
    opt_vol_yes = portfolio_volatility(w_yes, cov_yes.values)

    # 60/40 baseline
    r_6040 = build_60_40_returns(returns)
    pt_6040 = (ann_volatility(r_6040), ann_return(r_6040))

    # Plot Monte Carlo clouds
    ax.scatter(cloud_no["vol"],  cloud_no["ret"],
               c=COLORS["frontier_no"],  alpha=0.06, s=4, zorder=1)
    ax.scatter(cloud_yes["vol"], cloud_yes["ret"],
               c=COLORS["ETH"],          alpha=0.06, s=4, zorder=1)

    # Plot frontiers
    ax.plot(front_no["vol"],  front_no["ret"],
            color=COLORS["frontier_no"],  lw=2.2, zorder=3,
            label="Frontier: SPY + AGG + Gold (no crypto)")
    ax.plot(front_yes["vol"], front_yes["ret"],
            color=COLORS["frontier_yes"], lw=2.2, zorder=3,
            label="Frontier: SPY + AGG + Gold + BTC + ETH")

    # Plot optimal points
    ax.scatter(opt_vol_no,  opt_ret_no,
               color=COLORS["frontier_no"],  s=120, zorder=5, marker="D")
    ax.scatter(opt_vol_yes, opt_ret_yes,
               color=COLORS["optimal"], s=150, zorder=5, marker="*",
               label=f"Max Sharpe (with crypto) — BTC {w_yes[3]:.1%} / ETH {w_yes[4]:.1%}")

    # Plot 60/40
    ax.scatter(*pt_6040, color=COLORS["60_40"], s=120, zorder=5, marker="s",
               label=f"60/40 Benchmark (Sharpe: {sharpe(r_6040):.2f})")

    # Individual assets
    for ticker, col in [("BTC", COLORS["BTC"]), ("ETH", COLORS["ETH"]),
                        ("GLD", COLORS["GLD"]), ("SPY", COLORS["SPY"])]:
        rv, rr = ann_volatility(returns[ticker]), ann_return(returns[ticker])
        ax.scatter(rv, rr, color=col, s=80, zorder=4, marker="o", alpha=0.85)
        ax.annotate(ticker, (rv, rr), textcoords="offset points",
                    xytext=(6, 3), fontsize=8, color=col, fontweight="bold")

    # Sharpe ratio improvement annotation
    shr_no  = (opt_ret_no  - CONFIG["rf_annual"]) / opt_vol_no
    shr_yes = (opt_ret_yes - CONFIG["rf_annual"]) / opt_vol_yes
    improvement = (shr_yes - shr_no) / shr_no * 100
    ax.annotate(
        f"Sharpe ↑ {improvement:+.1f}%\nwith crypto",
        xy=(opt_vol_yes, opt_ret_yes),
        xytext=(opt_vol_yes + 0.05, opt_ret_yes + 0.05),
        fontsize=9, color=COLORS["optimal"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["optimal"], lw=1.2),
    )

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.set_title(
        "Figure 1 — Efficient Frontier: With vs Without Digital Assets\n"
        f"Sample: Jan 2018 – Mar 2026 | Risk-Free Rate: {CONFIG['rf_annual']:.1%}",
        pad=14,
    )
    ax.legend(loc="upper left", ncol=1)
    add_source_note(ax)
    add_watermark(fig)

    path = os.path.join(CONFIG["output_dir"], "fig1_efficient_frontier.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 10.  CHART 2 — Sharpe vs Allocation (0–10%)
# =============================================================================

def plot_metrics_vs_allocation(alloc_df: pd.DataFrame):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Figure 2 — Risk-Adjusted Metrics vs Crypto Allocation (0%–10%)\n"
        "BTC/ETH mix: 71.4% / 28.6% | Benchmark: 60/40 SPY+AGG",
        fontsize=13, fontweight="bold", y=1.01,
    )

    x    = alloc_df["Crypto Total"] * 100
    opt  = alloc_df[alloc_df["Optimal"]]["Crypto Total"].values[0] * 100

    metrics = [
        ("Sharpe Ratio",    "Sharpe Ratio",    COLORS["SPY"],  "Sharpe Ratio"),
        ("Sortino Ratio",   "Sortino Ratio",   COLORS["ETH"],  "Sortino Ratio"),
        ("Calmar Ratio",    "Calmar Ratio",     COLORS["GLD"],  "Calmar Ratio"),
        ("Max Drawdown",    "Max Drawdown",    COLORS["BTC"],  "Max Drawdown"),
    ]

    for ax, (col, label, color, title) in zip(axes.flat, metrics):
        y = alloc_df[col]
        if col == "Max Drawdown":
            y = y * 100   # show as %

        ax.plot(x, y, color=color, lw=2.2, zorder=3)
        ax.fill_between(x, y, alpha=0.08, color=color)

        # 60/40 baseline (0% crypto)
        baseline = alloc_df.loc[alloc_df["Crypto Total"] == 0.0, col].values
        if len(baseline):
            bval = baseline[0] * 100 if col == "Max Drawdown" else baseline[0]
            ax.axhline(bval, color=COLORS["frontier_no"],
                       lw=1.2, ls="--", alpha=0.7, label="60/40 baseline")

        # Optimal marker
        opt_y = alloc_df.loc[alloc_df["Optimal"], col].values
        if len(opt_y):
            yv = opt_y[0] * 100 if col == "Max Drawdown" else opt_y[0]
            ax.axvline(opt, color=COLORS["optimal"], lw=1.2, ls=":",
                       alpha=0.8, label=f"Optimal: {opt:.0f}%")
            ax.scatter([opt], [yv], color=COLORS["optimal"],
                       s=80, zorder=5, marker="D")

        ax.set_xlabel("Total Crypto Allocation (%)")
        ax.set_ylabel(label + (" (%)" if col == "Max Drawdown" else ""))
        ax.set_title(title, pad=8)
        ax.legend(fontsize=8)
        add_source_note(ax, "")

    add_source_note(axes[1][1],
                    "Source: Yahoo Finance / CoinGecko | Author: Paolo Maizza")
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(CONFIG["output_dir"], "fig2_metrics_vs_allocation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 11.  CHART 3 — Rolling 12M Sharpe
# =============================================================================

def plot_rolling_sharpe(roll_df: pd.DataFrame):
    set_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    palette = [COLORS["frontier_no"], COLORS["SPY"], COLORS["BTC"]]
    for col, color in zip(roll_df.columns, palette):
        ax.plot(roll_df.index, roll_df[col], lw=1.8, color=color,
                label=col, alpha=0.9)

    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)

    # Shade positive Sharpe zones for crypto portfolio
    ax.fill_between(
        roll_df.index, 0, roll_df["60/40 + 4% Crypto"],
        where=(roll_df["60/40 + 4% Crypto"] > 0),
        alpha=0.08, color=COLORS["SPY"],
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("")
    ax.set_ylabel("Rolling 12-Month Sharpe Ratio")
    ax.set_title(
        "Figure 3 — Rolling 12-Month Sharpe Ratio: 60/40 vs 60/40+Crypto vs BTC\n"
        "Window: 252 trading days | Crypto allocation: 4% total (2.86% BTC + 1.14% ETH)",
        pad=12,
    )
    ax.legend(loc="upper right")
    add_source_note(ax)
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(CONFIG["output_dir"], "fig3_rolling_sharpe.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 12.  CHART 4 — Rebalancing Analysis
# =============================================================================

def plot_rebalancing(reb_df: pd.DataFrame):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(
        "Figure 4 — Rebalancing Frequency Analysis (4% Crypto: 2.86% BTC + 1.14% ETH)\n"
        "Following 21Shares (2024) methodology",
        fontsize=12, fontweight="bold",
    )

    metrics   = ["Sharpe Ratio", "Ann. Return", "Max Drawdown"]
    ax_titles = ["Sharpe Ratio", "Annualised Return", "Max Drawdown"]
    mult      = [1, 100, 100]
    colors    = [COLORS["SPY"], COLORS["ETH"], COLORS["BTC"], COLORS["GLD"]]

    for ax, metric, title, m in zip(axes, metrics, ax_titles, mult):
        vals   = reb_df[metric].values * m
        labels = reb_df["Rebalancing"].values
        bars   = ax.bar(labels, vals, color=colors[:len(labels)],
                        width=0.55, edgecolor="white", linewidth=0.8, zorder=3)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(vals).max() * 0.02,
                f"{val:.2f}" if metric == "Sharpe Ratio" else f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        unit = "" if metric == "Sharpe Ratio" else " (%)"
        ax.set_ylabel(title + unit)
        ax.set_title(title, pad=8)
        ax.tick_params(axis="x", labelrotation=15)
        ax.set_axisbelow(True)

    add_source_note(axes[2], "Source: Yahoo Finance / CoinGecko | Author: Paolo Maizza")
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(CONFIG["output_dir"], "fig4_rebalancing.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 13.  CHART 5 — Summary Metrics Table
# =============================================================================

def plot_metrics_table(alloc_df: pd.DataFrame):
    set_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    cols_show = [
        "Crypto Total", "Sharpe Ratio", "Sortino Ratio",
        "Calmar Ratio", "Ann. Return", "Ann. Volatility",
        "Max Drawdown", "VaR 95% (daily)",
    ]
    display = alloc_df[cols_show].copy()
    display["Crypto Total"] = display["Crypto Total"].apply(lambda x: f"{x:.0%}")
    display["Sharpe Ratio"]     = display["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")
    display["Sortino Ratio"]    = display["Sortino Ratio"].apply(lambda x: f"{x:.3f}")
    display["Calmar Ratio"]     = display["Calmar Ratio"].apply(lambda x: f"{x:.3f}")
    display["Ann. Return"]      = display["Ann. Return"].apply(lambda x: f"{x:.1%}")
    display["Ann. Volatility"]  = display["Ann. Volatility"].apply(lambda x: f"{x:.1%}")
    display["Max Drawdown"]     = display["Max Drawdown"].apply(lambda x: f"{x:.1%}")
    display["VaR 95% (daily)"]  = display["VaR 95% (daily)"].apply(lambda x: f"{x:.2%}")

    headers = [
        "Crypto\nAlloc.", "Sharpe", "Sortino", "Calmar",
        "Ann.\nReturn", "Ann.\nVol.", "Max\nDD", "VaR\n95%",
    ]

    table = ax.table(
        cellText=display.values,
        colLabels=headers,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)

    # Style header row
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#1A3A5C")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight optimal row
    opt_idx = alloc_df["Optimal"].values
    for i, is_opt in enumerate(opt_idx):
        if is_opt:
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor("#FFF3CD")
                table[(i + 1, j)].set_text_props(fontweight="bold")

    # Alternate row shading
    for i in range(len(display)):
        if not opt_idx[i]:
            shade = "#F8F8F8" if i % 2 == 0 else "#FFFFFF"
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor(shade)

    ax.set_title(
        "Table 1 — Portfolio Metrics by Crypto Allocation (0%–10%)\n"
        "BTC/ETH mix: 71.4% / 28.6% | Optimal row highlighted in yellow",
        fontsize=12, fontweight="bold", pad=14, y=0.95,
    )
    ax.annotate(
        "Source: Yahoo Finance / CoinGecko | Author: Paolo Maizza | "
        "Not investment advice",
        xy=(0.5, 0.01), xycoords="axes fraction",
        fontsize=7, color=COLORS["subtext"], ha="center", fontstyle="italic",
    )
    add_watermark(fig)

    path = os.path.join(CONFIG["output_dir"], "fig5_metrics_table.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 14.  EXPORT RESULTS TO CSV
# =============================================================================

def export_csv(alloc_df: pd.DataFrame, reb_df: pd.DataFrame):
    p1 = os.path.join(CONFIG["output_dir"], "module1_allocation_table.csv")
    p2 = os.path.join(CONFIG["output_dir"], "module1_rebalancing_table.csv")
    alloc_df.to_csv(p1, index=False, float_format="%.6f")
    reb_df.to_csv(p2,   index=False, float_format="%.6f")
    print(f"  Saved: {p1}")
    print(f"  Saved: {p2}")


# =============================================================================
# 15.  PRINT SUMMARY — console report
# =============================================================================

def print_summary(alloc_df: pd.DataFrame, reb_df: pd.DataFrame):
    separator = "─" * 70
    print(f"\n{'═'*70}")
    print("  MODULE 1 — RESULTS SUMMARY")
    print(f"  Digital Assets in Institutional Portfolios")
    print(f"{'═'*70}\n")

    opt = alloc_df[alloc_df["Optimal"]].iloc[0]
    base = alloc_df[alloc_df["Crypto Total"] == 0.0].iloc[0]

    print(f"  BENCHMARK (0% Crypto — 60/40 SPY+AGG+GLD)")
    print(f"  {'Sharpe Ratio':<22}: {base['Sharpe Ratio']:.3f}")
    print(f"  {'Sortino Ratio':<22}: {base['Sortino Ratio']:.3f}")
    print(f"  {'Ann. Return':<22}: {base['Ann. Return']:.1%}")
    print(f"  {'Max Drawdown':<22}: {base['Max Drawdown']:.1%}")
    print(f"\n{separator}")

    print(f"\n  OPTIMAL CRYPTO ALLOCATION")
    print(f"  {'Total Crypto':<22}: {opt['Crypto Total']:.0%}")
    print(f"  {'BTC Weight':<22}: {opt['BTC Weight']:.2%}")
    print(f"  {'ETH Weight':<22}: {opt['ETH Weight']:.2%}")
    print(f"  {'Sharpe Ratio':<22}: {opt['Sharpe Ratio']:.3f}  "
          f"(Δ {opt['Sharpe Ratio']-base['Sharpe Ratio']:+.3f} vs 60/40)")
    print(f"  {'Sortino Ratio':<22}: {opt['Sortino Ratio']:.3f}")
    print(f"  {'Calmar Ratio':<22}: {opt['Calmar Ratio']:.3f}")
    print(f"  {'Ann. Return':<22}: {opt['Ann. Return']:.1%}")
    print(f"  {'Max Drawdown':<22}: {opt['Max Drawdown']:.1%}")
    print(f"\n{separator}")

    sharpe_improvement = (opt["Sharpe Ratio"] - base["Sharpe Ratio"]) \
                         / abs(base["Sharpe Ratio"]) * 100
    print(f"\n  KEY FINDING (for Executive Summary)")
    print(f"  Adding {opt['Crypto Total']:.0%} BTC+ETH to a 60/40 portfolio")
    print(f"  improves the Sharpe ratio by {sharpe_improvement:+.1f}% over")
    print(f"  the full sample period (Jan 2018 – Mar 2026).")
    print(f"  Consistent with VanEck (2024) optimal range of 3–6%.")

    print(f"\n{separator}")
    print(f"\n  REBALANCING ANALYSIS (4% Crypto Allocation)")
    cols_print = ["Rebalancing", "Sharpe Ratio", "Ann. Return", "Max Drawdown"]
    reb_show = reb_df[cols_print].copy()
    reb_show["Ann. Return"]  = reb_show["Ann. Return"].apply(lambda x: f"{x:.1%}")
    reb_show["Max Drawdown"] = reb_show["Max Drawdown"].apply(lambda x: f"{x:.1%}")
    reb_show["Sharpe Ratio"] = reb_show["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")
    print(reb_show.to_string(index=False))
    print(f"\n{'═'*70}\n")


# =============================================================================
# 16.  MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  MODULE 1 — Portfolio Optimization & Efficient Frontier")
    print("  Digital Assets in Institutional Portfolios")
    print("  Author: Paolo Maizza — NUS Master in Management")
    print("=" * 70 + "\n")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    prices  = download_prices()
    returns = compute_returns(prices)

    # Align all assets — drop rows with any NaN
    common_assets = [a for a in ["BTC", "ETH", "SPY", "AGG", "GLD"] if a in returns.columns]
    returns = returns[common_assets].dropna()
    prices  = prices[common_assets].loc[returns.index]
    print(f"  Analysis window: {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Observations: {len(returns):,} daily returns\n")

    # ── 2. Allocation sweep ──────────────────────────────────────────────────
    print("Computing allocation sweep (0%–10% crypto) …")
    alloc_df = allocation_sweep(returns)

    # ── 3. Rebalancing ───────────────────────────────────────────────────────
    print("Running rebalancing analysis …")
    reb_df = rebalancing_analysis(prices, btc_w=0.04 * 0.714, eth_w=0.04 * 0.286)

    # ── 4. Charts ────────────────────────────────────────────────────────────
    print("\nGenerating charts …")
    plot_efficient_frontier(returns)
    plot_metrics_vs_allocation(alloc_df)

    roll_df = rolling_sharpe(returns)
    plot_rolling_sharpe(roll_df)
    plot_rebalancing(reb_df)
    plot_metrics_table(alloc_df)

    # ── 5. Export ────────────────────────────────────────────────────────────
    print("\nExporting CSV tables …")
    export_csv(alloc_df, reb_df)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print_summary(alloc_df, reb_df)

    print(f"All outputs saved to: ./{CONFIG['output_dir']}/")
    print("Next step: python module2_macro_regimes.py\n")


if __name__ == "__main__":
    main()
