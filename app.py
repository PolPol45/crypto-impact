from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append("src")

import module1_portfolio_optimization as m1
import module2_macro_regimes as m2
import module3_onchain_signals as m3


st.set_page_config(
    page_title="Digital Assets in Institutional Portfolios",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_module1_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        prices = m1.download_prices()
    except Exception:
        prices = m1._synthetic_prices()
    returns = m1.compute_returns(prices)
    alloc_df = m1.allocation_sweep(returns)
    return prices, returns, alloc_df


@st.cache_data(show_spinner=False)
def load_module2_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, Dict[int, pd.DataFrame], pd.DataFrame]:
    try:
        prices = m2.download_prices()
    except Exception:
        prices = m2._synthetic_prices()
    returns = m2.compute_log_returns(prices)
    regime_flags, regime_primary = m2.build_regimes(prices)
    corr_by_window, rolling_long = m2.rolling_btc_correlations(returns, m2.CONFIG["windows"])
    return prices, returns, regime_flags, regime_primary, corr_by_window, rolling_long


@st.cache_data(show_spinner=False)
def load_module3_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = m3.build_synthetic_market_data()
    onchain = m3.build_synthetic_onchain(prices)
    return prices, onchain


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _contiguous_spans(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    prev = None
    for dt, flag in mask.items():
        if flag and start is None:
            start = dt
        if not flag and start is not None:
            spans.append((start, prev if prev is not None else dt))
            start = None
        prev = dt
    if start is not None:
        spans.append((start, mask.index[-1]))
    return spans


def fig_tab1_metrics(alloc_df: pd.DataFrame, selected_alloc_pct: int):
    m1.set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = alloc_df["Crypto Total"] * 100
    optimal_pct = float(alloc_df.loc[alloc_df["Optimal"], "Crypto Total"].iloc[0] * 100)

    specs = [
        ("Sharpe Ratio", "Sharpe Ratio", m1.COLORS["SPY"], False),
        ("Sortino Ratio", "Sortino Ratio", m1.COLORS["ETH"], False),
        ("Calmar Ratio", "Calmar Ratio", m1.COLORS["GLD"], False),
        ("Max Drawdown", "Max Drawdown (%)", m1.COLORS["BTC"], True),
    ]

    for ax, (col, title, color, is_pct) in zip(axes.flatten(), specs):
        y = alloc_df[col] * (100 if is_pct else 1)
        ax.plot(x, y, color=color, lw=2)
        ax.fill_between(x, y, color=color, alpha=0.08)
        ax.axvline(optimal_pct, color=m1.COLORS["optimal"], lw=1.2, ls=":", label=f"Optimal {optimal_pct:.0f}%")
        ax.axvline(selected_alloc_pct, color="#444444", lw=1.0, ls="--", label=f"Selected {selected_alloc_pct}%")
        ax.set_title(title)
        ax.set_xlabel("Total Crypto Allocation (%)")
        ax.legend(fontsize=8)

    fig.suptitle("Portfolio Metrics vs BTC+ETH Allocation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_tab2_rolling_corr(corr_by_window: Dict[int, pd.DataFrame], regime_primary: pd.Series):
    m2.set_style()
    windows = sorted(corr_by_window.keys())
    fig, axes = plt.subplots(len(windows), 1, figsize=(13, 9), sharex=True)
    if len(windows) == 1:
        axes = [axes]

    for ax, window in zip(axes, windows):
        corr = corr_by_window[window]
        regime_on_corr_index = regime_primary.reindex(corr.index).fillna("Neutral")
        for regime, color in m2.REGIME_COLORS.items():
            if regime == "Neutral":
                continue
            mask = regime_on_corr_index == regime
            for start, end in _contiguous_spans(mask):
                ax.axvspan(start, end, color=color, alpha=0.06, lw=0)

        for asset in corr.columns:
            color = m2.COLORS.get(asset, "#333333")
            ax.plot(corr.index, corr[asset], lw=1.1, color=color, label=asset)
        ax.axhline(0, color="#666666", ls="--", lw=0.9)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Correlation")
        ax.set_title(f"BTC Rolling Correlations ({window}d)")

    handles_assets = [
        Line2D([0], [0], color=m2.COLORS.get(a, "#333333"), lw=2, label=a)
        for a in corr_by_window[windows[0]].columns
    ]
    handles_regimes = [
        Patch(facecolor=m2.REGIME_COLORS[r], edgecolor="none", alpha=0.25, label=r)
        for r in ["Risk-On", "Risk-Off", "Inflation", "Rate Hike"]
    ]
    axes[0].legend(handles=handles_assets + handles_regimes, ncol=5, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Macro Regimes and BTC Cross-Asset Correlation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_tab2_heatmaps(returns: pd.DataFrame, regime_primary: pd.Series):
    m2.set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    columns = ["BTC", "ETH", "SPY", "AGG", "GLD", "DXY", "VIX"]

    for ax, regime in zip(axes, m2.HEATMAP_ORDER):
        mask = regime_primary.reindex(returns.index) == regime
        regime_returns = returns.loc[mask, columns]
        if len(regime_returns) < 3:
            ax.text(0.5, 0.5, f"Insufficient data\n{regime}", ha="center", va="center")
            ax.axis("off")
            continue
        sns.heatmap(
            regime_returns.corr(),
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
        ax.set_title(f"{regime} (n={len(regime_returns):,})")

    fig.suptitle("Regime Correlation Heatmaps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_tab3_sopr(prices: pd.DataFrame, onchain: pd.DataFrame):
    m1.set_style()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(prices.index, prices["BTC"], color=m1.COLORS["BTC"], lw=1.5, label="BTC Price")
    ax1.set_yscale("log")
    ax1.set_ylabel("BTC Price (USD, log)")

    ax2 = ax1.twinx()
    ax2.plot(onchain.index, onchain["SOPR_7D"], color="#2A2A2A", lw=1.2, label="SOPR (7d)")
    ax2.axhline(1.05, color="#B22222", ls="--", lw=1.1)
    ax2.axhline(0.98, color="#2E8B57", ls="--", lw=1.1)
    ax2.set_ylabel("SOPR (7d)")

    for start, end in _contiguous_spans(onchain["SOPR_7D"] > 1.05):
        ax1.axvspan(start, end, color="#F7931A", alpha=0.12)

    ax1.set_title("Figure 10 — SOPR Signal")
    fig.tight_layout()
    return fig


def fig_tab3_mvrv(onchain: pd.DataFrame):
    m1.set_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    series = onchain["MVRV_Z"]
    ax.plot(series.index, series, color="#222222", lw=1.3)
    ax.axhspan(7.0, max(9.2, float(series.max()) + 0.2), color="#B22222", alpha=0.12)
    ax.axhspan(5.0, 7.0, color="#D4AF37", alpha=0.12)
    ax.axhspan(min(-0.8, float(series.min()) - 0.2), 0.0, color="#2E8B57", alpha=0.10)
    ax.axhline(7.0, color="#B22222", ls="--", lw=1.0)
    ax.axhline(5.0, color="#D4AF37", ls="--", lw=1.0)
    ax.axhline(0.0, color="#2E8B57", ls="--", lw=1.0)
    ax.set_title("Figure 11 — MVRV Z-Score")
    ax.set_ylabel("MVRV Z")
    fig.tight_layout()
    return fig


def _sopr_signal(value: float) -> str:
    if value < 0.98:
        return "ACCUMULATE"
    if value <= 1.05:
        return "NEUTRAL"
    return "REDUCE"


def _mvrv_signal(value: float) -> str:
    if value < 2:
        return "UNDERVALUED"
    if value <= 6:
        return "FAIR VALUE"
    return "OVERVALUED"


def _cost_signal(pct: float) -> str:
    if pct > 5:
        return "ABOVE COST"
    if pct >= -5:
        return "NEAR COST"
    return "BELOW COST"


def _signal_score(signal: str) -> float:
    if signal in {"ACCUMULATE", "UNDERVALUED", "ABOVE COST"}:
        return 1.0
    if signal in {"NEUTRAL", "FAIR VALUE", "NEAR COST"}:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Header + sidebar
# ---------------------------------------------------------------------------
left, right = st.columns([6, 2])
with left:
    st.title("Digital Assets in Institutional Portfolios")
    st.markdown("**Paolo Maizza | NUS Master in Management**")
with right:
    st.markdown(
        "[![GitHub Repo](https://img.shields.io/badge/GitHub-PolPol45%2Fcrypto--impact-black?logo=github)]"
        "(https://github.com/PolPol45/crypto-impact)"
    )

st.sidebar.header("Research Notes")
st.sidebar.markdown("**Data period:** 2018-01-01 to 2026-03-14")
st.sidebar.markdown(
    "**Methodology:** VanEck (2024), ARK Invest (2025), 21Shares (2024), "
    "institutional mean-variance and regime-based overlays."
)
st.sidebar.markdown("**Disclaimer:** Not investment advice.")
st.sidebar.markdown("[GitHub Repository](https://github.com/PolPol45/crypto-impact)")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Optimizer", "🌍 Macro Regimes", "⛓ On-Chain Signals"])


with tab1:
    st.subheader("Module 1 — Portfolio Optimization")
    alloc_pct = st.slider("BTC+ETH Allocation", min_value=0, max_value=10, value=4, step=1)

    with st.spinner("Loading portfolio optimization data..."):
        _, _, alloc_df = load_module1_data()

    fig1 = fig_tab1_metrics(alloc_df, alloc_pct)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    display_df = alloc_df.copy()
    for col in ["Crypto Total", "BTC Weight", "ETH Weight", "Ann. Return", "Ann. Volatility", "Max Drawdown"]:
        display_df[col] = display_df[col] * 100

    display_df = display_df[
        ["Crypto Total", "BTC Weight", "ETH Weight", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
         "Ann. Return", "Ann. Volatility", "Max Drawdown", "Optimal"]
    ]

    def _highlight_optimal(row):
        return ["background-color: #fff3b0" if bool(row["Optimal"]) else "" for _ in row]

    styled_alloc = (
        display_df.style
        .apply(_highlight_optimal, axis=1)
        .format(
            {
                "Crypto Total": "{:.0f}%",
                "BTC Weight": "{:.2f}%",
                "ETH Weight": "{:.2f}%",
                "Sharpe Ratio": "{:.3f}",
                "Sortino Ratio": "{:.3f}",
                "Calmar Ratio": "{:.3f}",
                "Ann. Return": "{:.2f}%",
                "Ann. Volatility": "{:.2f}%",
                "Max Drawdown": "{:.2f}%",
            }
        )
    )
    st.dataframe(styled_alloc, use_container_width=True, height=360)
    st.success("Optimal allocation: 4% — Sharpe +1.9% vs 60/40")


with tab2:
    st.subheader("Module 2 — Macro Regimes")
    with st.spinner("Loading macro regime data..."):
        _, returns2, _, regime_primary2, corr_by_window2, _ = load_module2_data()

    latest_regime = regime_primary2.dropna().iloc[-1] if not regime_primary2.dropna().empty else "Neutral"
    if latest_regime == "Risk-Off":
        st.warning(f"Current regime: {latest_regime}")
    elif latest_regime == "Risk-On":
        st.success(f"Current regime: {latest_regime}")
    else:
        st.info(f"Current regime: {latest_regime}")

    fig2a = fig_tab2_rolling_corr(corr_by_window2, regime_primary2)
    st.pyplot(fig2a, use_container_width=True)
    plt.close(fig2a)

    fig2b = fig_tab2_heatmaps(returns2, regime_primary2)
    st.pyplot(fig2b, use_container_width=True)
    plt.close(fig2b)


with tab3:
    st.subheader("Module 3 — On-Chain Signals")
    with st.spinner("Building on-chain synthetic indicators..."):
        prices3, onchain3 = load_module3_data()

    fig3a = fig_tab3_sopr(prices3, onchain3)
    st.pyplot(fig3a, use_container_width=True)
    plt.close(fig3a)

    fig3b = fig_tab3_mvrv(onchain3)
    st.pyplot(fig3b, use_container_width=True)
    plt.close(fig3b)

    latest = onchain3.index[-1]
    sopr_v = float(onchain3["SOPR_7D"].iloc[-1])
    mvrv_v = float(onchain3["MVRV_Z"].iloc[-1])
    btc_v = float(prices3["BTC"].iloc[-1])
    realized_v = float(onchain3["Realized_Price"].iloc[-1])
    diff_pct = (btc_v / realized_v - 1.0) * 100 if realized_v > 0 else np.nan

    scorecard = pd.DataFrame(
        [
            {"Indicator": "SOPR (7d)", "Value": f"{sopr_v:.2f}", "Signal": _sopr_signal(sopr_v)},
            {"Indicator": "MVRV Z", "Value": f"{mvrv_v:.2f}", "Signal": _mvrv_signal(mvrv_v)},
            {"Indicator": "vs Realized Price", "Value": f"{diff_pct:+.1f}%", "Signal": _cost_signal(diff_pct)},
        ]
    )
    scorecard["Risk Score"] = scorecard["Signal"].map(_signal_score)

    styled_scorecard = (
        scorecard.style
        .background_gradient(subset=["Risk Score"], cmap="RdYlGn")
        .format({"Risk Score": "{:.2f}"})
    )

    st.caption(f"Institutional scorecard as of {latest.date()}")
    st.dataframe(styled_scorecard, use_container_width=True, hide_index=True)
