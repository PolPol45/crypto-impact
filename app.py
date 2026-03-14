from __future__ import annotations

import sys
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


@st.cache_data(show_spinner=False)
def load_personal_returns() -> pd.DataFrame:
    try:
        prices = m1.download_prices()
    except Exception:
        prices = m1._synthetic_prices()
    return m1.compute_returns(prices)


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


def _portfolio_returns_from_weights(returns: pd.DataFrame, weights_pct: Dict[str, float]) -> pd.Series:
    w_eq = weights_pct["equities"] / 100.0
    w_bd = weights_pct["bonds"] / 100.0
    w_gd = weights_pct["gold"] / 100.0
    w_cash = weights_pct["cash"] / 100.0
    w_btc = weights_pct["btc"] / 100.0
    w_eth = weights_pct["eth"] / 100.0
    w_oth = weights_pct["other"] / 100.0

    cash_series = pd.Series(0.0, index=returns.index)
    # Other crypto proxied with ETH returns as high-beta digital asset proxy.
    return (
        w_eq * returns["SPY"]
        + w_bd * returns["AGG"]
        + w_gd * returns["GLD"]
        + w_cash * cash_series
        + w_btc * returns["BTC"]
        + w_eth * returns["ETH"]
        + w_oth * returns["ETH"]
    )


def _compute_metrics(ret_series: pd.Series) -> Dict[str, float]:
    return {
        "Sharpe": m1.sharpe(ret_series),
        "Sortino": m1.sortino(ret_series),
        "Max Drawdown": m1.max_drawdown(ret_series),
        "Ann. Return": m1.ann_return(ret_series),
        "Volatility": m1.ann_volatility(ret_series),
    }


def _render_existing_portfolio_optimizer_page() -> None:
    st.subheader("Module 1 — Portfolio Optimization")
    st.markdown("### What is Portfolio Optimization?")
    st.markdown(
        """
Mean-variance optimization (Markowitz, 1952) identifies the portfolio
that maximizes return for a given level of risk. The **Sharpe ratio** —
excess return per unit of volatility — is the standard metric used by
institutional portfolio managers to compare allocations.

**The benchmark** used here is a classic 60/40 portfolio:  
60% S&P 500 (SPY) + 40% US Aggregate Bonds (AGG).  
This is the standard starting point for most institutional mandates.
"""
    )
    st.info(
        """
**How to read this page:**  
Use the slider to change the BTC+ETH allocation (0%–10%).  
Watch how Sharpe, Sortino, and Max Drawdown respond.  
The optimal point is where the Sharpe ratio peaks — typically around 4%.
"""
    )
    alloc_pct = st.slider("BTC+ETH Allocation", min_value=0, max_value=10, value=4, step=1)

    with st.spinner("Loading portfolio optimization data..."):
        _, _, alloc_df = load_module1_data()

    fig1 = fig_tab1_metrics(alloc_df, alloc_pct)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    with st.expander("📖 What do these metrics mean?"):
        st.markdown(
            """
| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Sharpe Ratio** | Return per unit of total volatility | Higher = better risk-adjusted return. Industry standard for comparing portfolios. |
| **Sortino Ratio** | Return per unit of *downside* volatility only | More relevant for crypto — penalises losses, not upside swings. |
| **Calmar Ratio** | Annual return / Maximum Drawdown | Measures recovery efficiency. Preferred by hedge funds. |
| **Max Drawdown** | Largest peak-to-trough loss | Critical for institutional mandates with drawdown limits. |
| **VaR 95%** | Daily loss not exceeded 95% of the time | Standard risk measure for regulatory reporting (Basel III). |
"""
        )

    display_df = alloc_df.copy()
    for col in ["Crypto Total", "BTC Weight", "ETH Weight", "Ann. Return", "Ann. Volatility", "Max Drawdown"]:
        display_df[col] = display_df[col] * 100

    display_df = display_df[
        [
            "Crypto Total",
            "BTC Weight",
            "ETH Weight",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Ann. Return",
            "Ann. Volatility",
            "Max Drawdown",
            "Optimal",
        ]
    ]

    def _highlight_optimal(row):
        return ["background-color: #fff3b0" if bool(row["Optimal"]) else "" for _ in row]

    styled_alloc = (
        display_df.style.apply(_highlight_optimal, axis=1).format(
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
    st.markdown("### 📌 What does this mean for your portfolio?")
    st.markdown(
        """
- Adding **1–4% BTC+ETH** to a 60/40 portfolio improves the Sharpe ratio
  without significantly increasing drawdown risk.
- Beyond **5–6%**, volatility increases faster than return —
  the risk-return tradeoff deteriorates.
- **Quarterly rebalancing** captures the rebalancing premium
  and maintains the target allocation efficiently.
- The Sortino ratio remains elevated even at higher allocations —
  suggesting BTC's volatility is predominantly *upside* volatility,
  not downside risk.
"""
    )


def _render_existing_macro_regimes_page() -> None:
    st.subheader("Module 2 — Macro Regimes")
    st.markdown("### Why do macro regimes matter for crypto?")
    st.markdown(
        """
Bitcoin's correlation with traditional assets is **not constant** —
it shifts dramatically depending on the macroeconomic environment.  
A portfolio manager who assumes a fixed correlation will
**misprice the diversification benefit** of crypto.

This analysis classifies every trading day from 2018 to 2026
into one of 4 regimes, then measures how BTC behaves in each.
"""
    )
    st.info(
        """
**How to read this page:**  
Look at the correlation heatmaps — **blue = low correlation**
(good for diversification), **red = high correlation** (poor diversification).  
The rolling correlation chart shows how BTC/SPY correlation
has evolved over time.
"""
    )
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

    with st.expander("📖 What do the 4 regimes mean?"):
        st.markdown(
            """
| Regime | Definition | BTC Behavior | Portfolio Impact |
|--------|-----------|--------------|-----------------|
| **Risk-On** | VIX < 20, SPY trending up | Low correlation with bonds and gold. Behaves as growth asset. | Good diversifier — add to position |
| **Risk-Off** | VIX > 25 (fear spike) | Correlation with SPY rises sharply. "Everything sells off." | Hedge properties weaken — reduce exposure |
| **Inflation Shock** | CPI > 5% (2021–2022) | Gold-like behavior. Positive real-asset narrative. | Strong diversifier — supports 4–6% allocation |
| **Rate Hike Cycle** | Fed tightening (2022–2023) | High sensitivity to rate changes. Behaves like long-duration asset. | Caution — reduce to 1–2% |
"""
        )

    st.markdown("### 📌 What does this mean for your portfolio?")
    st.markdown(
        """
- **Do not assume crypto always diversifies.** The benefit is
  regime-dependent — strongest during Risk-On and Inflation,
  weakest during Risk-Off and Rate Hike cycles.
- **Monitor VIX as a signal:** when VIX crosses 25, BTC correlation
  with equities historically rises — reducing the diversification benefit.
- **The post-ETF period (2024+)** shows declining BTC/SPY correlation
  as institutional flows mature — a structural shift worth monitoring.
- **Practical implication:** a dynamic allocation strategy
  (increase BTC in Risk-On/Inflation, reduce in Risk-Off/Rate Hike)
  outperforms a static allocation over full cycles.
"""
    )


def _render_existing_onchain_page() -> None:
    st.subheader("Module 3 — On-Chain Signals")
    st.markdown("### What are on-chain signals?")
    st.markdown(
        """
On-chain analysis examines **Bitcoin blockchain transaction data**
to infer the behavior of market participants.  
Unlike price-based indicators, on-chain signals reveal
whether holders are selling at a profit or loss —
a direct measure of market sentiment and cycle positioning.

This is the **unique differentiator** of this framework:  
no institutional paper (VanEck, ARK, Grayscale)
uses on-chain signals as a quantitative portfolio risk overlay.
"""
    )
    st.info(
        """
**How to read this page:**  
- **SOPR chart:** orange bands = euphoria periods (SOPR > 1.05)
  when the model reduces BTC allocation to 1%.  
- **MVRV Z-Score:** red zone = extreme overvaluation (reduce to 0%),
  green zone = undervaluation (accumulate).  
- **Scorecard:** your real-time positioning signal.
"""
    )
    with st.spinner("Building on-chain synthetic indicators..."):
        prices3, onchain3 = load_module3_data()

    fig3a = fig_tab3_sopr(prices3, onchain3)
    st.pyplot(fig3a, use_container_width=True)
    plt.close(fig3a)

    fig3b = fig_tab3_mvrv(onchain3)
    st.pyplot(fig3b, use_container_width=True)
    plt.close(fig3b)

    with st.expander("📖 What do these indicators mean?"):
        st.markdown(
            """
| Indicator | What it measures | Signal logic | Portfolio action |
|-----------|-----------------|--------------|-----------------|
| **SOPR** (Spent Output Profit Ratio) | Are Bitcoin holders selling at a profit or loss? | > 1.05 = euphoria (sellers profit) → **reduce to 1% BTC** | Trim allocation at cycle tops |
| **MVRV Z-Score** | How far is market cap above realized (cost basis) cap? | > 7.0 = extreme overvaluation → **reduce to 0% BTC** | Exit at historic bubble levels |
| **Realized Price** | Average purchase price of all BTC in circulation | BTC above realized price = market in profit | Use as dynamic support/resistance |
"""
        )

    st.markdown("### 📌 What does this mean for your portfolio?")
    st.markdown(
        """
- **On-chain signals act as a risk overlay** — they do not replace
  the strategic allocation from Module 1, but improve its timing.
- The **SOPR filter** historically reduces drawdown during cycle tops
  while preserving most of the upside by reducing, not eliminating, exposure.
- The **MVRV Z-Score** has historically peaked above 7 at every major
  BTC top (2017, 2021) — making it a useful extreme-overvaluation warning.
- **Practical implementation:** check SOPR and MVRV monthly.
  If both signal red, reduce BTC to 1–2%. If both signal green,
  consider increasing toward the 4–5% optimal range.
- **Limitation:** these signals are based on synthetic calibrated data.
  For live signals, use Glassnode or LookIntoBitcoin.com.
"""
    )

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

    styled_scorecard = scorecard.style.background_gradient(subset=["Risk Score"], cmap="RdYlGn").format({"Risk Score": "{:.2f}"})

    st.caption(f"Institutional scorecard as of {latest.date()}")
    st.dataframe(styled_scorecard, use_container_width=True, hide_index=True)


def _render_home_page() -> None:
    st.title("Digital Assets in Institutional Portfolios")
    st.markdown("### A Multi-Dimensional Investment Research Framework")
    st.markdown("**Paolo Maizza | NUS Master in Management | March 2026**")
    st.link_button("Open GitHub Repository", "https://github.com/PolPol45/crypto-impact")

    st.markdown("---")
    st.subheader("What this dashboard does")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 📊 Portfolio Optimization")
        st.write(
            "Quantifies the risk-adjusted benefit of adding BTC/ETH "
            "to a standard 60/40 institutional portfolio. "
            "Computes Sharpe, Sortino, Calmar ratios across "
            "allocations from 0% to 10%."
        )

    with c2:
        st.markdown("#### 🌍 Macro Regime Analysis")
        st.write(
            "Evaluates how BTC correlations with equities, bonds "
            "and gold shift across 4 macro regimes: Risk-On, "
            "Risk-Off, Inflation Shock, and Rate Hike cycles. "
            "Based on 2018–2026 daily data."
        )

    with c3:
        st.markdown("#### ⛓ On-Chain Signals")
        st.write(
            "Tests whether SOPR and MVRV Z-Score on-chain "
            "indicators can improve portfolio risk management "
            "by dynamically reducing BTC exposure at cycle tops."
        )

    with st.expander("📖 Read before you start"):
        st.markdown("**Step 1 — Start with Portfolio Optimizer:** use the slider to find your optimal BTC/ETH allocation")
        st.markdown("**Step 2 — Check Macro Regimes:** understand when crypto diversification works")
        st.markdown("**Step 3 — Monitor On-Chain Signals:** use the scorecard as a risk overlay")
        st.markdown("**Step 4 — Set My Portfolio:** enter your actual holdings and see personalized analysis")
        st.markdown("**Step 5 — Read Technical Summary:** understand the methodology and limitations")

    st.subheader("Key Findings")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal Allocation", "4%", "+1.9% Sharpe")
    k2.metric("Best Rebalancing", "Quarterly", "21Shares 2024")
    k3.metric("Regime Dependency", "Risk-On", "Best diversification")
    k4.metric("On-Chain Edge", "SOPR Filter", "Reduces drawdown")

    st.info(
        "📌 Disclaimer: This dashboard is for educational and research purposes only. "
        "Not investment advice."
    )


def _render_my_portfolio_page() -> None:
    st.subheader("Enter your current portfolio")

    with st.form("portfolio_form"):
        st.subheader("Traditional Assets")
        col1, col2 = st.columns(2)
        with col1:
            equities_pct = st.slider("Equities (stocks, ETFs)", 0, 100, 60)
            bonds_pct = st.slider("Bonds / Fixed Income", 0, 100, 40)
        with col2:
            gold_pct = st.slider("Gold / Commodities", 0, 100, 0)
            cash_pct = st.slider("Cash / Money Market", 0, 100, 0)

        st.subheader("Digital Assets")
        btc_pct = st.slider("Bitcoin (BTC)", 0, 20, 0)
        eth_pct = st.slider("Ethereum (ETH)", 0, 20, 0)
        other_crypto_pct = st.slider("Other Crypto", 0, 20, 0)

        st.subheader("Portfolio Details")
        portfolio_value = st.number_input("Total Portfolio Value (USD)", min_value=1000, value=100000, step=1000)
        risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])

        submit = st.form_submit_button("Analyze My Portfolio")

    if not submit:
        return

    total_alloc = equities_pct + bonds_pct + gold_pct + cash_pct + btc_pct + eth_pct + other_crypto_pct
    if total_alloc != 100:
        st.error(f"Allocation must sum to 100%. Current total: {total_alloc}%")
        return

    st.markdown(f"**Selected risk profile:** {risk_profile}")

    with st.spinner("Computing personalized analytics..."):
        returns = load_personal_returns()

    user_weights = {
        "equities": float(equities_pct),
        "bonds": float(bonds_pct),
        "gold": float(gold_pct),
        "cash": float(cash_pct),
        "btc": float(btc_pct),
        "eth": float(eth_pct),
        "other": float(other_crypto_pct),
    }

    # Suggested optimized allocation: add 4% BTC funded from equities.
    shift = min(4.0, user_weights["equities"])
    opt_weights = user_weights.copy()
    opt_weights["equities"] -= shift
    opt_weights["btc"] += shift

    ret_user = _portfolio_returns_from_weights(returns, user_weights)
    ret_opt = _portfolio_returns_from_weights(returns, opt_weights)
    ret_bmk = 0.60 * returns["SPY"] + 0.40 * returns["AGG"]

    met_user = _compute_metrics(ret_user)
    met_opt = _compute_metrics(ret_opt)
    met_bmk = _compute_metrics(ret_bmk)

    st.subheader("Your Portfolio Analysis")

    # 1) Pie charts: current vs suggested
    m1.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Equities", "Bonds", "Gold", "Cash", "BTC", "ETH", "Other Crypto"]
    colors = [
        m1.COLORS["SPY"],
        m1.COLORS["AGG"],
        m1.COLORS["GLD"],
        "#BFBFBF",
        m1.COLORS["BTC"],
        m1.COLORS["ETH"],
        "#8A63D2",
    ]

    user_vals = [equities_pct, bonds_pct, gold_pct, cash_pct, btc_pct, eth_pct, other_crypto_pct]
    opt_vals = [
        opt_weights["equities"],
        opt_weights["bonds"],
        opt_weights["gold"],
        opt_weights["cash"],
        opt_weights["btc"],
        opt_weights["eth"],
        opt_weights["other"],
    ]

    axes[0].pie(user_vals, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)
    axes[0].set_title("Current Allocation")
    axes[1].pie(opt_vals, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)
    axes[1].set_title("Suggested Optimized Allocation")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 2) Metric cards vs benchmark
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Sharpe Ratio", f"{met_user['Sharpe']:.3f}", f"{(met_user['Sharpe'] - met_bmk['Sharpe']):+.3f} vs 60/40")
    mcol2.metric(
        "Max Drawdown",
        f"{met_user['Max Drawdown'] * 100:.1f}%",
        f"{(met_user['Max Drawdown'] - met_bmk['Max Drawdown']) * 100:+.1f}pp vs 60/40",
    )
    mcol3.metric(
        "Estimated Annual Return",
        f"{met_user['Ann. Return'] * 100:.1f}%",
        f"{(met_user['Ann. Return'] - met_bmk['Ann. Return']) * 100:+.1f}pp vs 60/40",
    )

    # 3) Comparison table
    table = pd.DataFrame(
        {
            "Metric": ["Sharpe", "Sortino", "Max Drawdown", "Ann. Return", "Volatility"],
            "Your Portfolio": [
                met_user["Sharpe"],
                met_user["Sortino"],
                met_user["Max Drawdown"],
                met_user["Ann. Return"],
                met_user["Volatility"],
            ],
            "60/40 Benchmark": [
                met_bmk["Sharpe"],
                met_bmk["Sortino"],
                met_bmk["Max Drawdown"],
                met_bmk["Ann. Return"],
                met_bmk["Volatility"],
            ],
            "Optimized (4% BTC)": [
                met_opt["Sharpe"],
                met_opt["Sortino"],
                met_opt["Max Drawdown"],
                met_opt["Ann. Return"],
                met_opt["Volatility"],
            ],
        }
    )

    def _fmt_metric(metric: str, value: float) -> str:
        if metric in {"Max Drawdown", "Ann. Return", "Volatility"}:
            return f"{value * 100:.2f}%"
        return f"{value:.3f}"

    table_display = table.copy()
    for col in ["Your Portfolio", "60/40 Benchmark", "Optimized (4% BTC)"]:
        table_display[col] = [
            _fmt_metric(m, v) for m, v in zip(table_display["Metric"], table_display[col])
        ]
    st.dataframe(table_display, use_container_width=True, hide_index=True)

    # 4) Personalized recommendation
    crypto_total = btc_pct + eth_pct + other_crypto_pct
    if crypto_total == 0:
        st.warning(
            "⚠️ Your portfolio has no crypto exposure. Research suggests a 1–4% BTC allocation "
            "could improve your Sharpe ratio by up to +1.9% based on 2018–2026 data."
        )
    elif 0 < crypto_total <= 4:
        st.success(
            "✅ Your crypto allocation is within the optimal range (1–4%). "
            "Consider quarterly rebalancing."
        )
    elif 4 < crypto_total <= 8:
        st.info(
            "ℹ️ Your crypto allocation exceeds the Sharpe-optimal level. Higher allocations "
            "increase volatility without proportional return benefit."
        )
    else:
        st.error(
            "🔴 Your crypto allocation is significantly above optimal. Consider reducing to 3–5% "
            "to improve risk-adjusted returns."
        )

    # 5) Dollar breakdown table
    st.subheader("Your Allocation in Dollar Terms")
    rows = []
    asset_map = [
        ("Equities", "equities"),
        ("Bonds / Fixed Income", "bonds"),
        ("Gold / Commodities", "gold"),
        ("Cash / Money Market", "cash"),
        ("Bitcoin (BTC)", "btc"),
        ("Ethereum (ETH)", "eth"),
        ("Other Crypto", "other"),
    ]

    for label, key in asset_map:
        cur_w = user_weights[key]
        opt_w = opt_weights[key]
        cur_v = portfolio_value * cur_w / 100.0
        opt_v = portfolio_value * opt_w / 100.0
        rows.append(
            {
                "Asset": label,
                "Weight": f"{cur_w:.1f}%",
                "Value (USD)": f"${cur_v:,.0f}",
                "Suggested Rebalance": f"${(opt_v - cur_v):+,.0f}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_technical_summary_page() -> None:
    st.markdown(
        """
# Technical Executive Summary

**Digital Assets in Institutional Portfolios: A Multi-Dimensional Framework**  
Paolo Maizza | NUS Master in Management | March 2026

---

## Research Objective
This framework quantifies the role of digital assets (BTC/ETH) in institutional 
portfolio construction across three analytical dimensions: optimal allocation, 
macro regime behavior, and on-chain risk management.

---

## Module 1 — Portfolio Optimization

**Methodology:** Mean-variance optimization (Markowitz) applied to a 60% SPY / 
40% AGG benchmark portfolio. 11 portfolio configurations tested (0%–10% crypto 
in 1% increments), with BTC/ETH split following the VanEck (2024) optimal ratio 
of 71.4% / 28.6%. Metrics computed: Sharpe ratio, Sortino ratio, Calmar ratio, 
maximum drawdown, VaR 95%, CVaR 95%. Rebalancing frequency tested: monthly, 
quarterly, annual, buy-and-hold.

**Basis:** VanEck (2024) "Optimal Crypto Allocation for Portfolios"; 
ARK Invest (2025) "Measuring Bitcoin's Risk and Reward"; 
21Shares (2024) "Cryptoassets in a Diversified Portfolio"

**Key Finding:** A 4% BTC+ETH allocation (2.86% BTC / 1.14% ETH) maximizes 
the Sharpe ratio, improving it by +1.9% relative to the 60/40 benchmark over 
the 2018–2026 sample period. Quarterly rebalancing yields the best 
risk-adjusted outcome, consistent with 21Shares (2024).

---

## Module 2 — Macro Regime Analysis

**Methodology:** Four macro regimes defined with rule-based classification: 
Risk-On (VIX < 20 and SPY above 50-day SMA), Risk-Off (VIX > 25), 
Inflation Shock (March 2021 – June 2022, US CPI > 5%), Rate Hike Cycle 
(March 2022 – July 2023, Fed tightening). Rolling Pearson correlations 
computed at 30, 90, and 180-day windows for BTC vs SPY, AGG, GLD, DXY, VIX.

**Basis:** Fidelity Digital Assets (2025); Goldman Sachs Digital Asset Report 
(2026); CoinShares Research (2024)

**Key Finding:** BTC diversification benefits are regime-dependent, not 
structural. Correlation with SPY is lowest during Risk-On and Inflation regimes 
(genuine diversification), and rises sharply during Risk-Off periods 
(VIX > 25), undermining the hedge hypothesis under stress conditions.

---

## Module 3 — On-Chain Risk Signals

**Methodology:** Three synthetic on-chain indicators calibrated to historical 
BTC cycles: SOPR (Spent Output Profit Ratio, mean-reverting around 1.0 with 
cycle tops/bottoms); MVRV Z-Score (market value vs realized value, range 
-0.5 to 9.0); Realized Price (180-day EMA × 0.72 as aggregate cost basis). 
Three portfolio strategies backtested: A) Static 4% BTC; B) SOPR-filtered 
(reduce to 1% when SOPR 7d avg > 1.05); C) MVRV-filtered (reduce to 0% 
when MVRV Z > 7.0). Holdings-based backtest with quarterly + signal-triggered 
rebalancing.

**Basis:** CoinShares Research on-chain methodology; Glassnode (2024) 
"On-Chain Market Intelligence"

**Key Finding:** On-chain filters provide a meaningful risk management overlay. 
The SOPR-filtered strategy improves Sharpe ratio vs the static allocation, 
while the MVRV filter significantly reduces maximum drawdown during cycle tops.

---

## Data Sources & Sample Period
| Source | Data | Period |
|--------|------|--------|
| Yahoo Finance | BTC, ETH, SPY, AGG, GLD, DXY, VIX | Jan 2018 – Mar 2026 |
| Synthetic GBM | Offline fallback (calibrated to real stats) | Jan 2018 – Mar 2026 |
| On-chain (synthetic) | SOPR, MVRV Z-Score, Realized Price | Jan 2018 – Mar 2026 |

---

## Limitations
- Synthetic data used for offline demonstration; live results require 
  yfinance + real market data
- Markowitz optimization assumes normally distributed returns; crypto 
  exhibits fat tails and skewness
- On-chain signals are simulated, not sourced from live Glassnode data
- Results are sensitive to the sample period; the post-ETF regime 
  (Jan 2024 onwards) may alter optimal allocations
- Past performance does not guarantee future results

---

## References
- VanEck (2024). "Optimal Crypto Allocation for Portfolios"
- ARK Invest (2025). "Measuring Bitcoin's Risk and Reward"  
- 21Shares (2024). "Cryptoassets in a Diversified Portfolio"
- Fidelity Digital Assets (2025). "The Case for Bitcoin"
- Grayscale Research (2024). "The Role of Crypto in a Portfolio"
- Goldman Sachs (2026). "Digital Assets: From Experiment to Asset Class"
- CoinShares Research (2024). "Digital Asset Fund Flows"

---

*This dashboard is for educational and research purposes only.  
Not investment advice. Independent research — Paolo Maizza, March 2026.*

---
"""
    )

    st.download_button(
        label="📥 Download Technical Summary (PDF coming soon)",
        data="See report/Digital_Assets_Institutional_Portfolios_Paolo_Maizza.pdf",
        file_name="technical_summary_paolo_maizza.txt",
        mime="text/plain",
    )


# ---------------------------------------------------------------------------
# Shared header + sidebar nav
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

pages = [
    "🏠 Home",
    "📊 Portfolio Optimizer",
    "🌍 Macro Regimes",
    "⛓ On-Chain Signals",
    "🎯 My Portfolio",
    "📄 Technical Summary",
]
page = st.sidebar.radio("Navigation", pages, index=0)


if page == "🏠 Home":
    _render_home_page()
elif page == "📊 Portfolio Optimizer":
    _render_existing_portfolio_optimizer_page()
elif page == "🌍 Macro Regimes":
    _render_existing_macro_regimes_page()
elif page == "⛓ On-Chain Signals":
    _render_existing_onchain_page()
elif page == "🎯 My Portfolio":
    _render_my_portfolio_page()
else:
    _render_technical_summary_page()
