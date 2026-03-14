# Digital Assets in Institutional Portfolios
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

### A Multi-Dimensional Investment Research Framework

**Author:** Paolo Maizza | NUS Master in Management  
**Contact:** paolo.maizza2@gmail.com | [LinkedIn](https://linkedin.com/in/paolomaizza)  
**Repository:** [github.com/PolPol45/crypto-impact](https://github.com/PolPol45/crypto-impact)

---

## Executive Overview
This project quantifies the role of BTC/ETH in institutional portfolios across three dimensions:
1. **Optimal allocation** (mean-variance + risk-adjusted metrics)
2. **Macro regime behavior** (dynamic correlation structure)
3. **On-chain overlay** (SOPR, MVRV, Realized Price for risk control)

Primary deliverables:
- Reproducible quantitative analytics engine (Python)
- Interactive Streamlit dashboard (6 pages)
- Institutional 2-page Executive Summary PDF (share-ready)

---

## Documentation Standard
The project documentation follows an institutional research-note structure:
- **BLUF (Bottom Line Up Front)** for immediate decision relevance
- **Modular sections**: Objective, Methodology, Findings, Implications, Limitations
- **Reproducibility**: end-to-end commands, deterministic outputs, offline synthetic fallback
- **Clear separation** between quantitative engine (`src/`) and presentation layers (`app.py`, `report/`)

---

## Technical Architecture

### 1) Core Analytics (`src/`)
- `module1_portfolio_optimization.py`
  - Mean-variance optimization on a 60/40 benchmark
  - Crypto allocation sweep (0% to 10%)
  - Metrics: Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR
- `module2_macro_regimes.py`
  - Regime classification: Risk-On, Risk-Off, Inflation, Rate Hike
  - Rolling BTC cross-asset correlations (30/90/180d)
  - Synthetic fallback when `yfinance` is unavailable
- `module3_onchain_signals.py`
  - Calibrated synthetic on-chain indicators (SOPR, MVRV Z, Realized Price)
  - Backtests: static exposure vs SOPR/MVRV filtered overlays

### 2) Dashboard (`app.py`)
Sidebar navigation with 6 pages:
1. `🏠 Home`
2. `📊 Portfolio Optimizer`
3. `🌍 Macro Regimes`
4. `⛓ On-Chain Signals`
5. `🎯 My Portfolio`
6. `📄 Technical Summary`

### 3) Reporting (`report/`)
- `executive_summary.py`: generates the institutional 2-page A4 PDF
- `Executive_Summary_Paolo_Maizza_Digital_Assets.pdf`: final shareable output

---

## Repository Structure

```text
crypto-impact/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── module1_portfolio_optimization.py
│   ├── module2_macro_regimes.py
│   └── module3_onchain_signals.py
├── outputs/
│   ├── module1/
│   ├── module2/
│   └── module3/
├── report/
│   ├── executive_summary.py
│   └── Executive_Summary_Paolo_Maizza_Digital_Assets.pdf
├── docs/
└── files/
```

---

## Modules and Status

| Module | Topic | Status | Script |
|--------|-------|--------|--------|
| 1 | Portfolio Optimization & Efficient Frontier | ✅ Complete | `src/module1_portfolio_optimization.py` |
| 2 | Macro Regime Analysis & Correlations | ✅ Complete | `src/module2_macro_regimes.py` |
| 3 | On-Chain Risk Indicators | ✅ Complete | `src/module3_onchain_signals.py` |

---

## Key Findings (Summary)

### Module 1 — Portfolio Optimization
- Sharpe-optimal BTC+ETH allocation: **4%**
- Sharpe improvement vs 60/40 benchmark: **+1.9%**
- Most efficient rebalancing frequency: **Quarterly**

### Module 2 — Macro Regime Analysis
- BTC/SPY correlation is **regime-dependent**, not structural
- Diversification benefit is strongest in **Risk-On** and **Inflation Shock** windows
- In **Risk-Off (VIX > 25)** BTC correlation with SPY tends to increase materially

### Module 3 — On-Chain Signals
- SOPR filter improves risk-adjusted profile versus static exposure
- MVRV Z-score is effective as a top-cycle risk-reduction signal
- On-chain metrics are used as a quantitative overlay, not a replacement for strategic allocation

---

## Installation (All Commands in Sequence)

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Full End-to-End Run

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate

python src/module1_portfolio_optimization.py
python src/module2_macro_regimes.py
python src/module3_onchain_signals.py
```

Check outputs:

```bash
ls -la outputs/module1
ls -la outputs/module2
ls -la outputs/module3
```

---

## Launch Streamlit Dashboard

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate
streamlit run app.py
```

Live dashboard (if deployed):  
`https://crypto-impact-4mzxlfm56fjvpprtjsezq7.streamlit.app`

---

## Generate Executive Summary PDF

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate
python report/executive_summary.py
```

Output:
- `report/Executive_Summary_Paolo_Maizza_Digital_Assets.pdf`

---

## Quick Technical Tests

### 1) Syntax checks
```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate
python -m py_compile app.py
python -m py_compile src/module1_portfolio_optimization.py
python -m py_compile src/module2_macro_regimes.py
python -m py_compile src/module3_onchain_signals.py
python -m py_compile report/executive_summary.py
```

### 2) Smoke checks for key outputs
```bash
test -f outputs/module1/fig1_efficient_frontier.png && echo "OK module1"
test -f outputs/module2/module2_rolling_correlations_regimes.png && echo "OK module2"
test -f outputs/module3/fig10_sopr_signal.png && echo "OK module3"
test -f report/Executive_Summary_Paolo_Maizza_Digital_Assets.pdf && echo "OK report"
```

---

## Data Assumptions and Limitations
- Main sample window: **2018–2026**
- Primary market source: **Yahoo Finance**
- Offline mode: calibrated synthetic GBM fallback
- Current on-chain feed: calibrated synthetic signals (not live Glassnode API)
- Historical performance is not indicative of future results

---

## Disclaimer
This project is for educational and quantitative research purposes only.  
**It does not constitute financial advice, investment recommendation, or solicitation to buy/sell financial instruments.**
