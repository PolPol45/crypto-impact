# Digital Assets in Institutional Portfolios
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

### A Multi-Dimensional Investment Research Framework

**Author:** Paolo Maizza | NUS Master in Management  
**Contact:** paolo.maizza2@gmail.com | [LinkedIn](https://linkedin.com/in/paolomaizza)

## Overview
Quantitative research framework analyzing the role of digital assets (BTC/ETH) 
in institutional portfolio construction. Follows methodology from VanEck (2024), 
ARK Invest (2025), and 21Shares (2024).

## Modules
| Module | Topic | Status |
|--------|-------|--------|
| 1 | Portfolio Optimization & Efficient Frontier | ✅ Complete |
| 2 | Macro Regime Analysis & Correlations | ✅ Complete |
| 3 | On-Chain Risk Indicators | ✅ Complete |

## Key Findings

### Module 1 — Portfolio Optimization
- Optimal BTC+ETH allocation: **4%** improves Sharpe by **+1.9%** vs 60/40 benchmark
- Best rebalancing frequency: **Quarterly** (consistent with 21Shares 2024)
- Sortino ratio peaks at 4% allocation — confirming asymmetric return profile

### Module 2 — Macro Regime Analysis
- BTC/SPY correlation is **regime-dependent**, not structural
- Diversification benefit strongest during: **Risk-On** and **Inflation Shock** regimes
- During **Risk-Off** (VIX > 25): BTC correlation with SPY rises sharply — hedge properties weaken

### Module 3 — On-Chain Risk Signals
- SOPR-filtered strategy improves Sharpe vs static allocation
- MVRV Z-Score filter significantly reduces maximum drawdown
- On-chain signals provide actionable risk overlay for institutional portfolio management

## Setup
```bash
pip install yfinance scipy numpy pandas matplotlib seaborn
python module1_portfolio_optimization.py
```

## Disclaimer
This project is for educational and research purposes only. 
Not investment advice.
