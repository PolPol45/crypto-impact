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
Questo progetto quantifica il ruolo di BTC/ETH in portafogli istituzionali su tre assi:
1. **Allocazione ottimale** (mean-variance + metriche risk-adjusted)
2. **Comportamento per regime macro** (correlazioni dinamiche)
3. **Overlay on-chain** (SOPR, MVRV, Realized Price per gestione del rischio)

Output principali:
- Analisi quantitativa riproducibile (Python)
- Dashboard interattiva Streamlit (6 pagine)
- Executive Summary PDF istituzionale (2 pagine, share-ready)

---

## Standard Tecnico-Documentale Adottato
La struttura del progetto e del reporting segue una logica “institutional research note” allineata alle best practice:
- **BLUF (Bottom Line Up Front)** per conclusioni operative immediate
- **Sezioni modulari**: Objective, Methodology, Findings, Implications, Limitations
- **Riproducibilità**: comandi end-to-end, output deterministici, fallback sintetico offline
- **Separazione chiara** tra engine quantitativo (`src/`) e presentazione (`app.py`, `report/`)

---

## Architettura Tecnica

### 1) Core Analytics (`src/`)
- `module1_portfolio_optimization.py`
  - Mean-variance optimization su benchmark 60/40
  - Sweep allocazioni crypto 0–10%
  - Metriche: Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR
- `module2_macro_regimes.py`
  - Classificazione regimi: Risk-On, Risk-Off, Inflation, Rate Hike
  - Correlazioni rolling BTC vs asset tradizionali (30/90/180d)
  - Fallback sintetico se `yfinance` non disponibile
- `module3_onchain_signals.py`
  - Indicatori on-chain sintetici calibrati (SOPR, MVRV Z, Realized Price)
  - Backtest strategie: statica vs filtri SOPR/MVRV

### 2) Dashboard (`app.py`)
Navigazione sidebar con 6 pagine:
1. `🏠 Home`
2. `📊 Portfolio Optimizer`
3. `🌍 Macro Regimes`
4. `⛓ On-Chain Signals`
5. `🎯 My Portfolio`
6. `📄 Technical Summary`

### 3) Reporting (`report/`)
- `executive_summary.py`: genera PDF A4 istituzionale 2 pagine
- `Executive_Summary_Paolo_Maizza_Digital_Assets.pdf`: output finale distribuibile

---

## Struttura Repository

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

## Moduli e Stato

| Module | Topic | Status | Script |
|--------|-------|--------|--------|
| 1 | Portfolio Optimization & Efficient Frontier | ✅ Complete | `src/module1_portfolio_optimization.py` |
| 2 | Macro Regime Analysis & Correlations | ✅ Complete | `src/module2_macro_regimes.py` |
| 3 | On-Chain Risk Indicators | ✅ Complete | `src/module3_onchain_signals.py` |

---

## Key Findings (Sintesi)

### Module 1 — Portfolio Optimization
- Allocazione ottimale BTC+ETH: **4%**
- Miglioramento Sharpe vs 60/40: **+1.9%**
- Rebalancing più efficiente: **Quarterly**

### Module 2 — Macro Regime Analysis
- La correlazione BTC/SPY è **regime-dependent**, non strutturale
- Diversificazione più forte in **Risk-On** e **Inflation Shock**
- In **Risk-Off (VIX > 25)** BTC tende a correlarsi di più con SPY

### Module 3 — On-Chain Signals
- Filtro SOPR: miglior profilo risk-adjusted vs strategia statica
- MVRV Z-score: utile come segnale di riduzione rischio ai top di ciclo
- On-chain come overlay quantitativo, non sostituzione dell’asset allocation strategica

---

## Installazione (Comandi In Fila)

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Esecuzione Completa (End-to-End)

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate

python src/module1_portfolio_optimization.py
python src/module2_macro_regimes.py
python src/module3_onchain_signals.py
```

Verifica output:

```bash
ls -la outputs/module1
ls -la outputs/module2
ls -la outputs/module3
```

---

## Avvio Dashboard Streamlit

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate
streamlit run app.py
```

Dashboard live (se disponibile):  
`https://crypto-impact-4mzxlfm56fjvpprtjsezq7.streamlit.app`

---

## Generazione Executive Summary PDF

```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate
python report/executive_summary.py
```

Output:
- `report/Executive_Summary_Paolo_Maizza_Digital_Assets.pdf`

---

## Test Tecnici Rapidi

### 1) Check sintassi
```bash
cd "/Users/paolomaizza/crypto-impact/crypto-impact"
source .venv/bin/activate
python -m py_compile app.py
python -m py_compile src/module1_portfolio_optimization.py
python -m py_compile src/module2_macro_regimes.py
python -m py_compile src/module3_onchain_signals.py
python -m py_compile report/executive_summary.py
```

### 2) Smoke test output principali
```bash
test -f outputs/module1/fig1_efficient_frontier.png && echo "OK module1"
test -f outputs/module2/module2_rolling_correlations_regimes.png && echo "OK module2"
test -f outputs/module3/fig10_sopr_signal.png && echo "OK module3"
test -f report/Executive_Summary_Paolo_Maizza_Digital_Assets.pdf && echo "OK report"
```

---

## Data, Assunzioni e Limitazioni
- Periodo campione principale: **2018–2026**
- Fonte primaria mercato: **Yahoo Finance**
- Modalità offline: fallback sintetico GBM calibrato
- On-chain attuale: segnali sintetici calibrati (non feed live Glassnode)
- Le performance storiche non garantiscono risultati futuri

---

## Disclaimer
Questo progetto è destinato a fini educativi e di ricerca quantitativa.  
**Non costituisce consulenza finanziaria, raccomandazione d’investimento o sollecitazione all’acquisto/vendita di strumenti finanziari.**
