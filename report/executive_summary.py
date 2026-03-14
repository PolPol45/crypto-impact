from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = A4
MARGIN = 71  # ~2.5cm
BODY_FONT = "Times-Roman"
BODY_SIZE = 10.5
BODY_LEADING = 14
HEAD_FONT = "Helvetica-Bold"
NAVY = colors.HexColor("#1A3A5C")
LIGHT_BLUE = colors.HexColor("#A8C4E0")
BOX_FILL = colors.HexColor("#EEF4FB")
TEAL = colors.HexColor("#0F6E56")
AMBER = colors.HexColor("#BA7517")
RED_DARK = colors.HexColor("#8B0000")

OUT_PATH = Path("report/Executive_Summary_Paolo_Maizza_Digital_Assets.pdf")


def draw_watermark(c: canvas.Canvas) -> None:
    c.saveState()
    try:
        c.setFillAlpha(0.25)
    except Exception:
        pass
    c.setFillColor(colors.lightgrey)
    c.setFont("Helvetica-Bold", 22)
    c.translate(PAGE_W / 2, PAGE_H / 2)
    c.rotate(35)
    c.drawCentredString(0, 0, "INDEPENDENT RESEARCH — NOT INVESTMENT ADVICE")
    c.restoreState()


def draw_header(c: canvas.Canvas) -> None:
    y = PAGE_H - 32
    c.setFillColor(NAVY)
    c.rect(MARGIN, y, PAGE_W - 2 * MARGIN, 6, stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, y + 10, "Digital Assets in Institutional Portfolios")
    c.setFont("Helvetica", 9)
    c.drawRightString(PAGE_W - MARGIN, y + 10, "Paolo Maizza | March 2026")


def draw_footer(c: canvas.Canvas, page_num: int, total_pages: int = 2) -> None:
    y = 24
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(MARGIN, y, "Digital Assets in Institutional Portfolios — Paolo Maizza — NUS MSc Management")
    c.drawCentredString(PAGE_W / 2, y, f"Page {page_num} of {total_pages}")
    c.drawRightString(PAGE_W - MARGIN, y, "Not investment advice. For informational purposes only.")


def draw_section_title(c: canvas.Canvas, y: float, title: str) -> float:
    c.setFillColor(NAVY)
    c.setFont(HEAD_FONT, 11)
    c.drawString(MARGIN, y, title)
    c.setStrokeColor(NAVY)
    c.setLineWidth(0.5)
    c.line(MARGIN, y - 4, PAGE_W - MARGIN, y - 4)
    return y - 16


def draw_wrapped(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str = BODY_FONT,
    size: float = BODY_SIZE,
    leading: float = BODY_LEADING,
    color: colors.Color = colors.black,
) -> float:
    c.setFont(font, size)
    c.setFillColor(color)
    lines = simpleSplit(text, font, size, width)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_boxed_paragraph(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    w: float,
    title: str,
    text: str,
    border_color: colors.Color,
    fill_color: colors.Color,
) -> float:
    text_width = w - 16
    label_lines = simpleSplit(title, "Helvetica-Bold", 9, text_width)
    body_lines = simpleSplit(text, BODY_FONT, 10, text_width)
    box_h = 8 + len(label_lines) * 12 + len(body_lines) * 13 + 8

    c.setFillColor(fill_color)
    c.setStrokeColor(border_color)
    c.setLineWidth(1)
    c.rect(x, y_top - box_h, w, box_h, stroke=1, fill=1)

    y = y_top - 14
    for ln in label_lines:
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.black)
        c.drawString(x + 8, y, ln)
        y -= 12
    for ln in body_lines:
        c.setFont(BODY_FONT, 10)
        c.drawString(x + 8, y, ln)
        y -= 13
    return y_top - box_h


def draw_key_finding_card(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    w: float,
    h: float,
    bar_color: colors.Color,
    metric_name: str,
    metric_value: str,
    implication: str,
) -> None:
    c.setStrokeColor(colors.HexColor("#D0D8E2"))
    c.setFillColor(colors.white)
    c.rect(x, y_top - h, w, h, stroke=1, fill=1)

    c.setFillColor(bar_color)
    c.rect(x, y_top - 10, w, 10, stroke=0, fill=1)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 8, y_top - 24, metric_name)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(NAVY)
    c.drawString(x + 8, y_top - 42, metric_value)

    c.setFillColor(colors.black)
    c.setFont(BODY_FONT, 8.7)
    y = y_top - 58
    for ln in simpleSplit(implication, BODY_FONT, 8.7, w - 16):
        c.drawString(x + 8, y, ln)
        y -= 11


def draw_table(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    col_widths: Sequence[float],
    rows: Sequence[Sequence[str]],
    header_fill: colors.Color = NAVY,
    alt_fill: colors.Color = colors.HexColor("#F7FAFE"),
    font_size: float = 8.5,
) -> float:
    row_h = 18
    total_w = sum(col_widths)

    # Header
    c.setFillColor(header_fill)
    c.rect(x, y_top - row_h, total_w, row_h, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", font_size)
    cx = x + 4
    for i, cell in enumerate(rows[0]):
        c.drawString(cx, y_top - 12, cell)
        cx += col_widths[i]

    y = y_top - row_h
    for ridx, row in enumerate(rows[1:], start=1):
        y_next = y - row_h
        fill = alt_fill if ridx % 2 == 0 else colors.white
        c.setFillColor(fill)
        c.rect(x, y_next, total_w, row_h, stroke=0, fill=1)
        c.setStrokeColor(colors.HexColor("#C9D4E1"))
        c.rect(x, y_next, total_w, row_h, stroke=1, fill=0)

        c.setFillColor(colors.black)
        c.setFont(BODY_FONT, font_size)
        cx = x + 4
        for i, cell in enumerate(row):
            c.drawString(cx, y_next + 6, cell)
            cx += col_widths[i]
        y = y_next

    # Vertical separators
    c.setStrokeColor(colors.HexColor("#C9D4E1"))
    cx = x
    for w in col_widths[:-1]:
        cx += w
        c.line(cx, y_top - row_h, cx, y)

    return y


def draw_cover_and_summary_page(c: canvas.Canvas) -> None:
    draw_watermark(c)
    draw_header(c)
    draw_footer(c, 1)

    top_y = PAGE_H - MARGIN - 16

    # Cover block
    cover_h = 110
    c.setFillColor(NAVY)
    c.rect(MARGIN, top_y - cover_h, PAGE_W - 2 * MARGIN, cover_h, stroke=0, fill=1)

    tx = MARGIN + 14
    y = top_y - 26
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(tx, y, "Digital Assets in Institutional Portfolios")

    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(tx, y, "A Multi-Dimensional Investment Research Framework")

    y -= 22
    c.setFillColor(LIGHT_BLUE)
    c.setFont("Helvetica", 9)
    c.drawString(tx, y, "Paolo Maizza  |  NUS Master in Management  |  March 2026")

    y -= 16
    c.drawString(tx, y, "github.com/PolPol45/crypto-impact  |  Live Dashboard: streamlit.app")

    y = top_y - cover_h - 16

    # Research question box
    rq_text = (
        "What is the optimal allocation to Bitcoin and Ethereum in a diversified "
        "institutional portfolio, and under which macroeconomic regimes does this "
        "allocation provide genuine risk-adjusted diversification benefits?"
    )
    y = draw_boxed_paragraph(
        c,
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        "RESEARCH QUESTION",
        rq_text,
        NAVY,
        BOX_FILL,
    )

    y -= 18

    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, y, "Bottom Line")
    c.setStrokeColor(NAVY)
    c.setLineWidth(0.5)
    c.line(MARGIN, y - 4, PAGE_W - MARGIN, y - 4)

    y -= 18
    bluf = (
        "A 4% allocation to BTC+ETH (2.86% BTC / 1.14% ETH) improves the Sharpe "
        "ratio of a standard 60/40 institutional portfolio by +1.9% over the "
        "2018–2026 sample period. This benefit is regime-dependent: strongest during "
        "Risk-On and Inflation Shock environments, materially weaker during Risk-Off "
        "and Rate Hike cycles. On-chain risk signals (SOPR, MVRV Z-Score) provide "
        "a quantitative overlay to reduce drawdown at cycle tops without sacrificing "
        "the strategic allocation."
    )
    y = draw_wrapped(c, bluf, MARGIN, y, PAGE_W - 2 * MARGIN)

    y -= 10

    card_w = (PAGE_W - 2 * MARGIN - 2 * 10) / 3
    card_h = 150
    draw_key_finding_card(
        c,
        MARGIN,
        y,
        card_w,
        card_h,
        NAVY,
        "PORTFOLIO OPTIMIZATION",
        "4% BTC+ETH",
        "Improves Sharpe +1.9% vs 60/40. Quarterly rebalancing optimal (21Shares 2024).",
    )
    draw_key_finding_card(
        c,
        MARGIN + card_w + 10,
        y,
        card_w,
        card_h,
        TEAL,
        "MACRO REGIME ANALYSIS",
        "Regime-Driven",
        "BTC/SPY correlation: 0.68 (Risk-Off) vs 0.31 (Risk-On). Diversification is conditional, not structural.",
    )
    draw_key_finding_card(
        c,
        MARGIN + 2 * (card_w + 10),
        y,
        card_w,
        card_h,
        AMBER,
        "ON-CHAIN SIGNALS",
        "SOPR + MVRV",
        "SOPR filter reduces max drawdown. MVRV Z>7 signals cycle top with historical accuracy.",
    )


def draw_methodology_sources(c: canvas.Canvas, y_top: float) -> float:
    y = draw_section_title(c, y_top, "1. Methodology & Data Sources")

    gap = 16
    total_w = PAGE_W - 2 * MARGIN
    left_w = total_w * 0.58
    right_w = total_w - left_w - gap

    lx = MARGIN
    rx = lx + left_w + gap

    ly = y
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(NAVY)
    c.drawString(lx, ly, "Module 1 — Portfolio Optimization")
    ly -= 12
    ly = draw_wrapped(
        c,
        "Mean-variance optimization (Markowitz 1952) applied to 60/40 benchmark "
        "(SPY+AGG). 11 portfolio configurations tested (0–10% crypto, 1% steps). "
        "Metrics: Sharpe, Sortino, Calmar, Max Drawdown, VaR 95%, CVaR. Follows VanEck "
        "(2024) and ARK Invest (2025) methodology.",
        lx,
        ly,
        left_w,
        BODY_FONT,
        9.2,
        12,
    )

    ly -= 6
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(NAVY)
    c.drawString(lx, ly, "Module 2 — Macro Regime Analysis")
    ly -= 12
    ly = draw_wrapped(
        c,
        "Four rule-based regimes classified daily: Risk-On (VIX<20, SPY>SMA50), "
        "Risk-Off (VIX>25), Inflation Shock (Mar 2021–Jun 2022), Rate Hike Cycle "
        "(Mar 2022–Jul 2023). Rolling Pearson correlations at 30/90/180-day windows. "
        "Follows Fidelity Digital Assets (2025) framework.",
        lx,
        ly,
        left_w,
        BODY_FONT,
        9.2,
        12,
    )

    ly -= 6
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(NAVY)
    c.drawString(lx, ly, "Module 3 — On-Chain Risk Signals")
    ly -= 12
    ly = draw_wrapped(
        c,
        "SOPR (Spent Output Profit Ratio), MVRV Z-Score, and Realized Price calibrated "
        "to historical BTC cycles. Three backtest strategies: static 4% BTC, SOPR-filtered "
        "(reduce to 1% at euphoria), MVRV-filtered (reduce to 0% at extreme overvaluation). "
        "Methodology consistent with CoinShares Research (2024).",
        lx,
        ly,
        left_w,
        BODY_FONT,
        9.2,
        12,
    )

    rows = [
        ("Source", "Dataset", "Period"),
        ("Yahoo Finance", "BTC, ETH, SPY, AGG, GLD, DXY, VIX", "2018–2026"),
        ("FRED", "CPI, Fed Funds Rate, T-Bill 3M", "2018–2026"),
        ("Synthetic GBM", "Offline calibrated fallback", "2018–2026"),
        ("On-chain (synthetic)", "SOPR, MVRV Z-Score", "2018–2026"),
    ]
    ry_bottom = draw_table(c, rx, y, [right_w * 0.32, right_w * 0.45, right_w * 0.23], rows, font_size=7.8)

    note_y = ry_bottom - 12
    note = (
        "Sample covers 2 complete BTC bull/bear cycles plus the post-ETF approval "
        "regime (Jan 2024+). Risk-free rate: 3M T-Bill proxy at 4.5% (2024–25 avg)."
    )
    note_y = draw_wrapped(c, note, rx, note_y, right_w, "Times-Italic", 9, 12)

    return min(ly, note_y) - 6


def draw_implication_block(c: canvas.Canvas, x: float, y_top: float, w: float, h: float, border_color: colors.Color, title: str, text: str) -> None:
    c.setStrokeColor(colors.HexColor("#D7DDE6"))
    c.setFillColor(colors.white)
    c.rect(x, y_top - h, w, h, stroke=1, fill=1)
    c.setFillColor(border_color)
    c.rect(x, y_top - h, 3, h, stroke=0, fill=1)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 8, y_top - 14, title)

    y = y_top - 28
    c.setFont(BODY_FONT, 8.8)
    for ln in simpleSplit(text, BODY_FONT, 8.8, w - 14):
        c.drawString(x + 8, y, ln)
        y -= 10.8


def draw_practical_implications(c: canvas.Canvas, y_top: float) -> float:
    y = draw_section_title(c, y_top, "2. Practical Implications")

    gap = 12
    w = (PAGE_W - 2 * MARGIN - gap) / 2
    h = 86

    blocks = [
        (
            NAVY,
            "FOR PORTFOLIO MANAGERS",
            "A 1–4% BTC/ETH strategic allocation improves risk-adjusted returns without materially increasing drawdown risk. The optimal implementation is a quarterly-rebalanced overlay funded proportionally from equities and bonds, consistent with the institutional 60/40+ framework emerging across the industry (JPMorgan AM, 2026 Year-Ahead Outlook).",
        ),
        (
            TEAL,
            "FOR RISK COMMITTEES",
            "Crypto diversification benefits are regime-conditional. Risk-Off periods (VIX>25) see elevated BTC/equity correlation — hedge properties weaken precisely when they are most needed. Committees should model regime-dependent correlation matrices rather than static assumptions.",
        ),
        (
            AMBER,
            "FOR MACRO INVESTMENT TEAMS",
            "BTC behaves as a risk asset during liquidity crises and as an inflation hedge during CPI shock regimes. Dynamic allocation (increase in Risk-On/Inflation, reduce in Risk-Off/Rate Hike) outperforms static exposure over full market cycles.",
        ),
        (
            RED_DARK,
            "FOR DIGITAL ASSET TEAMS",
            "On-chain indicators (SOPR, MVRV Z-Score) provide actionable portfolio signals unavailable from price-only analysis. This framework is the first to integrate on-chain overlays quantitatively into a Markowitz optimization context — differentiating it from VanEck (2024), ARK (2025), and Grayscale (2024) papers.",
        ),
    ]

    draw_implication_block(c, MARGIN, y, w, h, blocks[0][0], blocks[0][1], blocks[0][2])
    draw_implication_block(c, MARGIN + w + gap, y, w, h, blocks[1][0], blocks[1][1], blocks[1][2])
    y2 = y - h - 10
    draw_implication_block(c, MARGIN, y2, w, h, blocks[2][0], blocks[2][1], blocks[2][2])
    draw_implication_block(c, MARGIN + w + gap, y2, w, h, blocks[3][0], blocks[3][1], blocks[3][2])

    return y2 - h - 8


def draw_dashboard_section(c: canvas.Canvas, y_top: float) -> float:
    y = draw_section_title(c, y_top, "3. Interactive Dashboard")

    y = draw_wrapped(
        c,
        "The live dashboard at the URL below allows portfolio managers to explore all three analytical modules interactively. No installation required.",
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        BODY_FONT,
        9.2,
        12,
    )

    y -= 6
    rows = [
        ("Step", "Page", "Action", "Output"),
        ("1", "Portfolio Optimizer", "Adjust allocation slider 0–10%", "Real-time Sharpe/Sortino/Calmar"),
        ("2", "Macro Regimes", "Review heatmaps by regime", "Correlation structure by cycle"),
        ("3", "On-Chain Signals", "Check institutional scorecard", "SOPR + MVRV signal (🟢🟡🔴)"),
        ("4", "My Portfolio", "Enter your current holdings", "Personalized allocation analysis"),
        ("5", "Technical Summary", "Review full methodology", "Reproducible research framework"),
    ]
    y = draw_table(
        c,
        MARGIN,
        y,
        [40, 130, 190, PAGE_W - 2 * MARGIN - 40 - 130 - 190],
        rows,
        font_size=7.6,
    )

    y -= 12
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, y, "Live Dashboard: https://crypto-impact-4mzxlfm56fjvpprtjsezq7.streamlit.app")
    y -= 12
    c.drawString(MARGIN, y, "GitHub Repository: https://github.com/PolPol45/crypto-impact")

    return y - 8


def draw_references(c: canvas.Canvas, y_top: float) -> None:
    refs = [
        '[1] VanEck (2024). "Optimal Crypto Allocation for Portfolios."',
        '[2] ARK Invest (2025). "Measuring Bitcoin\'s Risk and Reward."',
        '[3] 21Shares (2024). "Cryptoassets in a Diversified Portfolio."',
        '[4] Fidelity Digital Assets (2025). "The Case for Bitcoin — Institutional Insights."',
        '[5] Grayscale Research (2024). "The Role of Crypto in a Portfolio."',
        '[6] CoinShares Research (2024). "Digital Asset Fund Flows Weekly."',
        '[7] Goldman Sachs (2026). "Digital Assets: From Experiment to Asset Class."',
        '[8] JPMorgan Asset Management (2026). "2026 Year-Ahead Investment Outlook."',
        '[9] CFA Institute (2020). "Equity Research Report Essentials."',
        '[10] Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77–91.',
    ]

    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, y_top, "References")
    c.setStrokeColor(NAVY)
    c.setLineWidth(0.5)
    c.line(MARGIN, y_top - 3, PAGE_W - MARGIN, y_top - 3)

    col_gap = 14
    col_w = (PAGE_W - 2 * MARGIN - col_gap) / 2
    left_refs = refs[:5]
    right_refs = refs[5:]

    y_left = y_top - 13
    c.setFillColor(colors.black)
    c.setFont(BODY_FONT, 8)
    for r in left_refs:
        for ln in simpleSplit(r, BODY_FONT, 8, col_w):
            c.drawString(MARGIN, y_left, ln)
            y_left -= 10

    y_right = y_top - 13
    x_right = MARGIN + col_w + col_gap
    for r in right_refs:
        for ln in simpleSplit(r, BODY_FONT, 8, col_w):
            c.drawString(x_right, y_right, ln)
            y_right -= 10


def draw_page_2(c: canvas.Canvas) -> None:
    draw_watermark(c)
    draw_header(c)
    draw_footer(c, 2)

    y = PAGE_H - MARGIN - 16
    y = draw_methodology_sources(c, y)
    y = draw_practical_implications(c, y)
    y = draw_dashboard_section(c, y)
    draw_references(c, y)


def generate_pdf(path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)

    draw_cover_and_summary_page(c)
    c.showPage()
    draw_page_2(c)
    c.save()

    return path


def main() -> None:
    output = generate_pdf()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
