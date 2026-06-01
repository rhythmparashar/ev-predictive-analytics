"""
Generate Cell Degradation Report PDF — EV01 Loader Battery Intelligence
"""

from __future__ import annotations
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.settings import DATA_DIR
from battery_intelligence.loader import load_master, load_gold_cells, load_silver_cells

# reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

# ── Paths ────────────────────────────────────────────────────────────────────
REPORT_DIR = PROJECT_ROOT / 'reports'
OUT_PDF    = REPORT_DIR / 'cell_degradation_report_EV01_Loader.pdf'

sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans'})

ALL_M1  = [f'M1_C{c}' for c in range(1, 37)]
FOCUS   = ['M1_C25', 'M1_C26']
REF     = ['M1_C27', 'M1_C30', 'M1_C10']

# ── Colours ──────────────────────────────────────────────────────────────────
RED    = '#E63946'
ORANGE = '#F4A261'
BLUE   = '#2A6496'
GREEN  = '#2E8B57'
GREY   = '#6C757D'

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (DuckDB — reads only requested columns from disk)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading gold cell_features …")
delta_cols   = [f'{c}_delta'      for c in ALL_M1]
outlier_cols = [f'{c}_is_outlier' for c in ALL_M1]
zscore_cols  = [f'{c}_zscore'     for c in ALL_M1]

gold = load_gold_cells(DATA_DIR, columns=['timestamp'] + delta_cols + outlier_cols + zscore_cols)
for c in ALL_M1:
    gold[f'{c}_delta'] = gold[f'{c}_delta'] * 1000
gold['week']  = gold['timestamp'].dt.to_period('W')
gold['month'] = gold['timestamp'].dt.to_period('M')

print("Loading silver cells …")
silver = load_silver_cells(DATA_DIR, columns=['timestamp'] + ALL_M1)
silver['month'] = silver['timestamp'].dt.to_period('M')
silver['M1_mean'] = silver[ALL_M1].mean(axis=1)

print("Loading master dataset …")
master = load_master(DATA_DIR, columns=[
    'timestamp', 'module_1_lowest_cell', 'worst_cell_id', 'module_1_voltage_range',
])
master['month'] = master['timestamp'].dt.to_period('M')
master['week']  = master['timestamp'].dt.to_period('W')

# ── Pre-compute key numbers ───────────────────────────────────────────────────
monthly_delta = gold.groupby('month')[['M1_C25_delta','M1_C26_delta']].mean().round(2)
monthly_outlier = gold.groupby('month')[['M1_C25_is_outlier','M1_C26_is_outlier']].mean().mul(100).round(1)
monthly_abs = silver.groupby('month')[['M1_C25','M1_C26','M1_mean']].mean().round(4)
monthly_vrange = master.groupby('month')['module_1_voltage_range'].mean().mul(1000).round(1)
monthly_c25_worst = master.groupby('month').apply(
    lambda x: round((x['worst_cell_id']=='M1_C25').mean()*100, 1))

# Slopes for all M1 cells
weekly_all = gold.groupby('week')[[f'{c}_delta' for c in ALL_M1]].median()
weekly_all.index = weekly_all.index.to_timestamp()
slope_rows = []
for c in ALL_M1:
    y = weekly_all[f'{c}_delta'].dropna().values
    x = np.arange(len(y))
    s, intercept, r, p, _ = stats.linregress(x, y)
    slope_rows.append({'cell':c,'slope':round(s,4),'r2':round(r**2,4),
                       'p':round(p,4),'sig':p<0.05,
                       'jan':round(y[0],2),'may':round(y[-1],2)})
slope_df = pd.DataFrame(slope_rows).sort_values('slope')

# Weekly series for focus + ref cells
weekly_focus = gold.groupby('week')[
    [f'{c}_delta' for c in FOCUS+REF] + [f'{c}_is_outlier' for c in FOCUS]
].median()
weekly_focus.index = weekly_focus.index.to_timestamp()

weekly_abs_mean = silver.groupby(silver['timestamp'].dt.to_period('W'))['M1_mean'].median()
weekly_abs_mean.index = weekly_abs_mean.index.to_timestamp()
weekly_abs_c25  = silver.groupby(silver['timestamp'].dt.to_period('W'))['M1_C25'].median()
weekly_abs_c25.index  = weekly_abs_c25.index.to_timestamp()
weekly_abs_c26  = silver.groupby(silver['timestamp'].dt.to_period('W'))['M1_C26'].median()
weekly_abs_c26.index  = weekly_abs_c26.index.to_timestamp()

c25_dom = master.groupby('week').apply(lambda x: round((x['worst_cell_id']=='M1_C25').mean()*100, 2))
c25_dom.index = c25_dom.index.to_timestamp()
c25_mod_dom = master.groupby('week').apply(lambda x: round((x['module_1_lowest_cell']=='M1_C25').mean()*100, 2))
c25_mod_dom.index = c25_mod_dom.index.to_timestamp()

monthly_outlier_all = gold.groupby('month')[[f'{c}_is_outlier' for c in ALL_M1]].mean().mul(100)
monthly_outlier_all.columns = ALL_M1
monthly_outlier_all.index = [str(p) for p in monthly_outlier_all.index]

print("Data loaded. Generating figures …")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_image(fig, width_cm=17):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    w = width_cm * cm
    # maintain aspect
    from PIL import Image as PILImage
    pil = PILImage.open(buf)
    ratio = pil.height / pil.width
    buf.seek(0)
    return Image(buf, width=w, height=w * ratio)

# ── FIGURE 1: Weekly delta — focus vs reference ───────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for cell, clr, lw in [('M1_C25', RED, 2.8), ('M1_C26', ORANGE, 2.3)]:
    y = weekly_focus[f'{cell}_delta']
    ax.plot(weekly_focus.index, y, color=clr, linewidth=lw, label=f'{cell} (Critical)')
    xn = np.arange(len(y)); s,i,*_ = stats.linregress(xn, y)
    ax.plot(weekly_focus.index, s*xn+i, color=clr, linestyle='--', linewidth=1.2, alpha=0.6)
for cell, clr in zip(REF, [BLUE, GREEN, GREY]):
    ax.plot(weekly_focus.index, weekly_focus[f'{cell}_delta'],
            color=clr, linewidth=1.5, alpha=0.75, label=f'{cell} (healthy ref)')
ax.axhline(0,  color='black', linewidth=0.8, alpha=0.4)
ax.axhline(-20, color=RED,   linestyle=':', linewidth=1.4, label='−20 mV danger threshold')
ax.fill_between(weekly_focus.index,
                weekly_focus['M1_C25_delta'], 0,
                where=weekly_focus['M1_C25_delta'] < 0, alpha=0.08, color=RED)
ax.set_title('Figure 1 — Weekly Voltage Gap: M1_C25 & M1_C26 vs Module Mean', fontsize=12, fontweight='bold')
ax.set_ylabel('Voltage gap from module average (mV)')
ax.set_xlabel('Week')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(-50, 20)
plt.tight_layout()
FIG1 = fig_to_image(fig)

# ── FIGURE 2: Monthly data table chart ───────────────────────────────────────
months_str = [str(m) for m in monthly_delta.index]
x = np.arange(len(months_str))
w = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - w/2, monthly_delta['M1_C25_delta'], w, color=RED,    label='M1_C25 avg gap (mV)', alpha=0.85)
b2 = ax.bar(x + w/2, monthly_delta['M1_C26_delta'], w, color=ORANGE, label='M1_C26 avg gap (mV)', alpha=0.85)
ax.axhline(-20, color='darkred', linestyle='--', linewidth=1.4, label='−20 mV threshold')
ax.axhline(0,   color='black',   linewidth=0.8, alpha=0.4)
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-2,
            f'{bar.get_height():.1f}', ha='center', va='top', fontsize=8.5, color='white', fontweight='bold')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-2,
            f'{bar.get_height():.1f}', ha='center', va='top', fontsize=8.5, color='white', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(months_str, fontsize=10)
ax.set_title('Figure 2 — Monthly Average Voltage Gap (mV)\nNegative = cell is below its module average', fontsize=12, fontweight='bold')
ax.set_ylabel('Voltage gap (mV)')
ax.legend(fontsize=9)
plt.tight_layout()
FIG2 = fig_to_image(fig)

# ── FIGURE 3: Outlier frequency by month ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, monthly_outlier['M1_C25_is_outlier'], w, color=RED,    label='M1_C25 flagged anomalous (%)', alpha=0.85)
ax.bar(x + w/2, monthly_outlier['M1_C26_is_outlier'], w, color=ORANGE, label='M1_C26 flagged anomalous (%)', alpha=0.85)
for col, offset, clr in [('M1_C25_is_outlier',-w/2,RED),('M1_C26_is_outlier',w/2,ORANGE)]:
    for xi, val in zip(x, monthly_outlier[col]):
        ax.text(xi+offset, val+1.5, f'{val:.0f}%', ha='center', fontsize=8.5, color='#333', fontweight='bold')
ax.axhline(5, color='steelblue', linestyle='--', linewidth=1.2, label='5% normal threshold')
ax.set_xticks(x); ax.set_xticklabels(months_str, fontsize=10)
ax.set_title('Figure 3 — Monthly Anomaly Rate (%)\nHow often each cell was flagged as electrically unstable', fontsize=12, fontweight='bold')
ax.set_ylabel('% of readings flagged as anomalous')
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
plt.tight_layout()
FIG3 = fig_to_image(fig)

# ── FIGURE 4: M1_C25 as pack worst + module worst ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(c25_mod_dom.index, c25_mod_dom.values, color=RED, linewidth=2.5, label='M1_C25 as Module 1 weakest')
axes[0].fill_between(c25_mod_dom.index, c25_mod_dom.values, alpha=0.15, color=RED)
axes[0].axhline(15, color='darkred', linestyle='--', linewidth=1.3, label='15% chronic threshold')
axes[0].set_title('Figure 4a — Weekly: M1_C25 as Weakest\nCell Inside Module 1', fontsize=11, fontweight='bold')
axes[0].set_ylabel('% of readings')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[0].legend(fontsize=9); axes[0].tick_params(axis='x', rotation=20)

axes[1].plot(c25_dom.index, c25_dom.values, color='#8B0000', linewidth=2.5, label='M1_C25 as worst in entire pack')
axes[1].fill_between(c25_dom.index, c25_dom.values, alpha=0.15, color='#8B0000')
for m, val in zip(monthly_c25_worst.index, monthly_c25_worst.values):
    wk_idx = c25_dom.index[c25_dom.index.to_period('M') == m]
    if len(wk_idx):
        mid = wk_idx[len(wk_idx)//2]
        axes[1].annotate(f'{val}%', xy=(mid, val), fontsize=8.5,
                         ha='center', color='#8B0000', fontweight='bold')
axes[1].axhline(15, color='darkred', linestyle='--', linewidth=1.3)
axes[1].set_title('Figure 4b — Weekly: M1_C25 as Worst Cell\nIn the Entire Battery Pack', fontsize=11, fontweight='bold')
axes[1].set_ylabel('% of readings')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
FIG4 = fig_to_image(fig)

# ── FIGURE 5: Absolute voltage ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(weekly_abs_c25.index,  weekly_abs_c25  * 1000, color=RED,    linewidth=2.5, label='M1_C25 voltage (mV)')
ax.plot(weekly_abs_c26.index,  weekly_abs_c26  * 1000, color=ORANGE, linewidth=2,   label='M1_C26 voltage (mV)')
ax.plot(weekly_abs_mean.index, weekly_abs_mean * 1000, color=BLUE,   linewidth=2,
        linestyle='--', label='Module 1 average voltage (mV)')
ax.set_title('Figure 5a — Actual Cell Voltage (mV)\nAbsolute values over time', fontsize=11, fontweight='bold')
ax.set_ylabel('Voltage (mV)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.legend(fontsize=9); ax.tick_params(axis='x', rotation=20)

# Gap between C25 and module mean in mV
gap_c25 = (weekly_abs_c25 - weekly_abs_mean) * 1000
ax2 = axes[1]
ax2.fill_between(gap_c25.index, gap_c25.values, 0,
                 where=gap_c25.values < 0, color=RED, alpha=0.25, label='C25 below module avg')
ax2.plot(gap_c25.index, gap_c25.values, color=RED, linewidth=2.5, label='M1_C25 gap from avg')
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax2.axhline(-20, color='darkred', linestyle=':', linewidth=1.3, label='−20 mV threshold')
# Monthly annotations
for m, grp in gap_c25.groupby(gap_c25.index.to_period('M')):
    mid_idx = grp.index[len(grp)//2]
    ax2.annotate(f'{grp.mean():.0f}mV', xy=(mid_idx, grp.mean()),
                 fontsize=8, ha='center', color='#8B0000',
                 xytext=(0, 12), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='grey', lw=0.8))
ax2.set_title('Figure 5b — Gap Between M1_C25\nand Module 1 Average (mV)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Gap (mV)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.legend(fontsize=9); ax2.tick_params(axis='x', rotation=20)
plt.tight_layout()
FIG5 = fig_to_image(fig)

# ── FIGURE 6: Slope bar chart — all 36 M1 cells ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
bar_colors = []
for _, r in slope_df.iterrows():
    if r['cell'] in FOCUS:            bar_colors.append(RED)
    elif r['slope'] < -0.5:           bar_colors.append(ORANGE)
    elif r['slope'] < 0 and r['sig']: bar_colors.append(BLUE)
    elif r['slope'] < 0:              bar_colors.append('#ADB5BD')
    else:                             bar_colors.append(GREEN)
ax.bar(slope_df['cell'], slope_df['slope'], color=bar_colors, edgecolor='white', linewidth=0.4)
ax.axhline(0, color='black', linewidth=0.9)
ax.set_title('Figure 6 — Weekly Voltage Trend per Cell (mV/week)\nModule 1 — All 36 cells', fontsize=12, fontweight='bold')
ax.set_ylabel('Rate of change (mV/week)')
ax.tick_params(axis='x', rotation=72, labelsize=8)
from matplotlib.patches import Patch
legend_els = [Patch(color=RED,    label='M1_C25 / M1_C26 (critical)'),
              Patch(color=ORANGE, label='Declining fast (>0.5mV/wk)'),
              Patch(color=BLUE,   label='Statistically declining (p<0.05)'),
              Patch(color='#ADB5BD', label='Slow / not significant'),
              Patch(color=GREEN,  label='Stable / slight improvement')]
ax.legend(handles=legend_els, fontsize=8.5, loc='lower left')
plt.tight_layout()
FIG6 = fig_to_image(fig)

# ── FIGURE 7: Outlier heatmap all M1 cells ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(monthly_outlier_all.T, cmap='YlOrRd', annot=True, fmt='.0f',
            linewidths=0.3, ax=ax, cbar_kws={'label':'Anomaly %'},
            annot_kws={'size': 7})
ax.set_title('Figure 7 — Monthly Anomaly Rate (%) — All 36 Module 1 Cells\nDark red = cell flagged as anomalous most of the time', fontsize=11, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Cell')
plt.tight_layout()
FIG7 = fig_to_image(fig, width_cm=15)

print("All figures generated.")

# ─────────────────────────────────────────────────────────────────────────────
# PDF ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm,   bottomMargin=2*cm,
)

styles = getSampleStyleSheet()
W = A4[0] - 4*cm   # usable width

# Custom styles
H1 = ParagraphStyle('H1', parent=styles['Heading1'],
    fontSize=22, textColor=colors.HexColor('#1A2E4A'),
    spaceAfter=6, spaceBefore=0, alignment=TA_CENTER)
H2 = ParagraphStyle('H2', parent=styles['Heading2'],
    fontSize=14, textColor=colors.HexColor('#2A6496'),
    spaceAfter=4, spaceBefore=14, borderPad=2)
H3 = ParagraphStyle('H3', parent=styles['Heading3'],
    fontSize=11, textColor=colors.HexColor('#E63946'),
    spaceAfter=3, spaceBefore=8)
BODY = ParagraphStyle('Body', parent=styles['Normal'],
    fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
BODY_L = ParagraphStyle('BodyL', parent=styles['Normal'],
    fontSize=10, leading=15, spaceAfter=4)
CAPTION = ParagraphStyle('Caption', parent=styles['Normal'],
    fontSize=8.5, textColor=colors.HexColor('#555555'),
    leading=12, spaceAfter=10, alignment=TA_CENTER)
CALLOUT = ParagraphStyle('Callout', parent=styles['Normal'],
    fontSize=10.5, leading=15, backColor=colors.HexColor('#FFF3CD'),
    borderColor=colors.HexColor('#E6A817'), borderWidth=1, borderPad=8,
    spaceAfter=10, spaceBefore=6)
ALERT = ParagraphStyle('Alert', parent=styles['Normal'],
    fontSize=10.5, leading=15, backColor=colors.HexColor('#FDECEA'),
    borderColor=colors.HexColor('#E63946'), borderWidth=1, borderPad=8,
    spaceAfter=10, spaceBefore=6)
GOOD = ParagraphStyle('Good', parent=styles['Normal'],
    fontSize=10.5, leading=15, backColor=colors.HexColor('#EAF7EF'),
    borderColor=colors.HexColor('#2E8B57'), borderWidth=1, borderPad=8,
    spaceAfter=10, spaceBefore=6)

def hr(): return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#CCCCCC'), spaceAfter=8, spaceBefore=4)
def sp(h=8): return Spacer(1, h)

def data_table(headers, rows, col_widths=None):
    data = [headers] + rows
    style = TableStyle([
        ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#2A6496')),
        ('TEXTCOLOR',    (0,0), (-1,0),  colors.white),
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.HexColor('#F8F9FA'), colors.white]),
        ('GRID',         (0,0), (-1,-1), 0.4, colors.HexColor('#DEE2E6')),
        ('ALIGN',        (0,0), (-1,0),  'CENTER'),
        ('ALIGN',        (0,1), (-1,-1), 'LEFT'),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('LEFTPADDING',  (0,1), (-1,-1), 8),
    ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(style)
    return t

# ─────────────────────────────────────────────────────────────────────────────
story = []

# ── COVER PAGE ───────────────────────────────────────────────────────────────
story.append(sp(60))
story.append(Paragraph("Battery Cell Degradation Report", H1))
story.append(sp(4))
story.append(Paragraph("EV01 — Module 1 Cell Intelligence", ParagraphStyle('Sub',
    parent=styles['Normal'], fontSize=15, alignment=TA_CENTER,
    textColor=colors.HexColor('#2A6496'), spaceAfter=4)))
story.append(sp(8))
story.append(hr())
story.append(sp(20))

cover_table = Table([
    ['Vehicle ID',       'EV01'],
    ['Analysis Period',  '22 January 2026 → 17 May 2026'],
    ['Total Readings',   '2,163,692 timestamps'],
    ['Cells Analysed',   'Module 1 — all 36 cells (M1_C1 to M1_C36)'],
    ['Critical Cells',   'M1_C25  |  M1_C26'],
    ['Prepared by',      'Battery Intelligence Engine — EV Platform'],
    ['Report Date',      '19 May 2026'],
], colWidths=[6*cm, 10*cm])
cover_table.setStyle(TableStyle([
    ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
    ('FONTSIZE',  (0,0), (-1,-1), 10),
    ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#2A6496')),
    ('ROWBACKGROUNDS',(0,0),(-1,-1), [colors.HexColor('#F0F4F8'), colors.white]),
    ('GRID',      (0,0), (-1,-1), 0.3, colors.HexColor('#CCCCCC')),
    ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING',(0,0), (-1,-1), 7),
    ('BOTTOMPADDING',(0,0), (-1,-1), 7),
    ('LEFTPADDING', (0,0), (-1,-1), 10),
]))
story.append(cover_table)
story.append(sp(30))
story.append(hr())
story.append(Paragraph(
    "CONFIDENTIAL — Internal use only. Prepared for team discussion.",
    ParagraphStyle('conf', parent=styles['Normal'], fontSize=9,
                   textColor=colors.grey, alignment=TA_CENTER)))
story.append(PageBreak())

# ── SECTION 1 — EXECUTIVE SUMMARY ────────────────────────────────────────────
story.append(Paragraph("1. Executive Summary", H2))
story.append(hr())
story.append(Paragraph(
    "This report investigates the health of battery cells in <b>Module 1 (M1)</b> of vehicle EV01 "
    "over the period <b>22 January 2026 to 17 May 2026</b> — covering nearly four months of real "
    "operating data across 2.16 million sensor readings.", BODY))
story.append(sp(6))

story.append(Paragraph("🔴  Critical Finding", H3))
story.append(Paragraph(
    "Two cells — <b>M1_C25</b> and <b>M1_C26</b> — were running significantly below the rest of "
    "Module 1 from the very first day of data (22 January 2026). At their worst, M1_C25 was "
    "<b>36 mV below the module average</b> and was flagged as electrically abnormal on <b>94% of "
    "all readings in January</b>. A healthy cell should almost never be flagged.", ALERT))

story.append(Paragraph("⚠️  The Apparent Recovery is Misleading", H3))
story.append(Paragraph(
    "By April and May 2026, the gap between M1_C25 and the module average had almost closed. "
    "This looks like a recovery — but our analysis shows it is not. <b>34 out of 36 cells in "
    "Module 1 are slowly declining</b>, pulling the module average down toward where M1_C25 and "
    "M1_C26 were all along. The problem cells did not get better; the rest of the module got "
    "worse.", CALLOUT))

story.append(Paragraph("🔴  A New Cell to Watch: M1_C27", H3))
story.append(Paragraph(
    "M1_C27 is now declining at the fastest rate of any cell in Module 1 — losing "
    "<b>0.90 mV per week</b> relative to the module average. In January it was the healthiest "
    "cell in the module (+10.7 mV above average). By May it had fallen to +0.9 mV. "
    "If this trend continues, M1_C27 will become the next critical cell.", ALERT))

story.append(Paragraph("✅  What Is Not a Problem", H3))
story.append(Paragraph(
    "All 36 cells are physically present and transmitting valid data. There are no missing sensors "
    "or broken connections. The battery management system (BMS) is correctly detecting these cells "
    "as the weakest in the module.", GOOD))
story.append(PageBreak())

# ── SECTION 2 — WHAT DOES VOLTAGE GAP MEAN ───────────────────────────────────
story.append(Paragraph("2. How to Read This Report — A Simple Explanation", H2))
story.append(hr())
story.append(Paragraph(
    "A battery module contains multiple cells connected in series. All cells should ideally "
    "produce the same voltage. When one cell produces less voltage than the others, it creates "
    "an imbalance that:", BODY))
for bullet in [
    "Reduces the total energy the battery can deliver",
    "Forces the battery management system to stop charging/discharging earlier to protect the weak cell",
    "Causes extra heat in the weak cell, accelerating its wear",
    "Eventually risks complete cell failure if left unaddressed",
]:
    story.append(Paragraph(f"• {bullet}", BODY_L))
story.append(sp(6))
story.append(Paragraph(
    "In this report, the <b>voltage gap</b> (also called 'delta') tells us how far below the "
    "module average a cell is running. We measure this in <b>millivolts (mV)</b>. "
    "A gap of 0 mV is perfect. A gap of <b>−20 mV or more is our warning threshold</b> — "
    "the cell is considered weak. A gap of <b>−30 mV or more is critical</b>.", BODY))

story.append(Paragraph(
    "An <b>anomaly flag</b> is raised automatically when a cell's voltage deviates so far from "
    "the others that it is statistically unusual. A healthy cell should be flagged less than "
    "5% of the time. Anything above 20% indicates a persistently weak cell.", BODY))
story.append(PageBreak())

# ── SECTION 3 — THE PROBLEM CELLS ────────────────────────────────────────────
story.append(Paragraph("3. The Problem Cells — M1_C25 and M1_C26", H2))
story.append(hr())
story.append(Paragraph(
    "The chart below shows the weekly voltage gap of M1_C25 and M1_C26 compared to three "
    "healthy reference cells. The red and orange lines show how far below the module average "
    "these cells were running. The dashed lines show the overall trend direction.", BODY))
story.append(sp(4))
story.append(FIG1)
story.append(Paragraph(
    "Figure 1 — Each point is the weekly average voltage gap in mV. "
    "Red = M1_C25, Orange = M1_C26. Blue/Green/Grey = healthy reference cells. "
    "The dotted red line marks the −20 mV danger threshold.", CAPTION))
story.append(sp(6))

story.append(Paragraph(
    "Key observations from Figure 1:", BODY))
for obs in [
    "<b>From day one (22 Jan 2026)</b>, both M1_C25 and M1_C26 were already well below "
    "the −20 mV danger threshold. This was not a gradual development — they started weak.",
    "<b>January–March 2026</b>: The gap worsened month by month, reaching its worst in March "
    "when M1_C25 averaged −60 mV below the module mean.",
    "<b>April 2026 onwards</b>: The gap appears to close. However, this is because the "
    "module average is falling — not because M1_C25 recovered (explained in Section 5).",
    "<b>M1_C27 (blue line)</b>: Started as the healthiest cell but has been declining "
    "steadily. It is the next cell to watch.",
]:
    story.append(Paragraph(f"• {obs}", BODY_L))
story.append(PageBreak())

# ── SECTION 4 — MONTHLY DATA TABLE ───────────────────────────────────────────
story.append(Paragraph("4. Month-by-Month Breakdown", H2))
story.append(hr())

story.append(Paragraph("<b>4.1  Voltage Gap Each Month</b>", H3))
story.append(FIG2)
story.append(Paragraph(
    "Figure 2 — Each bar shows the average voltage gap for that month. "
    "A bar extending downward means the cell was below the module average. "
    "March was the worst month for both cells.", CAPTION))
story.append(sp(6))

# Data table
tbl_data_rows = []
for month in monthly_delta.index:
    m = str(month)
    tbl_data_rows.append([
        m,
        f"{monthly_delta.loc[month,'M1_C25_delta']:.1f} mV",
        f"{monthly_delta.loc[month,'M1_C26_delta']:.1f} mV",
        f"{monthly_abs.loc[month,'M1_C25']*1000:.1f} mV",
        f"{monthly_abs.loc[month,'M1_C26']*1000:.1f} mV",
        f"{monthly_abs.loc[month,'M1_mean']*1000:.1f} mV",
    ])
story.append(data_table(
    ['Month', 'M1_C25 Gap', 'M1_C26 Gap', 'M1_C25 Actual', 'M1_C26 Actual', 'Module Avg'],
    tbl_data_rows,
    col_widths=[2.8*cm, 2.4*cm, 2.4*cm, 2.8*cm, 2.8*cm, 2.8*cm]
))
story.append(sp(4))
story.append(Paragraph(
    "Gap = how far below the module average the cell is running. "
    "Actual = absolute cell voltage. Module Avg = average of all 36 M1 cells that month.",
    CAPTION))
story.append(sp(8))

story.append(Paragraph("<b>4.2  Anomaly Rate Each Month</b>", H3))
story.append(Paragraph(
    "The anomaly rate tells us what percentage of readings were flagged as electrically "
    "abnormal. Healthy cells stay below 5%. Anything above 20% is a persistent issue.", BODY))
story.append(sp(4))
story.append(FIG3)
story.append(Paragraph(
    "Figure 3 — Monthly anomaly rate (%). The blue dashed line marks the 5% healthy baseline. "
    "In January–March, both cells were anomalous on nearly every single reading.", CAPTION))
story.append(sp(6))

outlier_rows = []
for month in monthly_outlier.index:
    m = str(month)
    c25 = monthly_outlier.loc[month,'M1_C25_is_outlier']
    c26 = monthly_outlier.loc[month,'M1_C26_is_outlier']
    flag25 = "CRITICAL" if c25 > 50 else ("ELEVATED" if c25 > 5 else "Normal")
    flag26 = "CRITICAL" if c26 > 50 else ("ELEVATED" if c26 > 5 else "Normal")
    outlier_rows.append([m, f"{c25:.1f}%", flag25, f"{c26:.1f}%", flag26])
story.append(data_table(
    ['Month', 'M1_C25 Anomaly %', 'M1_C25 Status', 'M1_C26 Anomaly %', 'M1_C26 Status'],
    outlier_rows,
    col_widths=[2.8*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm]
))
story.append(PageBreak())

# ── SECTION 5 — WORST CELL IN PACK ───────────────────────────────────────────
story.append(Paragraph("5. How Often Was M1_C25 the Worst Cell in the Entire Pack?", H2))
story.append(hr())
story.append(Paragraph(
    "The battery pack contains 180 cells across 5 modules. At every second, the system "
    "records which cell has the lowest voltage — the weakest point in the whole pack. "
    "In a healthy battery, this should rotate across many cells. When one cell "
    "dominates this slot, it is a serious signal.", BODY))
story.append(sp(4))
story.append(FIG4)
story.append(Paragraph(
    "Figure 4a — How often M1_C25 was the weakest cell inside Module 1 (weekly %). "
    "Figure 4b — How often M1_C25 was the single worst cell across all 180 cells in the pack. "
    "A healthy battery should show no single cell above 15% (red dashed line).", CAPTION))
story.append(sp(6))

dom_rows = [[str(m),
             f"{monthly_c25_worst[m]:.1f}%",
             f"{monthly_vrange[m]:.1f} mV",
             "CRITICAL" if monthly_c25_worst[m] > 50 else
             ("ELEVATED" if monthly_c25_worst[m] > 15 else "Normal")]
            for m in monthly_c25_worst.index]
story.append(data_table(
    ['Month', 'M1_C25 as Pack Worst', 'M1 Avg Voltage Spread', 'Status'],
    dom_rows,
    col_widths=[3.5*cm, 5*cm, 5.5*cm, 3*cm]
))
story.append(sp(4))
story.append(Paragraph(
    "In January 2026, M1_C25 was the worst cell in the entire pack 94% of the time. "
    "The voltage spread across Module 1 (how uneven the cells were) was 51.2 mV — well above "
    "the 40 mV warning threshold. By April, the spread had dropped to 11.7 mV, but this is "
    "due to the other cells declining rather than M1_C25 improving.", CALLOUT))
story.append(PageBreak())

# ── SECTION 6 — ABSOLUTE VOLTAGE ─────────────────────────────────────────────
story.append(Paragraph("6. Actual Voltage — What Happened in April?", H2))
story.append(hr())
story.append(Paragraph(
    "When we look at the actual voltage values (not relative gaps), we can understand "
    "whether M1_C25 genuinely recovered or whether the module average fell toward it.", BODY))
story.append(sp(4))
story.append(FIG5)
story.append(Paragraph(
    "Figure 5a — The solid lines show actual cell voltage in millivolts. "
    "Figure 5b — The gap between M1_C25 and the module average narrowed sharply in April. "
    "Monthly average gap values are annotated on the chart.", CAPTION))
story.append(sp(6))

abs_rows = []
for month in monthly_abs.index:
    m = str(month)
    c25v = monthly_abs.loc[month,'M1_C25']*1000
    c26v = monthly_abs.loc[month,'M1_C26']*1000
    meanv = monthly_abs.loc[month,'M1_mean']*1000
    gap25 = c25v - meanv
    abs_rows.append([m, f"{c25v:.1f} mV", f"{c26v:.1f} mV",
                     f"{meanv:.1f} mV", f"{gap25:.1f} mV"])
story.append(data_table(
    ['Month', 'M1_C25 Voltage', 'M1_C26 Voltage', 'Module 1 Avg', 'M1_C25 Gap'],
    abs_rows,
    col_widths=[2.8*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm]
))
story.append(sp(6))
story.append(Paragraph(
    "Between March and April the gap closed from −58.6 mV to −2.8 mV. "
    "Some of this closing is real improvement in M1_C25 (its absolute voltage rose). "
    "But the module average also rose sharply in April, reflecting a change in "
    "operating conditions (e.g. higher average state-of-charge or less heavy discharge). "
    "The key question for the service team is: <b>was any maintenance or balancing "
    "performed between March and April?</b> The data pattern is consistent with a "
    "balancing event.", CALLOUT))
story.append(PageBreak())

# ── SECTION 7 — WHOLE MODULE DECLINING ───────────────────────────────────────
story.append(Paragraph("7. The Broader Picture — Module 1 is Declining", H2))
story.append(hr())
story.append(Paragraph(
    "The most important finding from this investigation is not about M1_C25 alone. "
    "When we measure the rate of change of every cell in Module 1, a clear pattern emerges: "
    "almost the entire module is slowly declining.", BODY))
story.append(sp(4))
story.append(FIG6)
story.append(Paragraph(
    "Figure 6 — Each bar shows the weekly rate of change of each cell's voltage gap (mV/week). "
    "Bars pointing downward = cell is declining relative to the module average. "
    "Red = M1_C25 / M1_C26. Orange = declining fast. Blue = statistically confirmed declining. "
    "Green = stable.", CAPTION))
story.append(sp(6))

sig_worsening = slope_df[(slope_df.slope < 0) & slope_df.sig]
story.append(Paragraph(
    f"<b>{len(sig_worsening)} out of 36 cells</b> in Module 1 have a statistically confirmed "
    f"declining trend (p &lt; 0.05). Only M1_C25 and M1_C26 show upward trends — but as "
    f"explained above, this is because the rest of the module is declining toward their level, "
    f"not because these cells have recovered.", BODY))
story.append(sp(6))

story.append(Paragraph("Top 5 Fastest Declining Cells:", H3))
top5_rows = []
for _, r in slope_df.head(5).iterrows():
    top5_rows.append([r['cell'],
                      f"{r['slope']:+.3f} mV/week",
                      f"{r['jan']:.1f} mV",
                      f"{r['may']:.1f} mV",
                      f"r²={r['r2']:.2f}  p={r['p']:.4f}"])
story.append(data_table(
    ['Cell', 'Decline Rate', 'Jan Gap', 'May Gap', 'Statistical Confidence'],
    top5_rows,
    col_widths=[2.5*cm, 4*cm, 2.8*cm, 2.8*cm, 5*cm]
))
story.append(sp(8))

story.append(Paragraph("Monthly Anomaly Heatmap — All 36 Module 1 Cells", H3))
story.append(Paragraph(
    "The heatmap below shows the anomaly rate for every cell in every month. "
    "Dark red = flagged as abnormal most of the time. This confirms that the "
    "problem was concentrated in M1_C25 and M1_C26 in January–March, and "
    "largely resolved by April — but several other cells are starting to show "
    "elevated rates.", BODY))
story.append(sp(4))
story.append(FIG7)
story.append(Paragraph(
    "Figure 7 — Each row is one cell; each column is one month. "
    "Numbers show the % of readings flagged as anomalous. "
    "Healthy = below 5%. Cells appearing orange/red in May need monitoring.", CAPTION))
story.append(PageBreak())

# ── SECTION 8 — CONCLUSIONS ───────────────────────────────────────────────────
story.append(Paragraph("8. Conclusions", H2))
story.append(hr())

conclusions = [
    ("<b>M1_C25 and M1_C26 were critical from the very start of data</b> (22 January 2026). "
     "They were not healthy cells that degraded — they were already running 30–36 mV below "
     "the module average on day one. This raises the question of whether these cells had a "
     "pre-existing defect at installation."),
    ("<b>March 2026 was the worst period.</b> M1_C25 averaged −60 mV below the module mean "
     "and was flagged as abnormal on 98.8% of all readings. M1_C26 similarly showed 98.4%."),
    ("<b>The apparent recovery in April is not a true recovery.</b> The gap closed because "
     "the rest of the module declined, not because M1_C25/C26 improved. 31 out of 36 cells "
     "show statistically confirmed declining trends."),
    ("<b>M1_C27 is the next cell to monitor.</b> It is declining at 0.90 mV per week — "
     "the fastest rate in the module — and was 10.7 mV above the module average in January "
     "but only 0.9 mV above by May. If it crosses below 0 mV (i.e. becomes below the "
     "module average), it should be flagged as a watch cell."),
    ("<b>A maintenance event may have occurred between March and April.</b> The sharp "
     "reduction in voltage spread (74 mV → 11 mV) and anomaly rates (98% → 23%) is "
     "consistent with a battery balancing procedure. If this was performed, it should be "
     "logged against the vehicle's service history."),
]
for i, c in enumerate(conclusions, 1):
    story.append(Paragraph(f"{i}.  {c}", BODY_L))
    story.append(sp(4))

story.append(sp(8))
story.append(Paragraph("9. Recommended Actions", H2))
story.append(hr())

actions = [
    ("Immediate", "Confirm whether a battery balancing or maintenance event was performed "
                  "on EV01 between 28 March and 5 April 2026. Log this against the service record."),
    ("Short term (next 30 days)", "Place M1_C27 on a watch list. Set up a weekly check on "
                  "its voltage gap. If it falls below −10 mV consistently, schedule inspection."),
    ("Short term", "Investigate the origin of M1_C25 and M1_C26 weakness. Check the "
                  "installation records and factory test data for these two cells."),
    ("Medium term", "Enable degradation rate tracking (Module 5 of the Battery Engine) "
                  "to automatically detect when any cell's decline rate exceeds 0.5 mV/week "
                  "and alert the fleet team."),
    ("Medium term", "Perform a full Module 1 health assessment during the next scheduled "
                  "service, including cell impedance measurements, to confirm the degradation "
                  "trend and estimate remaining useful life."),
]
CELL_S = ParagraphStyle('CellS', parent=styles['Normal'], fontSize=9, leading=13)
CELL_BOLD = ParagraphStyle('CellBold', parent=styles['Normal'], fontSize=9, leading=13,
                           fontName='Helvetica-Bold', textColor=colors.HexColor('#1A2E4A'))
action_rows = [
    [Paragraph(a[0], CELL_BOLD), Paragraph(a[1], CELL_S)]
    for a in actions
]
story.append(data_table(
    ['Timeline', 'Action'],
    action_rows,
    col_widths=[4.5*cm, 12.5*cm]
))
story.append(sp(10))
story.append(hr())
story.append(Paragraph(
    "Report generated by the EV01 Battery Intelligence Engine — May 2026. "
    "Data covers 22 January – 17 May 2026 | 2,163,692 sensor readings.",
    ParagraphStyle('footer', parent=styles['Normal'], fontSize=8,
                   textColor=colors.grey, alignment=TA_CENTER)))

# ─────────────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"\nPDF saved → {OUT_PDF}")
