"""
Daily Battery Intelligence PDF Report
======================================
One chart per page. Every chart is annotated at key events.
Every chart page has a dedicated analysis section below it.

Page structure:
  1  — Cover (vehicle, date, title only)
  2  — Daily statistics summary (table, no chart)
  3  — SOC: actual vs predicted
  4  — SOC prediction error
  5  — Power output profile
  6  — Operating regime breakdown
  7  — Battery temperature profile
  8  — Motor (IGBT) temperature profile
  9  — Cell module voltages
  10 — Pack voltage imbalance trend
  11 — Alerts breakdown
  12 — Insights & recommendations

Usage:
    python scripts/generate_daily_report.py --date 2026-05-31
    python scripts/generate_daily_report.py --date 2026-05-31 --vehicle-id EV01
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import DATA_DIR, MACHINE_TYPES

MASTER_DIR = DATA_DIR / "analysis" / "master"
SOC_DIR    = PROJECT_ROOT / "outputs" / "soc_scores"
BAT_DIR    = PROJECT_ROOT / "outputs" / "battery_health"
OUT_DIR    = PROJECT_ROOT / "reports" / "daily_pdf"

# ── Design tokens ──────────────────────────────────────────────────────────────
C = {
    "navy":       "#1B3A5C",
    "blue":       "#2471A3",
    "blue_lt":    "#5DADE2",
    "green":      "#1E8449",
    "green_lt":   "#52BE80",
    "amber":      "#B7770D",
    "orange":     "#CA6F1E",
    "red":        "#C0392B",
    "red_lt":     "#E74C3C",
    "purple":     "#6C3483",
    "grey_dark":  "#2C3E50",
    "grey_mid":   "#626567",
    "grey_lt":    "#CCD1D1",
    "bg_chart":   "#F4F6F7",
    "bg_panel":   "#EAF2FF",
    "bg_warn":    "#FDEDEC",
    "white":      "#FFFFFF",
}

FONT = "DejaVu Sans"
plt.rcParams.update({
    "figure.facecolor":  C["white"],
    "axes.facecolor":    C["bg_chart"],
    "axes.edgecolor":    C["grey_lt"],
    "axes.linewidth":    0.7,
    "axes.labelcolor":   C["grey_dark"],
    "axes.labelsize":    9.5,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   C["navy"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       C["grey_mid"],
    "xtick.labelsize":   8.5,
    "ytick.color":       C["grey_mid"],
    "ytick.labelsize":   8.5,
    "grid.color":        "#DCE0E4",
    "grid.linewidth":    0.5,
    "legend.fontsize":   8.5,
    "legend.framealpha": 0.93,
    "legend.edgecolor":  C["grey_lt"],
    "font.family":       FONT,
    "font.size":         9,
    "text.color":        C["grey_dark"],
    "lines.linewidth":   1.4,
})

PAGE_W, PAGE_H = 11.69, 8.27   # A4 landscape

# Chart page layout (figure-fraction coordinates)
HDR_BOT  = 0.910   # header band bottom edge
CHT_TOP  = 0.895   # chart axes top
CHT_BOT  = 0.325   # chart axes bottom  — gap above panel prevents overlap
PNL_TOP  = 0.295   # analysis panel top
PNL_BOT  = 0.022   # analysis panel bottom
MARGIN_L = 0.07
MARGIN_R = 0.07


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_master(dt: str, vid: str) -> pd.DataFrame | None:
    p = MASTER_DIR / f"dt={dt}" / f"vehicle_id={vid}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_soc(dt: str, vid: str) -> pd.DataFrame | None:
    p = SOC_DIR / f"dt={dt}" / f"vehicle_id={vid}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_health(dt: str, vid: str) -> pd.DataFrame | None:
    p = BAT_DIR / f"dt={dt}" / f"scores_{vid}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_alerts(dt: str, vid: str) -> pd.DataFrame | None:
    p = BAT_DIR / f"dt={dt}" / f"alerts_{vid}.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ── Time axis helpers ─────────────────────────────────────────────────────────

def _hours(ts_series: pd.Series) -> np.ndarray:
    """Convert UTC timestamps to fractional hours since midnight of the first row."""
    t0 = ts_series.iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
    return ((ts_series - t0).dt.total_seconds() / 3600).values


def _fmt_time(ax: plt.Axes, max_h: float) -> None:
    """Label x-axis as HH:MM every 2 hours."""
    ticks = [h for h in range(0, int(max_h) + 2, 2) if h <= max_h + 0.1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(h):02d}:00" for h in ticks])
    ax.set_xlabel("Time of Day (HH:MM)")


# ── Low-level drawing helpers ──────────────────────────────────────────────────

def _new_fig() -> plt.Figure:
    return plt.figure(figsize=(PAGE_W, PAGE_H))


def _header(fig: plt.Figure, section: str, vid: str, machine: str,
            dt: str, pg: int, total: int) -> None:
    """Navy header band across full width."""
    ax_h = fig.add_axes([0, HDR_BOT, 1, 1 - HDR_BOT])
    ax_h.set_facecolor(C["navy"])
    ax_h.axis("off")
    ax_h.text(0.015, 0.72, section,
              fontsize=13, fontweight="bold", color=C["white"],
              va="center", transform=ax_h.transAxes)
    ax_h.text(0.015, 0.22, f"{vid}  ·  {machine}  ·  {dt}",
              fontsize=8.5, color="#AED6F1",
              va="center", transform=ax_h.transAxes)
    ax_h.text(0.985, 0.72, f"Page {pg} / {total}",
              fontsize=8, color=C["white"], ha="right",
              va="center", transform=ax_h.transAxes)
    ax_h.text(0.985, 0.22, "Battery Intelligence Platform",
              fontsize=7.5, color="#AED6F1", ha="right",
              va="center", transform=ax_h.transAxes)


def _chart_ax(fig: plt.Figure) -> plt.Axes:
    """Return a single full-width chart axes in the chart zone."""
    return fig.add_axes([MARGIN_L, CHT_BOT, 1 - MARGIN_L - MARGIN_R, CHT_TOP - CHT_BOT])


def _panel(fig: plt.Figure) -> plt.Axes:
    """Return the analysis panel axes below the chart."""
    ax = fig.add_axes([MARGIN_L, PNL_BOT, 1 - MARGIN_L - MARGIN_R, PNL_TOP - PNL_BOT])
    ax.set_facecolor(C["bg_panel"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="square,pad=0",
        transform=ax.transAxes,
        facecolor="none",
        edgecolor=C["blue"],
        linewidth=0.8,
    ))
    ax.text(0.012, 0.935, "Chart Analysis",
            fontsize=9.5, fontweight="bold", color=C["navy"],
            va="top", transform=ax.transAxes)
    return ax


def _write_panel(ax: plt.Axes, lines: list[dict]) -> None:
    """Write analysis text into the panel axes."""
    y = 0.865
    LINE_H = 0.100   # height per wrapped line
    GAP    = 0.025   # gap between entries

    for entry in lines:
        raw   = entry.get("text", "")
        bold  = entry.get("bold", False)
        warn  = entry.get("warn", False)
        color = C["red"] if warn else C["grey_dark"]
        size  = 8.5

        wrapped = textwrap.wrap(raw, width=132)
        block_h = len(wrapped) * LINE_H + GAP

        if warn:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.005, y - block_h + 0.02), 0.990, block_h + 0.01,
                boxstyle="round,pad=0.008",
                transform=ax.transAxes,
                facecolor=C["bg_warn"],
                edgecolor=C["red_lt"],
                linewidth=0.6,
            ))

        for wl in wrapped:
            ax.text(0.014, y, wl, fontsize=size,
                    fontweight="bold" if (bold or warn) else "normal",
                    color=color, va="top", transform=ax.transAxes)
            y -= LINE_H
        y -= GAP
        if y < 0.02:
            break


def _annotate(ax: plt.Axes, x, y_val: float, label: str,
              color: str = None, offset=(0, 20)) -> None:
    """Drop an annotated arrow marker on the chart at (x, y_val)."""
    col = color or C["red"]
    ax.annotate(
        label,
        xy=(x, y_val),
        xytext=(offset[0], offset[1]),
        textcoords="offset points",
        fontsize=7.5,
        color=col,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=col, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=col, linewidth=0.7, alpha=0.92),
    )


def _no_data_ax(ax: plt.Axes, msg: str = "No data available") -> None:
    ax.set_facecolor(C["white"])
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11,
            color=C["grey_mid"], style="italic", transform=ax.transAxes)


# ── Page 1 — Cover ─────────────────────────────────────────────────────────────

def page_cover(pdf: PdfPages, vid: str, machine: str, dt: str,
               total_pages: int) -> None:
    fig = _new_fig()

    # Top navy half
    ax_top = fig.add_axes([0, 0.45, 1, 0.55])
    ax_top.set_facecolor(C["navy"])
    ax_top.axis("off")

    ax_top.text(0.5, 0.78, "Daily Battery Intelligence Report",
                ha="center", va="center", fontsize=22, fontweight="bold",
                color=C["white"], transform=ax_top.transAxes)
    ax_top.text(0.5, 0.55, f"{vid}  —  {machine}",
                ha="center", va="center", fontsize=17,
                color="#AED6F1", transform=ax_top.transAxes)
    ax_top.text(0.5, 0.35, dt,
                ha="center", va="center", fontsize=15,
                color="#D6EAF8", transform=ax_top.transAxes)

    # Thin accent stripe
    ax_stripe = fig.add_axes([0, 0.435, 1, 0.018])
    ax_stripe.set_facecolor(C["blue_lt"])
    ax_stripe.axis("off")

    # Bottom white half
    ax_bot = fig.add_axes([0, 0, 1, 0.435])
    ax_bot.set_facecolor(C["white"])
    ax_bot.axis("off")
    ax_bot.text(0.5, 0.60, "Battery Intelligence Platform",
                ha="center", va="center", fontsize=12,
                color=C["grey_mid"], transform=ax_bot.transAxes)
    ax_bot.text(0.5, 0.38,
                f"Report generated: {date.today().isoformat()}  ·  Confidential",
                ha="center", va="center", fontsize=9,
                color=C["grey_lt"], transform=ax_bot.transAxes)
    ax_bot.text(0.5, 0.18,
                f"This report contains {total_pages} pages of operational data, "
                "performance analysis, and recommendations.",
                ha="center", va="center", fontsize=9,
                color=C["grey_mid"], transform=ax_bot.transAxes,
                style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 2 — Daily Statistics Summary ──────────────────────────────────────────

def page_stats(pdf: PdfPages, master: pd.DataFrame,
               soc_df, health_df, alerts_df,
               vid: str, machine: str, dt: str, pg: int, total: int) -> None:
    fig = _new_fig()
    _header(fig, "Daily Statistics Summary", vid, machine, dt, pg, total)

    ax = fig.add_axes([MARGIN_L, 0.04, 1 - MARGIN_L - MARGIN_R, HDR_BOT - 0.06])
    ax.set_facecolor(C["white"])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def _soc(col):
        if col in master.columns:
            s = master[col].dropna()
            s = s[(s >= 0) & (s <= 100)]
            return s
        return pd.Series(dtype=float)

    def _temp(col):
        if col in master.columns:
            t = master[col].dropna()
            return t[(t > -10) & (t < 80)]
        return pd.Series(dtype=float)

    def _mt():
        if "motor_temperature_c" in master.columns:
            t = master["motor_temperature_c"].dropna()
            return t[(t > -39) & (t < 174)]
        return pd.Series(dtype=float)

    soc  = _soc("soc_pct")
    pwr  = master["output_power_kw"].dropna() if "output_power_kw" in master.columns else pd.Series()
    bat  = _temp("avg_battery_temp_c")
    mot  = _mt()
    imb  = master["pack_voltage_range"].dropna() * 1000 if "pack_voltage_range" in master.columns else pd.Series()
    kwh  = master["total_kwh_consumed"].dropna() if "total_kwh_consumed" in master.columns else pd.Series()

    active_h = (pwr > 2).sum() / 3600 if not pwr.empty else 0

    mae_v = None
    if soc_df is not None and "y_soc_t_plus_300s" in soc_df.columns and "soc_pred_t300s" in soc_df.columns:
        mae_v = float(np.mean(np.abs(
            soc_df["y_soc_t_plus_300s"].astype(float) - soc_df["soc_pred_t300s"].astype(float))))

    avg_hs = None
    if health_df is not None and "health_score" in health_df.columns:
        avg_hs = health_df["health_score"].dropna().mean()

    n_alerts = len(alerts_df) if alerts_df is not None else 0
    n_trips  = master["trip_id"].dropna().nunique() if "trip_id" in master.columns else 0
    energy   = (kwh.iloc[-1] - kwh.iloc[0]) if len(kwh) > 1 else 0

    def fmt(val, unit="", dec=1, warn_above=None, warn_below=None):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—", C["grey_mid"]
        s = f"{val:.{dec}f}{unit}"
        color = C["grey_dark"]
        if warn_above is not None and val > warn_above:
            color = C["red"]
        if warn_below is not None and val < warn_below:
            color = C["red"]
        return s, color

    rows = [
        ("OPERATIONS",      None),
        ("Operating hours",           fmt(active_h, " h")),
        ("Trips detected",            (str(n_trips), C["grey_dark"])),
        ("Energy consumed",           fmt(energy, " kWh")),
        ("",                None),
        ("STATE OF CHARGE",  None),
        ("SOC — start of day",        fmt(soc.iloc[0] if not soc.empty else None, "%", 1)),
        ("SOC — end of day",          fmt(soc.iloc[-1] if not soc.empty else None, "%", 1)),
        ("SOC — minimum",             fmt(soc.min() if not soc.empty else None, "%", 1, warn_above=999, warn_below=20)),
        ("SOC — maximum",             fmt(soc.max() if not soc.empty else None, "%", 1)),
        ("SOC forecast MAE",          fmt(mae_v, "%", 3, warn_above=3.0)),
        ("",                None),
        ("POWER",            None),
        ("Peak power output",         fmt(pwr.max() if not pwr.empty else None, " kW", 1, warn_above=120)),
        ("Average power output",      fmt(pwr.mean() if not pwr.empty else None, " kW", 1)),
        ("",                None),
        ("TEMPERATURE",      None),
        ("Battery temp — average",    fmt(bat.mean() if not bat.empty else None, "°C", 1)),
        ("Battery temp — peak",       fmt(bat.max() if not bat.empty else None, "°C", 1, warn_above=45)),
        ("Motor (IGBT) temp — peak",  fmt(mot.max() if not mot.empty else None, "°C", 1, warn_above=80)),
        ("",                None),
        ("CELL HEALTH",      None),
        ("Voltage imbalance — avg",   fmt(imb.mean() if not imb.empty else None, " mV", 1, warn_above=50)),
        ("Voltage imbalance — peak",  fmt(imb.max() if not imb.empty else None, " mV", 1, warn_above=100)),
        ("Battery health score",      fmt(avg_hs, "/100", 1, warn_below=70)),
        ("",                None),
        ("ALERTS",           None),
        ("Total alert events",        (f"{n_alerts:,}", C["red"] if n_alerts > 5000 else C["grey_dark"])),
    ]

    col1_x, col2_x = 0.02, 0.52
    row_h  = 0.043
    y      = 0.960

    for item in rows:
        label, val = item
        if val is None:
            # Section header
            if label:
                ax.text(col1_x, y, label, fontsize=9, fontweight="bold",
                        color=C["navy"], va="top", transform=ax.transAxes)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (col1_x, y - 0.030), 0.96, 0.002,
                    boxstyle="square,pad=0", transform=ax.transAxes,
                    facecolor=C["navy"], edgecolor="none", alpha=0.25,
                ))
            y -= row_h * 0.9
            continue

        # Alternate row shading
        if int(y * 100) % int(row_h * 200) < int(row_h * 100):
            ax.add_patch(mpatches.FancyBboxPatch(
                (col1_x, y - 0.030), 0.96, row_h,
                boxstyle="square,pad=0", transform=ax.transAxes,
                facecolor="#F2F6FB", edgecolor="none",
            ))

        ax.text(col1_x + 0.01, y, label, fontsize=8.8, color=C["grey_dark"],
                va="top", transform=ax.transAxes)
        v_str, v_color = val
        ax.text(col2_x, y, v_str, fontsize=8.8, fontweight="bold",
                color=v_color, va="top", transform=ax.transAxes)

        # Warning icon
        if v_color == C["red"]:
            ax.text(col2_x + 0.12, y, "⚠ exceeds threshold",
                    fontsize=7.5, color=C["red"], va="top", transform=ax.transAxes,
                    style="italic")
        y -= row_h

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Generic chart page builder ─────────────────────────────────────────────────

def _chart_page(pdf, title, vid, machine, dt, pg, total, draw_fn, analysis_lines):
    fig = _new_fig()
    _header(fig, title, vid, machine, dt, pg, total)
    ax = _chart_ax(fig)
    draw_fn(ax)
    pan = _panel(fig)
    _write_panel(pan, analysis_lines)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 3 — SOC actual vs predicted ───────────────────────────────────────────

def draw_soc(ax, master, soc_df):
    if "soc_pct" not in master.columns or "timestamp" not in master.columns:
        _no_data_ax(ax)
        return

    hrs = _hours(master["timestamp"])
    soc = master["soc_pct"].copy()
    ax.plot(hrs, soc, color=C["blue"], linewidth=1.3, label="Actual SOC", zorder=3)
    ax.fill_between(hrs, soc, alpha=0.08, color=C["blue"])

    if soc_df is not None and "soc_pred_t300s" in soc_df.columns:
        pred_hrs = np.linspace(hrs[0], hrs[-1], len(soc_df))
        ax.plot(pred_hrs, soc_df["soc_pred_t300s"].values,
                color=C["orange"], linewidth=1.1, linestyle="--",
                label="Predicted SOC (+5 min)", alpha=0.85, zorder=2)

    ax.axhline(20, color=C["red"], linewidth=0.9, linestyle=":",
               alpha=0.8, label="Low SOC threshold (20%)")
    ax.axhline(80, color=C["green"], linewidth=0.9, linestyle=":",
               alpha=0.8, label="High SOC (80%)")

    valid = soc.dropna()
    if not valid.empty:
        _annotate(ax, hrs[valid.idxmin()], valid.min(),
                  f"Min SOC\n{valid.min():.1f}%", C["red"], (20, 25))
        _annotate(ax, hrs[valid.idxmax()], valid.max(),
                  f"Max SOC\n{valid.max():.1f}%", C["green"], (20, -30))

    crosses = (soc.shift(1) > 20) & (soc <= 20)
    if crosses.any():
        cx = crosses.idxmax()
        _annotate(ax, hrs[cx], 20, "Crossed 20%\nthreshold", C["red"], (-60, 25))

    ax.set_ylabel("State of Charge (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Battery State of Charge — Actual vs. 5-Minute-Ahead Forecast", pad=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    _fmt_time(ax, hrs[-1])


def soc_analysis(master, soc_df) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  The blue line is the battery's measured State of Charge (SOC) reported "
        "by the BMS at each second.  The orange dashed line is the model's prediction of where SOC will be "
        "5 minutes into the future.  The green and red dotted lines mark the 80% and 20% SOC thresholds."
    )})
    if "soc_pct" in master.columns:
        s = master["soc_pct"].dropna()
        if not s.empty:
            s0, s1 = s.iloc[0], s.iloc[-1]
            dir_ = "fell" if s1 < s0 else "rose"
            lines.append({"text": (
                f"Today's behaviour:  SOC opened at {s0:.1f}% and {dir_} to {s1:.1f}% by end of shift "
                f"— a net change of {abs(s1-s0):.1f} percentage points across the day.  "
                f"The shaded region under the SOC curve shows the total charge consumed visually."
            )})
            if s.min() < 20:
                lines.append({"text": (
                    f"⚠  SOC dropped below 20% (minimum was {s.min():.1f}%).  Deep discharge below 20% "
                    "stresses LFP cells and shortens pack life.  Annotated on the chart above.  "
                    "Recommend adjusting shift schedule so recharging begins before SOC reaches 25%."
                ), "warn": True})
    if soc_df is not None and "y_soc_t_plus_300s" in soc_df.columns and "soc_pred_t300s" in soc_df.columns:
        err  = (soc_df["y_soc_t_plus_300s"].astype(float) - soc_df["soc_pred_t300s"].astype(float)).abs()
        mae  = err.mean()
        p95  = err.quantile(0.95)
        grade = "excellent" if mae < 1.0 else ("good" if mae < 2.0 else "elevated — review data quality")
        lines.append({"text": (
            f"Forecast accuracy:  Mean Absolute Error = {mae:.3f}%  |  95th-percentile error = {p95:.2f}%.  "
            f"Performance is {grade}.  "
            "A well-calibrated model should stay below 1.5% MAE under normal operating conditions."
        )})
    return lines


# ── Page 4 — SOC Prediction Error ──────────────────────────────────────────────

def draw_soc_error(ax, soc_df):
    if soc_df is None or "y_soc_t_plus_300s" not in soc_df.columns:
        _no_data_ax(ax, "SOC prediction data not available")
        return

    err = (soc_df["y_soc_t_plus_300s"].astype(float) - soc_df["soc_pred_t300s"].astype(float)).abs()
    x   = np.arange(len(err))
    ax.plot(x, err.values, color=C["red_lt"], linewidth=0.9, alpha=0.8, label="Absolute error")
    ax.fill_between(x, err.values, alpha=0.12, color=C["red_lt"])

    mae = err.mean()
    ax.axhline(mae, color=C["orange"], linewidth=1.2, linestyle="--",
               label=f"Mean error: {mae:.3f}%")
    ax.axhline(3.0, color=C["red"], linewidth=0.9, linestyle=":",
               label="Action threshold (3%)", alpha=0.8)

    # Annotate peak error
    peak_idx = err.idxmax()
    _annotate(ax, peak_idx, err.max(),
              f"Peak error\n{err.max():.2f}%", C["red"], (20, 20))

    # Annotate sustained high-error region if p90 > 2
    p90 = err.quantile(0.90)
    if p90 > 2.0:
        high = (err > 2.0)
        first_high = high.idxmax() if high.any() else None
        if first_high:
            _annotate(ax, first_high, 2.0,
                      "Error > 2%\nstarts here", C["orange"], (-50, 25))

    ax.set_ylabel("Absolute Error (%)")
    ax.set_title("SOC Prediction Error Over Time (Absolute Value)", pad=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")


def soc_error_analysis(soc_df) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  This is the magnitude of the SOC model's prediction error at every "
        "timestep — i.e., how far off the 5-minute forecast was from what actually happened.  "
        "Lower is better.  The orange dashed line is the daily mean error.  "
        "The red dotted line at 3% is the action threshold above which the model's output should not be trusted."
    )})
    if soc_df is not None and "y_soc_t_plus_300s" in soc_df.columns:
        err  = (soc_df["y_soc_t_plus_300s"].astype(float) - soc_df["soc_pred_t300s"].astype(float)).abs()
        mae  = err.mean()
        p95  = err.quantile(0.95)
        pct_ok = (err < 1.5).mean() * 100
        lines.append({"text": (
            f"Error distribution:  {pct_ok:.0f}% of predictions had error below 1.5%.  "
            f"The 95th percentile error was {p95:.2f}% — meaning in the worst 5% of moments "
            f"the model was off by at least {p95:.2f} percentage points of SOC."
        )})
        if mae > 3.0:
            lines.append({"text": (
                f"⚠  Mean error of {mae:.2f}% exceeds the 3% action threshold.  "
                "This can be caused by BMS reset events, unusual duty cycles, or data quality issues.  "
                "The model may need retraining on recent data from this vehicle."
            ), "warn": True})
        else:
            lines.append({"text": (
                "Interpretation:  Spikes in error often coincide with sharp SOC transitions — "
                "sudden heavy loads, regen events, or BMS recalibration jumps.  "
                "Sustained high error (not just spikes) is the indicator to watch for model degradation."
            )})
    return lines


# ── Page 5 — Power Output Profile ──────────────────────────────────────────────

def draw_power(ax, master):
    if "output_power_kw" not in master.columns:
        _no_data_ax(ax)
        return

    hrs = _hours(master["timestamp"])
    pwr = master["output_power_kw"].fillna(0)

    ax.plot(hrs, pwr, color=C["orange"], linewidth=1.0, alpha=0.85, label="Output power (kW)")
    ax.fill_between(hrs, pwr, alpha=0.10, color=C["orange"])

    avg = pwr.mean()
    ax.axhline(avg, color=C["blue"], linewidth=1.1, linestyle="--",
               label=f"Daily average: {avg:.1f} kW")
    ax.axhline(100, color=C["red"], linewidth=0.9, linestyle=":",
               label="100 kW reference", alpha=0.8)

    peak_idx = pwr.idxmax()
    _annotate(ax, hrs[peak_idx], pwr.max(),
              f"Peak: {pwr.max():.1f} kW", C["red"], (15, 20))

    idle = pwr < 2
    if idle.any():
        groups       = idle.ne(idle.shift()).cumsum()
        idle_lengths = idle.groupby(groups).sum()
        if idle_lengths.max() > 300:
            longest_group = idle_lengths.idxmax()
            longest_start = idle[idle.groupby(groups).transform("idxmax") == idle.index].index
            start_idx = idle_lengths.index.get_loc(longest_group)
            _annotate(ax, hrs[start_idx], 0,
                      f"Idle: {idle_lengths.max()//60:.0f} min", C["grey_mid"], (30, 30))

    ax.set_ylabel("Power Output (kW)")
    ax.set_title("Battery Output Power Profile Throughout the Shift", pad=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    _fmt_time(ax, hrs[-1])


def power_analysis(master) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  Output power (kW) drawn from the battery at each second of the shift.  "
        "Peaks indicate heavy work cycles (lifting, digging, acceleration).  "
        "Flat sections near zero are idle periods.  Negative values (if present) indicate regenerative braking "
        "returning energy to the battery.  The peak is annotated with an arrow."
    )})
    if "output_power_kw" in master.columns:
        pwr = master["output_power_kw"].dropna()
        if not pwr.empty:
            pct_heavy = (pwr > 80).mean() * 100
            pct_idle  = (pwr < 2).mean() * 100
            lines.append({"text": (
                f"Load profile:  Peak demand was {pwr.max():.1f} kW.  Average over the shift was {pwr.mean():.1f} kW.  "
                f"{pct_heavy:.1f}% of operating time was spent above 80 kW (heavy discharge regime), "
                f"while {pct_idle:.1f}% of the shift was spent idle (< 2 kW).  "
                "High percentage of time in heavy discharge is the primary driver of battery stress."
            )})
            if pwr.max() > 100:
                lines.append({"text": (
                    f"⚠  Peak power of {pwr.max():.1f} kW was recorded today.  "
                    "Frequent high-current pulses above 100 kW increase lithium plating risk and accelerate "
                    "cell degradation.  Consider reviewing operator load cycles if this pattern persists."
                ), "warn": True})
    return lines


# ── Page 6 — Operating Regime ──────────────────────────────────────────────────

def draw_regime(ax, master):
    if "operating_regime" not in master.columns:
        _no_data_ax(ax, "Operating regime data not available")
        return

    counts = master["operating_regime"].value_counts()
    total  = counts.sum()
    palette = {
        "heavy_discharge":  C["red"],
        "light_discharge":  C["orange"],
        "idle":             C["grey_lt"],
        "light_regen":      C["blue_lt"],
        "heavy_regen":      C["green_lt"],
    }
    colors = [palette.get(r, C["grey_mid"]) for r in counts.index]
    labels = [f"{r.replace('_', ' ').title()}\n{cnt/total*100:.0f}%"
              for r, cnt in zip(counts.index, counts.values)]

    wedges, texts = ax.pie(
        counts.values, colors=colors, labels=labels,
        startangle=90, textprops={"fontsize": 9.5},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title("Operating Regime Distribution — Time Spent in Each Mode", pad=12)

    # Legend with absolute times
    legend_labels = [f"{r.replace('_',' ').title()}:  {cnt//60:.0f} min"
                     for r, cnt in zip(counts.index, counts.values)]
    ax.legend(wedges, legend_labels, loc="center left",
              bbox_to_anchor=(1.05, 0.5), fontsize=9)


def regime_analysis(master) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  The pie chart divides the entire shift into five operating regimes based "
        "on battery current direction and magnitude.  "
        "'Heavy Discharge' means the battery was under maximum load (e.g., excavating or lifting).  "
        "'Light Discharge' is moderate load.  'Idle' is engine-on but no significant load.  "
        "'Light/Heavy Regen' is energy being returned to the battery (downhill travel, lowering loads)."
    )})
    if "operating_regime" in master.columns:
        counts = master["operating_regime"].value_counts()
        total  = counts.sum()
        hd = counts.get("heavy_discharge", 0)
        idle = counts.get("idle", 0)
        lines.append({"text": (
            f"Today's usage pattern:  The vehicle spent {hd/total*100:.0f}% of its shift in heavy discharge "
            f"and {idle/total*100:.0f}% idle.  "
            "A high idle percentage suggests opportunities for engine-off or reduced-load idle protocols "
            "to extend battery life and reduce unnecessary energy draw."
        )})
        if hd / total > 0.40:
            lines.append({"text": (
                f"⚠  {hd/total*100:.0f}% of the shift was spent in heavy discharge — above the recommended 40% guideline.  "
                "Sustained heavy operation accelerates both cell degradation and thermal stress.  "
                "Consider duty cycle management or scheduled rest periods."
            ), "warn": True})
    return lines


# ── Page 7 — Battery Temperature ───────────────────────────────────────────────

def draw_battery_temp(ax, master):
    has_any = any(c in master.columns for c in
                  ("avg_battery_temp_c", "max_battery_temp_c", "min_battery_temp_c"))
    if not has_any:
        _no_data_ax(ax)
        return

    hrs = _hours(master["timestamp"])
    for col, label, color, lw in [
        ("max_battery_temp_c", "Max cell temp", C["red_lt"],  1.0),
        ("avg_battery_temp_c", "Avg pack temp", C["blue"],    1.4),
        ("min_battery_temp_c", "Min cell temp", C["green_lt"],1.0),
    ]:
        if col in master.columns:
            t = master[col].where((master[col] > -10) & (master[col] < 80))
            ax.plot(hrs, t, color=color, linewidth=lw, label=label, alpha=0.9)

    if "max_battery_temp_c" in master.columns and "min_battery_temp_c" in master.columns:
        tmax = master["max_battery_temp_c"].where((master["max_battery_temp_c"] > -10) & (master["max_battery_temp_c"] < 80))
        tmin = master["min_battery_temp_c"].where((master["min_battery_temp_c"] > -10) & (master["min_battery_temp_c"] < 80))
        ax.fill_between(hrs, tmin, tmax, alpha=0.07, color=C["blue"], label="Min–Max band")

    ax.axhline(45, color=C["red"], linewidth=1.0, linestyle=":",
               label="Warning threshold (45°C)", alpha=0.85)

    if "avg_battery_temp_c" in master.columns:
        t = master["avg_battery_temp_c"].where((master["avg_battery_temp_c"] > -10) & (master["avg_battery_temp_c"] < 80))
        if t.dropna().any():
            _annotate(ax, hrs[t.idxmax()], t.max(),
                      f"Peak avg\n{t.max():.1f}°C", C["red"], (15, 20))

        crosses = (master["avg_battery_temp_c"].shift(1) < 45) & (master["avg_battery_temp_c"] >= 45)
        if crosses.any():
            _annotate(ax, hrs[crosses.idxmax()], 45,
                      "Crossed 45°C", C["red"], (-70, 25))

    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Battery Pack Temperature Profile — Min / Avg / Max Cell Temperature", pad=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    _fmt_time(ax, hrs[-1])


def battery_temp_analysis(master) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  The blue line is average pack temperature.  The red and green lines are "
        "the hottest and coolest individual cells respectively.  The shaded band between them shows the "
        "thermal spread across all cells.  A wide shaded band means cells are at very different temperatures, "
        "which can cause uneven ageing.  The red dotted line at 45°C is the sustained-operation warning threshold."
    )})
    if "avg_battery_temp_c" in master.columns:
        t = master["avg_battery_temp_c"].dropna()
        t = t[(t > -10) & (t < 80)]
        if not t.empty:
            lines.append({"text": (
                f"Today's thermal profile:  Average pack temperature ranged from {t.min():.1f}°C to "
                f"{t.max():.1f}°C, with a mean of {t.mean():.1f}°C across the shift.  "
                + ("Pack temperature stayed within the optimal 20–45°C operating range throughout the day."
                   if t.max() <= 45 else
                   f"Temperature exceeded 45°C — see annotation on chart.  "
                   "Check coolant flow rate and inspect thermal management system.")
            )})
            if t.max() > 45:
                lines.append({"text": (
                    f"⚠  Battery temperature peaked at {t.max():.1f}°C.  "
                    "Sustained temperatures above 45°C accelerate electrolyte decomposition in LFP cells, "
                    "reducing both calendar life and cycle life.  "
                    "Inspect cooling system, reduce sustained heavy discharge during high-ambient conditions."
                ), "warn": True})
    return lines


# ── Page 8 — Motor (IGBT) Temperature ─────────────────────────────────────────

def draw_motor_temp(ax, master):
    if "motor_temperature_c" not in master.columns:
        _no_data_ax(ax, "Motor temperature data not available")
        return

    hrs = _hours(master["timestamp"])
    mt  = master["motor_temperature_c"].where(
        (master["motor_temperature_c"] > -39) & (master["motor_temperature_c"] < 174))

    ax.plot(hrs, mt, color=C["purple"], linewidth=1.3, alpha=0.88, label="IGBT temperature")
    ax.fill_between(hrs, mt, alpha=0.08, color=C["purple"])
    ax.axhline(80, color=C["red"], linewidth=1.0, linestyle=":",
               label="Warning threshold (80°C)", alpha=0.85)
    ax.axhline(60, color=C["amber"], linewidth=0.9, linestyle="--",
               label="Caution threshold (60°C)", alpha=0.75)

    valid = mt.dropna()
    if not valid.empty:
        _annotate(ax, hrs[valid.idxmax()], valid.max(),
                  f"Peak IGBT\n{valid.max():.1f}°C", C["red"], (15, 20))
        crosses = (mt.shift(1) < 60) & (mt >= 60)
        if crosses.any():
            _annotate(ax, hrs[crosses.idxmax()], 60,
                      "Crossed 60°C", C["amber"], (-75, -30))

    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Motor Inverter (IGBT) Temperature — Power Transistor Heat Profile", pad=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    _fmt_time(ax, hrs[-1])


def motor_temp_analysis(master) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  The motor temperature sensor (EDS object 0x6074) measures the "
        "IGBT (Insulated Gate Bipolar Transistor) temperature — the power switching transistor inside "
        "the inverter, NOT the motor winding temperature.  The IGBT is the thermal bottleneck for "
        "high-power operation.  Caution threshold is 60°C; warning threshold is 80°C."
    )})
    if "motor_temperature_c" in master.columns:
        mt = master["motor_temperature_c"].dropna()
        mt = mt[(mt > -39) & (mt < 174)]
        if not mt.empty:
            lines.append({"text": (
                f"Today's IGBT thermal profile:  Temperature ranged from {mt.min():.1f}°C to "
                f"{mt.max():.1f}°C with an average of {mt.mean():.1f}°C.  "
                + ("IGBT temperatures remained within safe operating limits throughout the shift."
                   if mt.max() <= 80 else
                   f"Temperature exceeded the 80°C warning threshold — thermal derating of the inverter may have been triggered.")
            )})
            if mt.max() > 80:
                lines.append({"text": (
                    f"⚠  IGBT temperature reached {mt.max():.1f}°C.  "
                    "Sustained operation above 80°C risks inverter module degradation and reduced power output.  "
                    "Verify inverter cooling fins are unobstructed, coolant level is adequate, "
                    "and check for any blockages in the thermal circuit."
                ), "warn": True})
    return lines


# ── Page 9 — Cell Module Voltages ──────────────────────────────────────────────

def draw_module_voltages(ax, master):
    modules = [1, 2, 3, 4, 5]
    v_means = [master[f"module_{m}_voltage_mean"].dropna().mean()
               if f"module_{m}_voltage_mean" in master.columns else np.nan for m in modules]

    if all(np.isnan(v) for v in v_means):
        _no_data_ax(ax, "Cell voltage data not available")
        return

    labels  = [f"Module {m}" for m in modules]
    avg_all = np.nanmean(v_means)

    bar_colors = []
    for v in v_means:
        if np.isnan(v):
            bar_colors.append(C["grey_lt"])
        elif abs(v - avg_all) * 1000 > 20:
            bar_colors.append(C["red_lt"])
        else:
            bar_colors.append(C["blue"])

    bars = ax.bar(labels, [v if not np.isnan(v) else 0 for v in v_means],
                  color=bar_colors, alpha=0.85, edgecolor="white",
                  width=0.55, zorder=3)

    ax.axhline(avg_all, color=C["red"], linewidth=1.2, linestyle="--",
               label=f"Pack average: {avg_all:.4f} V", zorder=4)

    # Annotate weakest and strongest modules
    valid = [(i, v) for i, v in enumerate(v_means) if not np.isnan(v)]
    if valid:
        weakest  = min(valid, key=lambda x: x[1])
        strongest = max(valid, key=lambda x: x[1])
        _annotate(ax, weakest[0], weakest[1],
                  f"Weakest\n{weakest[1]:.4f} V", C["red"], (0, -35))
        if weakest[0] != strongest[0]:
            _annotate(ax, strongest[0], strongest[1],
                      f"Strongest\n{strongest[1]:.4f} V", C["green"], (0, 25))

    rng = max(abs(v - avg_all) for v in v_means if not np.isnan(v))
    ax.set_ylim(avg_all - max(rng * 2.5, 0.003), avg_all + max(rng * 2.5, 0.003))
    ax.set_ylabel("Average Cell Voltage (V)")
    ax.set_title("Average Cell Voltage per Module — Daily Mean", pad=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")


def module_voltage_analysis(master) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  Each bar is the daily average cell voltage for one of the 5 battery modules.  "
        "In a healthy pack all bars should be nearly equal.  "
        "The dashed red line is the pack-wide average.  "
        "Bars coloured red deviate by more than 20 mV from the pack average — those modules warrant attention.  "
        "The weakest and strongest modules are annotated with arrows."
    )})
    modules = [1, 2, 3, 4, 5]
    v_means = [master[f"module_{m}_voltage_mean"].dropna().mean()
               if f"module_{m}_voltage_mean" in master.columns else np.nan for m in modules]
    valid = [(m, v) for m, v in zip(modules, v_means) if not np.isnan(v)]
    if valid:
        weakest  = min(valid, key=lambda x: x[1])
        strongest = max(valid, key=lambda x: x[1])
        spread_mv = (strongest[1] - weakest[1]) * 1000
        lines.append({"text": (
            f"Voltage balance today:  Module {weakest[0]} had the lowest average voltage "
            f"({weakest[1]:.4f} V) and Module {strongest[0]} the highest ({strongest[1]:.4f} V).  "
            f"The inter-module spread is {spread_mv:.1f} mV.  "
            + ("Inter-module balance is healthy — all modules are tracking closely."
               if spread_mv < 20 else
               f"A spread of {spread_mv:.1f} mV indicates the modules are diverging — "
               "likely caused by capacity fade in the weaker modules.")
        )})
        if spread_mv > 50:
            lines.append({"text": (
                f"⚠  Inter-module voltage spread of {spread_mv:.1f} mV is above the 50 mV concern threshold.  "
                "This level of divergence suggests at least one module is not holding charge as effectively "
                "as the others.  Schedule a capacity test on the weakest module and check BMS balancing logs."
            ), "warn": True})
    return lines


# ── Page 10 — Pack Voltage Imbalance Trend ─────────────────────────────────────

def draw_imbalance(ax, master):
    if "pack_voltage_range" not in master.columns:
        _no_data_ax(ax, "Imbalance data not available")
        return

    hrs_all = _hours(master["timestamp"])
    imb     = master["pack_voltage_range"].dropna() * 1000
    hrs_imb = hrs_all[imb.index]

    ax.plot(hrs_imb, imb.values, color=C["red_lt"], linewidth=1.1, alpha=0.85, label="Pack voltage range (mV)")
    ax.fill_between(hrs_imb, imb.values, alpha=0.10, color=C["red_lt"])

    ax.axhline(imb.mean(), color=C["blue"], linewidth=1.1, linestyle="--",
               label=f"Daily mean: {imb.mean():.1f} mV")
    ax.axhline(50, color=C["amber"], linewidth=0.9, linestyle=":",
               label="Warning threshold (50 mV)", alpha=0.85)
    ax.axhline(100, color=C["red"], linewidth=0.9, linestyle=":",
               label="Critical threshold (100 mV)", alpha=0.85)

    peak_loc = imb.idxmax()
    _annotate(ax, hrs_all[peak_loc], imb.max(),
              f"Peak: {imb.max():.1f} mV", C["red"], (15, 20))

    ax.set_ylabel("Voltage Imbalance (mV)")
    ax.set_title("Pack Voltage Imbalance Over Time — Max − Min Cell Voltage", pad=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    _fmt_time(ax, hrs_all[-1])


def imbalance_analysis(master) -> list[dict]:
    lines = []
    lines.append({"text": (
        "What this chart shows:  Pack voltage range is the difference between the highest and lowest "
        "individual cell voltage in the entire pack at each second.  "
        "It measures how far apart the cells have drifted in voltage.  "
        "A low, stable line means cells are well-balanced.  Spikes indicate moments when one cell "
        "is significantly ahead or behind the rest — often during heavy discharge or charge."
    )})
    if "pack_voltage_range" in master.columns:
        imb = master["pack_voltage_range"].dropna() * 1000
        if not imb.empty:
            pct_above_50 = (imb > 50).mean() * 100
            lines.append({"text": (
                f"Imbalance summary:  Daily average imbalance was {imb.mean():.1f} mV, "
                f"peaking at {imb.max():.1f} mV.  "
                f"{pct_above_50:.1f}% of the shift had imbalance above the 50 mV warning threshold.  "
                "Imbalance tends to be highest during peak load — this is normal and expected."
            )})
            if imb.mean() > 50:
                lines.append({"text": (
                    f"⚠  Average imbalance of {imb.mean():.1f} mV is above the 50 mV sustained warning level.  "
                    "When average (not just peak) imbalance is elevated, the BMS balancing circuit is not "
                    "keeping up with cell divergence.  Recommend a full balance charge at low C-rate."
                ), "warn": True})
    return lines


# ── Page 11 — Alerts ───────────────────────────────────────────────────────────

ALERT_DESC = {
    "HIGH_CURRENT_STRESS":            "Battery current exceeded the safe continuous rating. Repeated events accelerate electrode degradation.",
    "HIGH_CURRENT_WITH_HIGH_TEMP":    "Simultaneous high current AND high temperature — the most damaging stress condition for a lithium pack.",
    "TEMP_IMBALANCE_CRITICAL":        "Temperature gradient between modules exceeded critical bounds. Indicates uneven heat distribution.",
    "TEMP_IMBALANCE_WARNING":         "Thermal gradient approaching critical levels. Inspect cooling channels between modules.",
    "VOLTAGE_IMBALANCE_CRITICAL":     "Cell voltage divergence exceeded 100 mV. Immediate balancing charge is recommended.",
    "VOLTAGE_IMBALANCE_WARNING":      "Cell voltage divergence approaching 50 mV. Schedule a balancing cycle.",
    "BMS_OR_CELL_DATA_QUALITY_ISSUE": "BMS reported missing, zero-filled or inconsistent cell data. May affect health score accuracy.",
    "WEAK_MODULE_THERMAL":            "One or more modules sustained elevated temperature relative to pack average.",
}


def draw_alerts(ax, alerts_df):
    if alerts_df is None or alerts_df.empty or "rule_id" not in alerts_df.columns:
        _no_data_ax(ax, "No alerts triggered today — all systems within normal bounds.")
        return

    counts = alerts_df["rule_id"].value_counts()
    palette = [C["red"], C["orange"], C["amber"], C["purple"],
               C["blue_lt"], C["green_lt"], C["grey_mid"], C["grey_lt"]]
    colors  = palette[:len(counts)]
    labels  = [r.replace("_", " ").title() for r in counts.index]

    bars = ax.barh(labels, counts.values, color=colors, alpha=0.85,
                   edgecolor="white", height=0.60)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + counts.max() * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color=C["grey_dark"])

    # Annotate the top alert
    top_bar = bars[0]
    _annotate(ax, counts.values[0],
              top_bar.get_y() + top_bar.get_height() / 2,
              f"Most frequent\nalert type", C["red"], (20, 25))

    ax.set_xlabel("Number of Events")
    ax.set_title("Alert Events by Type — Full Day Count", pad=8)
    ax.set_xlim(0, counts.max() * 1.18)
    ax.grid(True, axis="x")
    ax.invert_yaxis()


def alerts_analysis(alerts_df) -> list[dict]:
    lines = []
    if alerts_df is None or alerts_df.empty:
        lines.append({"text": "No alerts were triggered today. All monitored parameters remained within defined thresholds."})
        return lines

    lines.append({"text": (
        "What this chart shows:  Each bar is the total count of rule-engine events for one alert type "
        "across the entire shift.  Alert counts are row-level — one event lasting 10 minutes at 1 Hz "
        "produces 600 alert rows.  Focus on which alert types dominate, not the raw count alone."
    )})
    n = len(alerts_df)
    lines.append({"text": f"Total alert events today: {n:,}."})

    if "rule_id" in alerts_df.columns:
        top = alerts_df["rule_id"].value_counts().head(4)
        for rule, cnt in top.items():
            desc = ALERT_DESC.get(rule, "See rule documentation.")
            pct  = cnt / n * 100
            is_critical = rule in ("HIGH_CURRENT_WITH_HIGH_TEMP", "VOLTAGE_IMBALANCE_CRITICAL",
                                   "TEMP_IMBALANCE_CRITICAL")
            lines.append({"text": f"{rule.replace('_', ' ')} ({pct:.0f}%):  {desc}",
                          "warn": is_critical})
    return lines


# ── Page 12 — Insights ─────────────────────────────────────────────────────────

def generate_insights(master, soc_df, health_df, alerts_df, vid, machine, dt):
    items = []

    if "output_power_kw" in master.columns:
        h = (master["output_power_kw"] > 2).sum() / 3600
        items.append(f"The vehicle was operationally active for approximately {h:.1f} hours today (power > 2 kW).")

    if "soc_pct" in master.columns:
        s = master["soc_pct"].dropna()
        if not s.empty:
            dir_ = "fell" if s.iloc[-1] < s.iloc[0] else "rose"
            items.append(
                f"SOC {dir_} from {s.iloc[0]:.1f}% at start of shift to {s.iloc[-1]:.1f}% at end — "
                f"a range of {s.min():.1f}% to {s.max():.1f}% across the day.")
            if s.min() < 20:
                items.append(
                    f"⚠  SOC reached a low of {s.min():.1f}% — deep discharge below 20% degrades LFP cells.  "
                    "Recommend adjusting recharge schedule.")

    if soc_df is not None and "y_soc_t_plus_300s" in soc_df.columns:
        mae = float(np.mean(np.abs(
            soc_df["y_soc_t_plus_300s"].astype(float) - soc_df["soc_pred_t300s"].astype(float))))
        grade = "Excellent" if mae < 1.0 else ("Good" if mae < 2.0 else "Elevated")
        items.append(f"SOC forecast accuracy today: MAE = {mae:.3f}%  ({grade}).  Threshold for good performance: < 1.5%.")
        if mae > 3.0:
            items.append(f"⚠  SOC error of {mae:.2f}% is above the 3% action threshold.  Model may need retraining.")

    if "output_power_kw" in master.columns:
        pwr = master["output_power_kw"].dropna()
        if not pwr.empty:
            items.append(f"Peak power: {pwr.max():.1f} kW.  Average demand: {pwr.mean():.1f} kW.")
            if pwr.max() > 100:
                items.append(f"⚠  Peak of {pwr.max():.1f} kW recorded — frequent high-current pulses accelerate cell degradation.")

    if "total_kwh_consumed" in master.columns:
        kwh = master["total_kwh_consumed"].dropna()
        if len(kwh) > 1:
            d = kwh.iloc[-1] - kwh.iloc[0]
            if d > 0:
                items.append(f"Total energy consumed today: {d:.1f} kWh.")

    if "avg_battery_temp_c" in master.columns:
        t = master["avg_battery_temp_c"].dropna()
        t = t[(t > -10) & (t < 80)]
        if not t.empty:
            items.append(f"Battery temperature: avg {t.mean():.1f}°C, peak {t.max():.1f}°C.")
            if t.max() > 45:
                items.append(f"⚠  Battery temperature peaked at {t.max():.1f}°C — above 45°C safe operating limit.")

    if "motor_temperature_c" in master.columns:
        mt = master["motor_temperature_c"].dropna()
        mt = mt[(mt > -39) & (mt < 174)]
        if not mt.empty:
            items.append(f"Motor (IGBT) temperature: avg {mt.mean():.1f}°C, peak {mt.max():.1f}°C.")
            if mt.max() > 80:
                items.append(f"⚠  IGBT temperature reached {mt.max():.1f}°C — inverter thermal derating risk.")

    if "pack_voltage_range" in master.columns:
        imb = master["pack_voltage_range"].dropna() * 1000
        if not imb.empty:
            items.append(f"Cell voltage imbalance: avg {imb.mean():.1f} mV, max {imb.max():.1f} mV.")
            if imb.mean() > 50:
                items.append(f"⚠  Sustained average imbalance of {imb.mean():.0f} mV — recommend balance charge.")

    if "worst_cell_id" in master.columns:
        worst = master["worst_cell_id"].dropna().mode()
        if not worst.empty and str(worst.iloc[0]) not in ("nan", "None", ""):
            items.append(f"Most frequently flagged weak cell: {worst.iloc[0]}.")

    if health_df is not None and "health_score" in health_df.columns:
        hs = health_df["health_score"].dropna()
        if not hs.empty:
            items.append(f"Battery health score: avg {hs.mean():.1f}/100  (min {hs.min():.1f}/100).")
            if hs.mean() < 70:
                items.append(f"⚠  Health score ({hs.mean():.0f}/100) below 70 — schedule detailed inspection.")

    if alerts_df is not None and not alerts_df.empty:
        items.append(f"Total alert events: {len(alerts_df):,}.")

    return items or ["Insufficient data to generate insights."]


def page_insights(pdf, master, soc_df, health_df, alerts_df,
                  vid, machine, dt, pg, total):
    insights = generate_insights(master, soc_df, health_df, alerts_df, vid, machine, dt)
    fig = _new_fig()
    _header(fig, "Insights & Recommendations", vid, machine, dt, pg, total)

    ax = fig.add_axes([MARGIN_L, 0.03, 1 - MARGIN_L - MARGIN_R, HDR_BOT - 0.05])
    ax.set_facecolor(C["bg_panel"])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.015, 0.968, "Full Operational Narrative — What Happened Today",
            fontsize=11, fontweight="bold", color=C["navy"], va="top",
            transform=ax.transAxes)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.015, 0.935), 0.40, 0.003,
        boxstyle="square,pad=0", transform=ax.transAxes,
        facecolor=C["blue"], edgecolor="none",
    ))

    y = 0.900
    for item in insights:
        is_warn = item.startswith("⚠")
        color   = C["red"] if is_warn else C["grey_dark"]
        weight  = "bold" if is_warn else "normal"

        if is_warn:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.010, y - 0.045), 0.978, 0.068,
                boxstyle="round,pad=0.008", transform=ax.transAxes,
                facecolor=C["bg_warn"], edgecolor=C["red_lt"],
                linewidth=0.6, alpha=0.9,
            ))

        for wl in textwrap.wrap(item.lstrip(), 125):
            ax.text(0.020, y, wl, fontsize=9.0, fontweight=weight,
                    color=color, va="top", transform=ax.transAxes)
            y -= 0.040
        y -= 0.015
        if y < 0.02:
            break

    fig.text(0.5, 0.008,
             f"Confidential  ·  Battery Intelligence Platform  ·  {vid} ({machine})  ·  {dt}",
             ha="center", fontsize=7.5, color=C["grey_mid"])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Orchestrator ────────────────────────────────────────────────────────────────

def generate_report(dt: str, vid: str) -> Path | None:
    machine   = MACHINE_TYPES.get(vid, "Unknown")
    master    = load_master(dt, vid)
    if master is None or master.empty:
        print(f"    ✗  No master data for {vid} on {dt}")
        return None

    soc_df    = load_soc(dt, vid)
    health_df = load_health(dt, vid)
    alerts_df = load_alerts(dt, vid)

    TOTAL = 12
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{vid}_{dt}.pdf"

    with PdfPages(str(out_path)) as pdf:
        d = pdf.infodict()
        d["Title"]  = f"Daily Battery Intelligence Report — {vid} {dt}"
        d["Author"] = "Battery Intelligence Platform"

        page_cover(pdf, vid, machine, dt, TOTAL)
        page_stats(pdf, master, soc_df, health_df, alerts_df, vid, machine, dt, 2, TOTAL)

        _chart_page(pdf, "State of Charge — Actual vs. Predicted",
                    vid, machine, dt, 3, TOTAL,
                    lambda ax: draw_soc(ax, master, soc_df),
                    soc_analysis(master, soc_df))

        _chart_page(pdf, "SOC Prediction Error Analysis",
                    vid, machine, dt, 4, TOTAL,
                    lambda ax: draw_soc_error(ax, soc_df),
                    soc_error_analysis(soc_df))

        _chart_page(pdf, "Power Output Profile",
                    vid, machine, dt, 5, TOTAL,
                    lambda ax: draw_power(ax, master),
                    power_analysis(master))

        _chart_page(pdf, "Operating Regime Distribution",
                    vid, machine, dt, 6, TOTAL,
                    lambda ax: draw_regime(ax, master),
                    regime_analysis(master))

        _chart_page(pdf, "Battery Temperature Profile",
                    vid, machine, dt, 7, TOTAL,
                    lambda ax: draw_battery_temp(ax, master),
                    battery_temp_analysis(master))

        _chart_page(pdf, "Motor Inverter (IGBT) Temperature",
                    vid, machine, dt, 8, TOTAL,
                    lambda ax: draw_motor_temp(ax, master),
                    motor_temp_analysis(master))

        _chart_page(pdf, "Cell Module Voltage Comparison",
                    vid, machine, dt, 9, TOTAL,
                    lambda ax: draw_module_voltages(ax, master),
                    module_voltage_analysis(master))

        _chart_page(pdf, "Pack Voltage Imbalance Trend",
                    vid, machine, dt, 10, TOTAL,
                    lambda ax: draw_imbalance(ax, master),
                    imbalance_analysis(master))

        _chart_page(pdf, "Alert & Anomaly Breakdown",
                    vid, machine, dt, 11, TOTAL,
                    lambda ax: draw_alerts(ax, alerts_df),
                    alerts_analysis(alerts_df))

        page_insights(pdf, master, soc_df, health_df, alerts_df,
                      vid, machine, dt, 12, TOTAL)

    size_kb = out_path.stat().st_size // 1024
    print(f"    ✓  {out_path}  ({size_kb} KB, {TOTAL} pages)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",       default=None)
    ap.add_argument("--vehicle-id", default=None, dest="vehicle_id")
    args = ap.parse_args()

    dt       = args.date or (date.today() - timedelta(days=1)).isoformat()
    vehicles = [args.vehicle_id] if args.vehicle_id else sorted(MACHINE_TYPES.keys())

    print(f"\nDaily PDF Report  ·  {dt}")
    for vid in vehicles:
        print(f"  Generating: {vid}")
        generate_report(dt, vid)
    print("\nDone — reports/daily_pdf/")


if __name__ == "__main__":
    main()
