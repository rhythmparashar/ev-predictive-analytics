"""
SOC Prediction Visualisation — drop-in replacement for the SOC chart section.
Paste this entire file as dashboard/soc_charts.py and import into app.py.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Shared layout base ────────────────────────────────────────
BASE = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0f1424",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    margin=dict(l=16, r=16, t=48, b=16),
    xaxis=dict(gridcolor="#1a2035", showgrid=True, zeroline=False, tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1a2035", showgrid=True, zeroline=False, tickfont=dict(size=11)),
    legend=dict(
        bgcolor="rgba(15,20,36,0.9)",
        bordercolor="#1e2640",
        borderwidth=1,
        font=dict(size=11),
        x=0.01, y=0.99,
    ),
    hoverlabel=dict(bgcolor="#111827", bordercolor="#1e2640", font_size=12, font_family="DM Mono"),
)

# ── Colours ───────────────────────────────────────────────────
C_ACTUAL    = "#60a5fa"   # blue — actual SOC
C_PRED      = "#f59e0b"   # amber — predicted SOC
C_ERROR_POS = "#ef4444"   # red — over-prediction fill
C_ERROR_NEG = "#22c55e"   # green — under-prediction fill
C_REGEN     = "#a78bfa"   # purple — regen events
C_FAULT     = "#fb923c"   # orange — fault markers
C_RATE      = "#34d399"   # emerald — SOC rate of change


def _add_fault_markers(fig, df, row=1, col=1):
    """Add vertical lines at fault moments."""
    faults = df[df["fault_any"] == 1] if "fault_any" in df.columns else pd.DataFrame()
    if faults.empty:
        return
    # cluster fault spans instead of one line per row
    faults = faults.copy().sort_values("timestamp")
    faults["gap"] = faults["timestamp"].diff().dt.total_seconds().gt(10).cumsum()
    for _, span in faults.groupby("gap"):
        fig.add_vrect(
            x0=span["timestamp"].iloc[0], x1=span["timestamp"].iloc[-1],
            fillcolor="rgba(251,146,60,0.08)", line_width=0,
            row=row, col=col,
        )
    # Single legend entry
    fig.add_trace(go.Scatter(
        x=[faults["timestamp"].iloc[0]], y=[None],
        mode="markers",
        marker=dict(symbol="line-ns", size=10, color=C_FAULT, line=dict(width=2, color=C_FAULT)),
        name="⚠ Active Fault",
        showlegend=True,
    ), row=row, col=col)


def _add_regen_markers(fig, df, row=1, col=1):
    """Shade regen zones where SOC is rising."""
    if "soc_pct" not in df.columns:
        return
    regen = df[df["soc_pct"].diff() > 0.05].copy()
    if regen.empty:
        return
    regen["gap"] = regen["timestamp"].diff().dt.total_seconds().gt(5).cumsum()
    first = True
    for _, span in regen.groupby("gap"):
        if len(span) < 3:
            continue
        fig.add_vrect(
            x0=span["timestamp"].iloc[0], x1=span["timestamp"].iloc[-1],
            fillcolor="rgba(167,139,250,0.07)", line_width=0,
            row=row, col=col,
        )
        if first:
            fig.add_trace(go.Scatter(
                x=[span["timestamp"].iloc[0]], y=[None],
                mode="markers",
                marker=dict(symbol="triangle-up", size=8, color=C_REGEN),
                name="⚡ Regen Zone",
                showlegend=True,
            ), row=row, col=col)
            first = False


def _add_worst_annotation(fig, df, row=1, col=1):
    """Mark the single worst prediction moment."""
    if df.empty or "abs_error" not in df.columns:
        return
    worst = df.loc[df["abs_error"].idxmax()]
    fig.add_annotation(
        x=worst["timestamp"],
        y=worst["pred"],
        text=f"  Worst: {worst['abs_error']:.2f}% err",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef4444",
        arrowsize=1,
        arrowwidth=1.5,
        font=dict(color="#ef4444", size=10, family="DM Mono"),
        bgcolor="rgba(15,20,36,0.85)",
        bordercolor="#ef4444",
        borderwidth=1,
        row=row, col=col,
    )


def _add_error_fill(fig, df, row=1, col=1):
    """Fill gap between actual and predicted — red=over, green=under."""
    df = df.sort_values("timestamp")
    ts = df["timestamp"].tolist()
    act = df["y_soc_t_plus_300s"].tolist()
    prd = df["pred"].tolist()

    # Over-prediction fill (pred > actual)
    y_upper = [max(p, a) for p, a in zip(prd, act)]
    y_lower = [min(p, a) for p, a in zip(prd, act)]

    fig.add_trace(go.Scatter(
        x=ts + ts[::-1],
        y=y_upper + y_lower[::-1],
        fill="toself",
        fillcolor="rgba(239,68,68,0.10)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        name="_err_fill",
    ), row=row, col=col)


# ══════════════════════════════════════════════════════════════
# VIEW 1 — Continuous day view (all trips joined, trip bands)
# ══════════════════════════════════════════════════════════════
def chart_continuous(df: pd.DataFrame) -> go.Figure:
    """All trips on one timeline. Trip boundaries shaded alternately."""
    df = df.sort_values("timestamp")
    sample = df if len(df) <= 5000 else df.sample(5000).sort_values("timestamp")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=["SOC — Actual vs Predicted", "Prediction Error (% SOC)"],
    )

    # Shade alternating trips
    trips = df["trip_id"].unique()
    for i, tid in enumerate(sorted(trips)):
        tdf = df[df["trip_id"] == tid]
        if tdf.empty:
            continue
        col = "rgba(96,165,250,0.04)" if i % 2 == 0 else "rgba(245,158,11,0.04)"
        fig.add_vrect(
            x0=tdf["timestamp"].min(), x1=tdf["timestamp"].max(),
            fillcolor=col, line_width=0, row="all", col=1,
        )
        # Trip label
        mid = tdf["timestamp"].iloc[len(tdf)//2]
        fig.add_annotation(
            x=mid, y=1.02, text=tid, showarrow=False,
            font=dict(size=9, color="#4a6fa5"),
            xref="x", yref="paper",
        )

    # Error fill
    _add_error_fill(fig, sample, row=1, col=1)

    # Actual
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["y_soc_t_plus_300s"],
        mode="lines", name="Actual SOC",
        line=dict(color=C_ACTUAL, width=2),
    ), row=1, col=1)

    # Predicted
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["pred"],
        mode="lines", name="Predicted SOC",
        line=dict(color=C_PRED, width=1.5, dash="dot"),
    ), row=1, col=1)

    # Fault markers
    _add_fault_markers(fig, sample, row=1, col=1)

    # Regen zones
    _add_regen_markers(fig, sample, row=1, col=1)

    # Worst annotation
    _add_worst_annotation(fig, sample, row=1, col=1)

    # Error panel
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["error"],
        mode="lines", name="Error",
        line=dict(color="#a78bfa", width=1),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.08)",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="#1e2640", line_width=1, row=2, col=1)

    layout = {**BASE}
    layout.update(dict(
        height=520,
        title=dict(text="Full Day — SOC Prediction Timeline", font=dict(size=15, color="#e8f0fe"), x=0.01),
        yaxis=dict(**BASE["yaxis"], title="SOC (%)"),
        yaxis2=dict(**BASE["yaxis"], title="Error (%)"),
        xaxis2=dict(**BASE["xaxis"], title="Time"),
    ))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════
# VIEW 2 — Per-trip subplots
# ══════════════════════════════════════════════════════════════
def chart_subplots(df: pd.DataFrame) -> go.Figure:
    """One subplot per trip, stacked vertically."""
    trips = sorted(df["trip_id"].unique())
    n = len(trips)
    if n == 0:
        return go.Figure()

    titles = [f"{tid} — MAE: {df[df['trip_id']==tid]['abs_error'].mean():.3f}%" for tid in trips]
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        vertical_spacing=max(0.04, 0.25/n),
        subplot_titles=titles,
    )

    for i, tid in enumerate(trips, start=1):
        tdf = df[df["trip_id"] == tid].sort_values("timestamp")
        if tdf.empty:
            continue

        _add_error_fill(fig, tdf, row=i, col=1)

        fig.add_trace(go.Scatter(
            x=tdf["timestamp"], y=tdf["y_soc_t_plus_300s"],
            mode="lines", name="Actual" if i == 1 else None,
            line=dict(color=C_ACTUAL, width=2),
            showlegend=(i == 1),
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=tdf["timestamp"], y=tdf["pred"],
            mode="lines", name="Predicted" if i == 1 else None,
            line=dict(color=C_PRED, width=1.5, dash="dot"),
            showlegend=(i == 1),
        ), row=i, col=1)

        _add_fault_markers(fig, tdf, row=i, col=1)
        _add_regen_markers(fig, tdf, row=i, col=1)
        _add_worst_annotation(fig, tdf, row=i, col=1)

        fig.update_yaxes(title_text="SOC %", row=i, col=1,
                         gridcolor="#1a2035", tickfont=dict(size=10))
        fig.update_xaxes(gridcolor="#1a2035", tickfont=dict(size=10), row=i, col=1)

    layout = {**BASE}
    layout.update(dict(
        height=max(280 * n, 400),
        title=dict(text="Per-Trip SOC Prediction", font=dict(size=15, color="#e8f0fe"), x=0.01),
        showlegend=True,
    ))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════
# VIEW 3 — Overlaid trips (normalised to trip start time)
# ══════════════════════════════════════════════════════════════
def chart_overlaid(df: pd.DataFrame) -> go.Figure:
    """All trips on same axes, x = seconds since trip start."""
    trips = sorted(df["trip_id"].unique())
    palette_actual = ["#60a5fa","#34d399","#f472b6","#a78bfa","#fb923c","#facc15"]
    palette_pred   = ["#1d4ed8","#065f46","#9d174d","#4c1d95","#7c2d12","#713f12"]

    fig = go.Figure()

    for i, tid in enumerate(trips):
        tdf = df[df["trip_id"] == tid].sort_values("timestamp").copy()
        t0 = tdf["timestamp"].min()
        tdf["elapsed_s"] = (tdf["timestamp"] - t0).dt.total_seconds()
        ca = palette_actual[i % len(palette_actual)]
        cp = palette_pred[i % len(palette_pred)]

        fig.add_trace(go.Scatter(
            x=tdf["elapsed_s"], y=tdf["y_soc_t_plus_300s"],
            mode="lines", name=f"{tid} — Actual",
            line=dict(color=ca, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=tdf["elapsed_s"], y=tdf["pred"],
            mode="lines", name=f"{tid} — Pred",
            line=dict(color=cp, width=1.5, dash="dot"),
        ))

    layout = {**BASE}
    layout.update(dict(
        height=400,
        title=dict(text="All Trips Overlaid — SOC from Trip Start", font=dict(size=15, color="#e8f0fe"), x=0.01),
        xaxis=dict(**BASE["xaxis"], title="Seconds since trip start"),
        yaxis=dict(**BASE["yaxis"], title="SOC (%)"),
    ))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════
# VIEW 4 — Rich single-trip (with rate-of-change + confidence)
# ══════════════════════════════════════════════════════════════
def chart_single_trip(df: pd.DataFrame, trip_id: str) -> go.Figure:
    """Three-panel deep dive for one trip."""
    tdf = df[df["trip_id"] == trip_id].sort_values("timestamp").copy()
    if tdf.empty:
        return go.Figure()

    # SOC rate of change (% per minute)
    tdf["soc_roc"] = tdf["y_soc_t_plus_300s"].diff() / \
                     tdf["timestamp"].diff().dt.total_seconds() * 60

    # Confidence band (±1 rolling std of error over 60s window)
    tdf["err_std"] = tdf["error"].rolling(60, min_periods=5).std().fillna(0.2)
    tdf["band_upper"] = tdf["pred"] + 1.5 * tdf["err_std"]
    tdf["band_lower"] = tdf["pred"] - 1.5 * tdf["err_std"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.05,
        subplot_titles=[
            f"SOC — Actual vs Predicted with Confidence Band ({trip_id})",
            "Prediction Error & SOC Rate of Change",
            "Active Fault Indicator",
        ],
    )

    # ── Panel 1: SOC + confidence band ─────────────────────────
    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([tdf["timestamp"], tdf["timestamp"].iloc[::-1]]),
        y=pd.concat([tdf["band_upper"], tdf["band_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(245,158,11,0.08)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=True,
        name="±1.5σ Confidence Band",
    ), row=1, col=1)

    # Error fill
    _add_error_fill(fig, tdf, row=1, col=1)

    # Actual SOC
    fig.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["y_soc_t_plus_300s"],
        mode="lines", name="Actual SOC",
        line=dict(color=C_ACTUAL, width=2.5),
    ), row=1, col=1)

    # Predicted SOC
    fig.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["pred"],
        mode="lines", name="Predicted SOC",
        line=dict(color=C_PRED, width=2, dash="dot"),
    ), row=1, col=1)

    # Regen zones
    _add_regen_markers(fig, tdf, row=1, col=1)

    # Fault zones on panel 1
    _add_fault_markers(fig, tdf, row=1, col=1)

    # Worst prediction annotation
    _add_worst_annotation(fig, tdf, row=1, col=1)

    # ── Panel 2: Error + SOC rate of change ────────────────────
    fig.add_trace(go.Bar(
        x=tdf["timestamp"], y=tdf["error"],
        name="Prediction Error",
        marker=dict(
            color=tdf["error"].apply(lambda e: C_ERROR_POS if e > 0 else C_ERROR_NEG),
            opacity=0.7,
        ),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["soc_roc"],
        mode="lines", name="SOC Rate (%/min)",
        line=dict(color=C_RATE, width=1.5),
        yaxis="y4",
    ), row=2, col=1)

    fig.add_hline(y=0, line_color="#1e2640", line_width=1, row=2, col=1)

    # ── Panel 3: Fault indicator ────────────────────────────────
    if "fault_any" in tdf.columns:
        fig.add_trace(go.Scatter(
            x=tdf["timestamp"], y=tdf["fault_any"],
            mode="lines", name="Fault Active",
            line=dict(color=C_FAULT, width=0),
            fill="tozeroy",
            fillcolor="rgba(251,146,60,0.35)",
        ), row=3, col=1)

    # ── Layout ──────────────────────────────────────────────────
    layout = {**BASE}
    layout.update(dict(
        height=640,
        title=dict(
            text=f"Deep Dive — {trip_id}  ·  MAE: {tdf['abs_error'].mean():.3f}%  ·  "
                 f"SOC: {tdf['y_soc_t_plus_300s'].iloc[0]:.1f}% → {tdf['y_soc_t_plus_300s'].iloc[-1]:.1f}%",
            font=dict(size=14, color="#e8f0fe"), x=0.01,
        ),
        yaxis =dict(**BASE["yaxis"], title="SOC (%)"),
        yaxis2=dict(**BASE["yaxis"], title="Error (%)"),
        yaxis3=dict(**BASE["yaxis"], title="Fault", tickvals=[0,1], ticktext=["Off","On"]),
        xaxis3=dict(**BASE["xaxis"], title="Time"),
    ))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════
# Streamlit renderer — call this from app.py
# ══════════════════════════════════════════════════════════════
def render_soc_section(df: pd.DataFrame):
    """
    Full SOC visualisation section.
    df must have: timestamp, trip_id, y_soc_t_plus_300s, pred, error,
                  abs_error, soc_pct, fault_any (optional)
    """
    if df.empty:
        st.warning("No predictions available for this date.")
        return

    st.markdown("""
    <div style='background:#111827;border:1px solid #1e3a6e;border-radius:12px;
                padding:16px 20px;margin-bottom:20px'>
        <div style='font-size:11px;font-weight:600;letter-spacing:1.5px;
                    text-transform:uppercase;color:#4a6fa5;margin-bottom:6px'>
            Chart Guide
        </div>
        <div style='font-size:12px;color:#64748b;line-height:1.8'>
            🔵 <strong style='color:#60a5fa'>Blue solid</strong> — Actual SOC measured by BMS &nbsp;|&nbsp;
            🟡 <strong style='color:#f59e0b'>Amber dashed</strong> — Model prediction 5 min ahead &nbsp;|&nbsp;
            🔴 <strong style='color:#ef4444'>Red fill</strong> — Over-prediction gap &nbsp;|&nbsp;
            🟢 <strong style='color:#22c55e'>Green fill</strong> — Under-prediction gap &nbsp;|&nbsp;
            🟣 <strong style='color:#a78bfa'>Purple band</strong> — Regen zone (SOC recovering) &nbsp;|&nbsp;
            🟠 <strong style='color:#fb923c'>Orange</strong> — Active hardware fault
        </div>
    </div>
    """, unsafe_allow_html=True)

    # View selector
    view = st.radio(
        "CHART VIEW",
        ["Continuous Day", "Per-Trip Panels", "Trips Overlaid", "Single Trip Deep Dive"],
        horizontal=True,
        label_visibility="visible",
    )

    if view == "Continuous Day":
        st.plotly_chart(chart_continuous(df), width="stretch")
        st.markdown("""<div style='font-size:12px;color:#4a6fa5;padding:8px 0'>
            Shaded bands separate individual trips. Hover any point to see exact values.
            The error panel below the main chart shows where predictions drifted.
        </div>""", unsafe_allow_html=True)

    elif view == "Per-Trip Panels":
        st.plotly_chart(chart_subplots(df), width="stretch")
        st.markdown("""<div style='font-size:12px;color:#4a6fa5;padding:8px 0'>
            Each trip shown independently. Subtitle shows MAE for that trip.
            Compare trip difficulty — longer trips with bigger SOC drops are harder to predict.
        </div>""", unsafe_allow_html=True)

    elif view == "Trips Overlaid":
        st.plotly_chart(chart_overlaid(df), width="stretch")
        st.markdown("""<div style='font-size:12px;color:#4a6fa5;padding:8px 0'>
            All trips normalised to start at t=0. Reveals whether prediction error grows
            with trip duration, or stays flat — and how different trips compare in SOC depletion rate.
        </div>""", unsafe_allow_html=True)

    elif view == "Single Trip Deep Dive":
        trips = sorted(df["trip_id"].unique())
        selected = st.selectbox("SELECT TRIP", trips, key="soc_trip_select")
        st.plotly_chart(chart_single_trip(df, selected), width="stretch")
        st.markdown("""<div style='font-size:12px;color:#4a6fa5;padding:8px 0'>
            <strong style='color:#34d399'>Panel 1:</strong> SOC with confidence band (±1.5σ rolling error) + fault zones + regen zones. &nbsp;
            <strong style='color:#34d399'>Panel 2:</strong> Bar = prediction error (🔴 over, 🟢 under), line = SOC rate of change (%/min). &nbsp;
            <strong style='color:#34d399'>Panel 3:</strong> Fault activity timeline.
        </div>""", unsafe_allow_html=True)