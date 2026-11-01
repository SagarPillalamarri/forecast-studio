"""src/plots.py
Plotly figure generators for all 10 dashboard analysis tabs.
Engineered with modern Big-Tech aesthetics (Google / Meta / Stripe minimal style):
- Clean white surfaces, subtle borders (#E2E8F0), refined typography
- Professional palette (Electric Azure #2563eb, Deep Indigo #4f46e5, Slate #0f172a)
- Shaded area gradients, rounded corners, unobtrusive grids (#f1f5f9)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.models import MODEL_COLORS

LAYOUT_THEME = dict(
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", size=12, color="#334155"),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=45, r=30, t=55, b=45),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0f172a",
        font_size=12,
        font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto",
        font_color="#ffffff",
        bordercolor="#334155"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=11, color="#475569")
    )
)


def plot_raw_moving_average(series: pd.Series, ma_series: pd.Series, window: int, title: str, unit: str):
    """Plot 1: Original series with Moving Average overlay + soft area gradient."""
    fig = go.Figure()
    
    # Soft filled area under the raw curve (Modern SaaS look)
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name="Observed Data",
        line=dict(color="#2563eb", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.06)",
        opacity=0.9
    ))
    
    # Crisp moving average line
    fig.add_trace(go.Scatter(
        x=ma_series.index,
        y=ma_series.values,
        mode="lines",
        name=f"Moving Average ({window}-period)",
        line=dict(color="#0d9488", width=2.8)
    ))
    
    fig.update_layout(
        **LAYOUT_THEME,
        title=dict(text=f"<b>{title}</b> <span style='font-size:12px;color:#64748b;font-weight:normal;'>| Trajectory & Smoothing</span>", x=0.01),
        xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(title=unit, showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    )
    return fig


def plot_decomposition(decomp_dict: dict, title: str, unit: str):
    """Plot 2: 4-panel decomposition chart (Observed, Trend, Seasonal, Residuals)."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        subplot_titles=[
            "Observed Data y(t)",
            "Secular Trend Component T(t)",
            "Seasonal Component S(t)",
            "Residual Irregular Component e(t)"
        ]
    )
    
    obs = decomp_dict["observed"]
    trend = decomp_dict["trend"]
    seas = decomp_dict["seasonal"]
    resid = decomp_dict["resid"]
    
    # Observed
    fig.add_trace(go.Scatter(x=obs.index, y=obs.values, mode="lines", name="Observed", line=dict(color="#2563eb", width=1.6), fill="tozeroy", fillcolor="rgba(2, 132, 199, 0.04)"), row=1, col=1)
    
    # Trend
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values, mode="lines", name="Trend", line=dict(color="#6366f1", width=2.2)), row=2, col=1)
    
    # Seasonal
    fig.add_trace(go.Scatter(x=seas.index, y=seas.values, mode="lines", name="Seasonal", line=dict(color="#0d9488", width=1.5)), row=3, col=1)
    
    # Residuals
    fig.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines", name="Residuals", line=dict(color="#64748b", width=1.2)), row=4, col=1)
    
    # Add horizontal zero reference line on residuals
    ref_y = 1.0 if "Multiplicative" in decomp_dict.get("type", "") else 0.0
    fig.add_hline(y=ref_y, line=dict(color="#94a3b8", dash="dash", width=1), row=4, col=1)
    
    fig.update_layout(
        **LAYOUT_THEME,
        height=660,
        title=dict(text=f"<b>{title}</b> <span style='font-size:12px;color:#64748b;font-weight:normal;'>| {decomp_dict.get('type', 'Decomposition')}</span>", x=0.01),
        showlegend=False
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="#f8fafc", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", row=i, col=1)
        
    return fig


def plot_seasonal_index(df_season: pd.DataFrame, title: str, period_label: str):
    """Plot 3: Seasonal index bar chart with modern rounded look."""
    fig = go.Figure()
    
    # Distinct positive and negative palette
    colors = ["#2563eb" if p >= 0 else "#0d9488" for p in df_season["percentage"]]
    
    fig.add_trace(go.Bar(
        x=df_season["period_label"],
        y=df_season["percentage"],
        text=[f"{p:+.1f}%" for p in df_season["percentage"]],
        textposition="outside",
        marker=dict(color=colors, line=dict(width=0)),
        name="Deviation (%)"
    ))
    
    fig.add_hline(y=0, line=dict(color="#cbd5e1", width=1.2))
    
    fig.update_layout(
        **LAYOUT_THEME,
        title=dict(text=f"<b>{title}</b> <span style='font-size:12px;color:#64748b;font-weight:normal;'>| Seasonal Index (% Deviation)</span>", x=0.01),
        xaxis=dict(title=period_label, showgrid=False),
        yaxis=dict(title="% Deviation from Cycle Average", showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    )
    return fig


def plot_correlation_function(
    corr_data_orig: dict,
    corr_data_diff: dict,
    corr_data_seas: dict,
    kind: str,
    title: str
):
    """Plot 4 & 5: ACF or PACF with confidence bands across original, differenced, and seasonally differenced series."""
    key = "acf" if kind.upper() == "ACF" else "pacf"
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"{kind.upper()} — Original Series",
            f"{kind.upper()} — First Differenced (d=1)",
            f"{kind.upper()} — First + Seasonal Differenced (d=1, D=1)"
        ]
    )
    
    datasets = [
        (corr_data_orig, 1, "#2563eb"),
        (corr_data_diff, 2, "#6366f1"),
        (corr_data_seas, 3, "#0d9488")
    ]
    
    for cdata, row_idx, color in datasets:
        lags = cdata["lags"]
        vals = cdata[key]
        bound = cdata["conf_bound"]
        
        # 95% confidence bands
        fig.add_hline(y=bound, line=dict(color="#94a3b8", dash="dash", width=1), row=row_idx, col=1)
        fig.add_hline(y=-bound, line=dict(color="#94a3b8", dash="dash", width=1), row=row_idx, col=1)
        fig.add_hline(y=0, line=dict(color="#cbd5e1", width=0.8), row=row_idx, col=1)
        
        # Stems
        for lag, val in zip(lags, vals):
            fig.add_trace(go.Scatter(
                x=[lag, lag],
                y=[0, val],
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
                hoverinfo="skip"
            ), row=row_idx, col=1)
            
        # Markers
        fig.add_trace(go.Scatter(
            x=lags,
            y=vals,
            mode="markers",
            marker=dict(color=color, size=6),
            name=f"Lag correlation (row {row_idx})",
            showlegend=False
        ), row=row_idx, col=1)
        
    fig.update_layout(
        **LAYOUT_THEME,
        height=640,
        title=dict(text=f"<b>{title}</b> <span style='font-size:12px;color:#64748b;font-weight:normal;'>| {kind.upper()} Correlogram</span>", x=0.01)
    )
    for i in range(1, 4):
        fig.update_xaxes(title="Lag" if i == 3 else "", showgrid=True, gridcolor="#f8fafc", row=i, col=1)
        fig.update_yaxes(title="Correlation", range=[-1.05, 1.05], showgrid=True, gridcolor="#f1f5f9", row=i, col=1)
        
    return fig


def plot_forecast_comparison(
    train: pd.Series,
    test: pd.Series,
    model_results: dict,
    title: str,
    unit: str
):
    """Plot 7: Training data, test actuals, and forecasts with 95% prediction intervals."""
    fig = go.Figure()
    
    # Show last portion of train so plot stays focused and readable
    show_train_points = min(len(train), len(test) * 3)
    train_zoom = train.iloc[-show_train_points:]
    
    # Train actuals
    fig.add_trace(go.Scatter(
        x=train_zoom.index,
        y=train_zoom.values,
        mode="lines",
        name="Historical Train",
        line=dict(color="#94a3b8", width=1.6),
        opacity=0.8
    ))
    
    # Test actuals (Bold Slate)
    fig.add_trace(go.Scatter(
        x=test.index,
        y=test.values,
        mode="lines+markers",
        name="Test Actuals",
        line=dict(color="#0f172a", width=3.0),
        marker=dict(size=4.5)
    ))
    
    # Selected forecasts
    for m_name, res in model_results.items():
        color = MODEL_COLORS.get(m_name, "#475569")
        fcst = res["forecast"]
        lower = res["lower_95"]
        upper = res["upper_95"]
        
        # 95% Confidence interval band
        fig.add_trace(go.Scatter(
            x=list(fcst.index) + list(fcst.index)[::-1],
            y=list(upper.values) + list(lower.values)[::-1],
            fill="toself",
            fillcolor=color,
            opacity=0.10,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=f"{m_name} (95% CI)",
            showlegend=False
        ))
        
        # Forecast Mean Line
        fig.add_trace(go.Scatter(
            x=fcst.index,
            y=fcst.values,
            mode="lines+markers",
            name=m_name,
            line=dict(color=color, width=2.4),
            marker=dict(size=3.5)
        ))
        
    fig.update_layout(
        **LAYOUT_THEME,
        height=540,
        title=dict(text=f"<b>{title}</b> <span style='font-size:12px;color:#64748b;font-weight:normal;'>| Forecast Comparison (Holdout Horizon = {len(test)})</span>", x=0.01),
        xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title=unit, showgrid=True, gridcolor="#f1f5f9")
    )
    return fig


def plot_residual_diagnostics(
    residuals: pd.Series,
    model_name: str,
    dataset_title: str
):
    """
    Plot 9: 4-panel residual diagnostics:
    1. Residuals over time
    2. Histogram with overlaid normal KDE curve
    3. Residual ACF with 95% bounds
    4. Residual QQ plot or Normal probability reference
    """
    clean_res = residuals.dropna()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Standardized Residuals Over Time",
            "Residual Distribution vs Standard Normal",
            "Residual Autocorrelation (ACF)",
            "Normal Q-Q Plot"
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # 1. Residuals over time
    mean_r = clean_res.mean()
    std_r = clean_res.std() if clean_res.std() > 0 else 1.0
    std_res = (clean_res - mean_r) / std_r
    
    fig.add_trace(go.Scatter(
        x=clean_res.index,
        y=std_res.values,
        mode="lines",
        name="Residuals",
        line=dict(color="#2563eb", width=1.3)
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#cbd5e1", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=2, line=dict(color="#0d9488", dash="dot", width=0.9), row=1, col=1)
    fig.add_hline(y=-2, line=dict(color="#0d9488", dash="dot", width=0.9), row=1, col=1)
    
    # 2. Histogram + Normal curve
    fig.add_trace(go.Histogram(
        x=std_res.values,
        histnorm="probability density",
        name="Residual Density",
        marker=dict(color="rgba(2, 132, 199, 0.45)"),
        nbinsx=24,
        showlegend=False
    ), row=1, col=2)
    
    x_range = np.linspace(-3.5, 3.5, 100)
    normal_curve = norm.pdf(x_range, 0, 1)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_curve,
        mode="lines",
        name="Standard Normal",
        line=dict(color="#0d9488", width=2),
        showlegend=False
    ), row=1, col=2)
    
    # 3. Residual ACF
    nlags = min(24, len(clean_res) // 3)
    from statsmodels.tsa.stattools import acf
    res_acf = acf(clean_res, nlags=nlags)
    conf_bound = 1.96 / np.sqrt(len(clean_res))
    lags = np.arange(len(res_acf))
    
    fig.add_hline(y=conf_bound, line=dict(color="#94a3b8", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=-conf_bound, line=dict(color="#94a3b8", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#cbd5e1", width=0.8), row=2, col=1)
    
    for lag, val in zip(lags, res_acf):
        fig.add_trace(go.Scatter(
            x=[lag, lag],
            y=[0, val],
            mode="lines",
            line=dict(color="#2563eb", width=1.5),
            showlegend=False,
            hoverinfo="skip"
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=lags,
        y=res_acf,
        mode="markers",
        marker=dict(color="#2563eb", size=6),
        showlegend=False
    ), row=2, col=1)
    
    # 4. QQ plot
    sorted_res = np.sort(std_res.values)
    n = len(sorted_res)
    theoretical_quantiles = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sorted_res,
        mode="markers",
        name="Quantiles",
        marker=dict(color="#2563eb", size=5),
        showlegend=False
    ), row=2, col=2)
    
    # Reference line
    min_q, max_q = min(theoretical_quantiles.min(), sorted_res.min()), max(theoretical_quantiles.max(), sorted_res.max())
    fig.add_trace(go.Scatter(
        x=[min_q, max_q],
        y=[min_q, max_q],
        mode="lines",
        line=dict(color="#0d9488", dash="dash", width=1.5),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        **LAYOUT_THEME,
        height=620,
        title=dict(text=f"<b>Residual Diagnostics</b> <span style='font-size:12px;color:#64748b;font-weight:normal;'>| Model: {model_name} ({dataset_title})</span>", x=0.01)
    )
    fig.update_xaxes(title="Time", showgrid=True, gridcolor="#f8fafc", row=1, col=1)
    fig.update_yaxes(title="Std Residuals", showgrid=True, gridcolor="#f1f5f9", row=1, col=1)
    fig.update_xaxes(title="Std Deviations", showgrid=True, gridcolor="#f8fafc", row=1, col=2)
    fig.update_yaxes(title="Density", showgrid=True, gridcolor="#f1f5f9", row=1, col=2)
    fig.update_xaxes(title="Lag", showgrid=True, gridcolor="#f8fafc", row=2, col=1)
    fig.update_yaxes(title="Correlation", range=[-1.05, 1.05], showgrid=True, gridcolor="#f1f5f9", row=2, col=1)
    fig.update_xaxes(title="Theoretical Quantiles", showgrid=True, gridcolor="#f8fafc", row=2, col=2)
    fig.update_yaxes(title="Sample Quantiles", showgrid=True, gridcolor="#f1f5f9", row=2, col=2)
    
    return fig
