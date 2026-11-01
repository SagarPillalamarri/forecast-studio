"""Streamlit workspace for applied time series exploration and forecasting."""
import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import streamlit as st
import pandas as pd
import numpy as np

from src.data import DATASET_CONFIGS, load_dataset, get_train_test_split
from src.analysis import (
    compute_moving_average,
    compute_decomposition,
    compute_seasonal_indices,
    compute_acf_pacf,
    get_stationarity_summary
)
from src.models import (
    AVAILABLE_MODELS,
    fit_and_forecast_model
)
from src.plots import (
    plot_raw_moving_average,
    plot_decomposition,
    plot_seasonal_index,
    plot_correlation_function,
    plot_forecast_comparison,
    plot_residual_diagnostics
)

# Page Configuration
st.set_page_config(
    page_title="Forecast Studio | ATSA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Executive analytics design system.
st.markdown("""
<style>
.stApp {background: #f5f7fb; color: #15243b;}
.stApp h1, .stApp h2, .stApp h3 {color: #14243d; letter-spacing: -.025em;}
.block-container {max-width: 1560px; padding: 2rem 2.5rem 3rem;}
header[data-testid="stHeader"] {background: transparent;}
section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e3e9f2;}
section[data-testid="stSidebar"] > div {padding-top: 1.5rem;}
#MainMenu, footer {visibility: hidden;}
.sidebar-brand {display:flex; align-items:center; gap:12px; padding:0 0 24px; margin-bottom:20px; border-bottom:1px solid #e9edf4;}
.brand-mark {background:#e8f1ff; color:#2563eb; border-radius:10px; padding:8px 12px; font-size:22px; font-weight:700;}
.brand-title {font-size:18px; font-weight:750; color:#15243b;}
.brand-subtitle {font-size:11px; color:#748298; margin-top:3px;}
.eyebrow {font-size:11px; font-weight:700; letter-spacing:.12em; color:#718096; text-transform:uppercase; margin:20px 0 8px;}
.page-heading {display:flex; justify-content:space-between; align-items:center; gap:20px; margin-bottom:24px;}
.page-heading h1 {font-size:30px; margin:4px 0 5px; padding:0; line-height:1.25;}
.page-heading p {margin:0; color:#718096; font-size:14px;}
.dataset-badge {background:#eaf2ff; color:#2458a8; border:1px solid #d9e7fc; border-radius:8px; padding:8px 12px; font-size:12px; white-space:nowrap;}
.metric-card {background:#fff; border:1px solid #e3e9f2; border-radius:10px; padding:19px 20px; min-height:125px; margin-bottom:20px;}
.metric-label {font-size:12px; color:#66758b; margin-bottom:9px;}
.metric-value {font-size:27px; font-weight:750; letter-spacing:-.04em; color:#14243d; line-height:1.2;}
.metric-note {font-size:11px; color:#718096; margin-top:9px;}
.caption-box {background:#eef4fc; border-left:3px solid #3b82f6; padding:14px 17px; border-radius:0 7px 7px 0; margin:12px 0 24px; font-size:13px; color:#46566c; line-height:1.65;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {background:#fff; border:1px solid #e3e9f2; border-radius:10px; overflow:hidden;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e3e9f2; border-radius:10px; padding:16px;}
div[data-testid="stSegmentedControl"] {margin-bottom:20px;}
.stButton button {border-radius:8px;}
@media(max-width:900px) {.block-container {padding:1.5rem 1rem;} .page-heading {align-items:flex-start; flex-direction:column;} .page-heading h1 {font-size:25px;} .metric-value {font-size:23px;}}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED MODEL FITTING WRAPPER
# ==============================================================================
@st.cache_data(show_spinner=True)
def get_fitted_model_results(dataset_id: str, test_size: int, selected_models_tuple: tuple):
    """
    Fits and caches all selected forecasting models for a dataset.
    Returns dictionary mapping model_name -> result_dict.
    """
    df = load_dataset(dataset_id)
    cfg = DATASET_CONFIGS[dataset_id]
    train_df, test_df = get_train_test_split(df, test_size)
    
    train_s = train_df["value"]
    test_s = test_df["value"]
    freq = cfg["frequency"]
    m = cfg["seasonal_period"]
    
    results = {}
    for model_name in selected_models_tuple:
        try:
            res = fit_and_forecast_model(
                model_name=model_name,
                train=train_s,
                test=test_s,
                frequency=freq,
                seasonal_period=m,
                dataset_id=dataset_id
            )
            results[model_name] = res
        except Exception as e:
            st.warning(f"⚠️ Model **{model_name}** failed to fit on {cfg['short_name']}: {e}")
            
    return results


# Workspace controls and grouped navigation.
from html import escape

st.sidebar.markdown("""
<div class="sidebar-brand"><div class="brand-mark">↗</div><div>
<div class="brand-title">Forecast Studio</div>
<div class="brand-subtitle">Applied Time Series Analysis</div></div></div>
""", unsafe_allow_html=True)
view_mode = st.sidebar.selectbox(
    "Dataset", ["Dataset 1 (NHS A&E)", "Dataset 2 (UK Electricity)", "Both side by side"], index=0
)
if view_mode == "Dataset 1 (NHS A&E)":
    active_dataset_keys = ["ae_monthly"]
elif view_mode == "Dataset 2 (UK Electricity)":
    active_dataset_keys = ["electricity_daily"]
else:
    active_dataset_keys = ["ae_monthly", "electricity_daily"]

st.sidebar.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
workspace = st.sidebar.radio("Workspace", ["Explore", "Forecast", "Evaluate"], label_visibility="collapsed")
view_groups = {
    "Explore": {"Historical series": "1. Raw + MA", "Decomposition": "2. Decomposition", "Seasonal patterns": "3. Seasonal Index", "Autocorrelation": "4. ACF", "Partial autocorrelation": "5. PACF", "Stationarity": "6. Stationarity (ADF)"},
    "Forecast": {"Forecast comparison": "7. Forecast Comparison"},
    "Evaluate": {"Model results": "8. Model Metrics", "Residual diagnostics": "9. Residual Diagnostics", "Final verdict": "10. Verdict"}
}
st.sidebar.markdown('<div class="eyebrow">Analysis settings</div>', unsafe_allow_html=True)
with st.sidebar.expander("Series & holdout", expanded=True):
    ma_window = st.slider("Moving average window", min_value=3, max_value=30, value=7, help="Rolling window size for trend extraction.")
    test_size_ae = st.number_input("NHS holdout · months", min_value=3, max_value=36, value=DATASET_CONFIGS["ae_monthly"]["default_test_size"], step=1)
    test_size_elec = st.number_input("Electricity holdout · days", min_value=7, max_value=180, value=DATASET_CONFIGS["electricity_daily"]["default_test_size"], step=1)
default_selected = ["Naive", "Seasonal Naive", "Holt-Winters (Additive)", "ARIMA(p,d,q)", "SARIMA", "Auto ARIMA", "Prophet", "SARIMAX (Exogenous)"]
with st.sidebar.expander("Benchmark models", expanded=False):
    selected_models = [name for name in AVAILABLE_MODELS if st.checkbox(name, value=name in default_selected, key=f"chk_{name}")]
    if not selected_models:
        st.warning("Select at least one model. Seasonal Naive is used as a fallback.")
        selected_models = ["Seasonal Naive"]
st.sidebar.caption(f"{len(selected_models)} of {len(AVAILABLE_MODELS)} models selected")
st.sidebar.divider()
st.sidebar.caption("ATSA · Comparative forecasting workspace")

views = view_groups[workspace]
if len(views) > 1:
    view_label = st.segmented_control("Analysis view", list(views), default=next(iter(views)), key=f"view_{workspace}", selection_mode="single", label_visibility="collapsed")
    view_label = view_label or next(iter(views))
else:
    view_label = next(iter(views))
active_tab = views[view_label]

def get_dataset_test_size(ds_id: str) -> int:
    return test_size_ae if ds_id == "ae_monthly" else test_size_elec

active_frames = {ds_id: load_dataset(ds_id) for ds_id in active_dataset_keys}
dataset_title = DATASET_CONFIGS[active_dataset_keys[0]]["short_name"] if len(active_dataset_keys) == 1 else "Compare datasets"
subtitles = {"Explore": "Explore historical patterns, seasonality, and time series structure.", "Forecast": "Compare model forecasts against the held-out observations.", "Evaluate": "Review accuracy, residual behavior, and model selection."}
st.markdown(f"""
<div class="page-heading"><div><div class="eyebrow" style="margin-top:0">{workspace} / {escape(view_label)}</div>
<h1>{escape(dataset_title)} — {escape(view_label)}</h1><p>{subtitles[workspace]}</p></div>
<div class="dataset-badge">{'Monthly series' if active_dataset_keys == ['ae_monthly'] else 'Daily series' if active_dataset_keys == ['electricity_daily'] else 'Monthly + daily series'}</div></div>
""", unsafe_allow_html=True)

def metric_card(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value">{escape(str(value))}</div><div class="metric-note">{escape(note)}</div></div>', unsafe_allow_html=True)

kpis = st.columns(4)
if len(active_dataset_keys) == 1:
    ds_id = active_dataset_keys[0]
    frame = active_frames[ds_id]
    cfg = DATASET_CONFIGS[ds_id]
    with kpis[0]:
        metric_card("Observations", f"{len(frame):,}", f"{frame.index.min():%b %Y} – {frame.index.max():%b %Y}")
    with kpis[1]:
        metric_card("Latest observation", f"{frame['value'].iloc[-1]:,.0f}", f"{frame.index[-1]:%d %b %Y} · {cfg['unit']}")
    with kpis[2]:
        metric_card("Average per period", f"{frame['value'].mean():,.0f}", cfg['unit'])
    with kpis[3]:
        metric_card("Holdout periods", get_dataset_test_size(ds_id), f"{len(selected_models)} benchmark models selected")
else:
    for idx, ds_id in enumerate(active_dataset_keys):
        frame = active_frames[ds_id]
        with kpis[idx]:
            metric_card(DATASET_CONFIGS[ds_id]['short_name'], f"{len(frame):,}", f"{frame.index.min():%b %Y} – {frame.index.max():%b %Y}")
    with kpis[2]:
        metric_card("Holdout periods", f"{test_size_ae}m / {test_size_elec}d", "Monthly NHS · daily electricity")
    with kpis[3]:
        metric_card("Selected models", len(selected_models), f"Of {len(AVAILABLE_MODELS)} available benchmarks")

# ==============================================================================
# TAB 1: RAW + MOVING AVERAGE
# ==============================================================================
if active_tab == "1. Raw + MA":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            df = load_dataset(ds_id)
            ma_s = compute_moving_average(df["value"], ma_window)
            fig = plot_raw_moving_average(df["value"], ma_s, ma_window, cfg["name"], cfg["unit"])
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                f'<div class="caption-box">'
                f'<b>Interpretation:</b> Displays the historical raw series along with a centered {ma_window}-period moving average.<br>'
                f'Look for underlying secular trend directions, smoothing of high-frequency noise, and distinct cyclical swings or shocks '
                f'(e.g., the sharp COVID pandemic dip in 2020 for A&E, or seasonal winter heating peaks in electricity demand).'
                f'</div>',
                unsafe_allow_html=True
            )

# ==============================================================================
# TAB 2: DECOMPOSITION
# ==============================================================================
elif active_tab == "2. Decomposition":
    decomp_mode = st.radio(
        "Decomposition Method:",
        ["STL (LOESS)", "Classical Additive", "Classical Multiplicative"],
        horizontal=True,
        key="decomp_radio"
    )
    mode_map = {
        "STL (LOESS)": "stl",
        "Classical Additive": "additive",
        "Classical Multiplicative": "multiplicative"
    }
    
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            df = load_dataset(ds_id)
            decomp = compute_decomposition(
                df["value"],
                period=cfg["seasonal_period"],
                model_type=mode_map[decomp_mode]
            )
            fig = plot_decomposition(decomp, cfg["name"], cfg["unit"])
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                f'<div class="caption-box">'
                f'<b>Interpretation:</b> Deconstructs the time series into independent observed, trend, seasonal, and residual components.<br>'
                f'Check whether the seasonal fluctuations stay constant across time (additive behavior) or scale proportionally with the trend '
                f'(multiplicative behavior). Residuals should resemble stationary white noise without leftover patterns.'
                f'</div>',
                unsafe_allow_html=True
            )

# ==============================================================================
# TAB 3: SEASONAL INDEX
# ==============================================================================
elif active_tab == "3. Seasonal Index":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            df = load_dataset(ds_id)
            s_idx_df = compute_seasonal_indices(df["value"], cfg["frequency"], cfg["seasonal_period"])
            fig = plot_seasonal_index(s_idx_df, cfg["name"], cfg["period_label"])
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                f'<div class="caption-box">'
                f'<b>Interpretation:</b> Normalized seasonal indices quantifying percentage deviation from the annual or weekly baseline.<br>'
                f'For monthly A&E attendances, look for summer peaks and holiday variation; for daily electricity demand, observe the significant '
                f'drop on Saturdays and Sundays relative to peak weekday industrial consumption.'
                f'</div>',
                unsafe_allow_html=True
            )

# ==============================================================================
# TAB 4: ACF
# ==============================================================================
elif active_tab == "4. ACF":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            df = load_dataset(ds_id)
            s = df["value"]
            m = cfg["seasonal_period"]
            
            c_orig = compute_acf_pacf(s, nlags=36)
            c_diff = compute_acf_pacf(s.diff().dropna(), nlags=36)
            c_seas = compute_acf_pacf(s.diff().diff(m).dropna(), nlags=36)
            
            fig = plot_correlation_function(c_orig, c_diff, c_seas, "ACF", cfg["name"])
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                f'<div class="caption-box">'
                f'<b>Interpretation:</b> Autocorrelation Function (ACF) showing serial correlation across increasing lag intervals.<br>'
                f'A very slow linear decay in the original series indicates non-stationarity and trend. Notice how first-differencing '
                f'rapidly cuts off lag correlations, while seasonal differencing removes seasonal echo at multiples of lag {m}.'
                f'</div>',
                unsafe_allow_html=True
            )

# ==============================================================================
# TAB 5: PACF
# ==============================================================================
elif active_tab == "5. PACF":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            df = load_dataset(ds_id)
            s = df["value"]
            m = cfg["seasonal_period"]
            
            c_orig = compute_acf_pacf(s, nlags=36)
            c_diff = compute_acf_pacf(s.diff().dropna(), nlags=36)
            c_seas = compute_acf_pacf(s.diff().diff(m).dropna(), nlags=36)
            
            fig = plot_correlation_function(c_orig, c_diff, c_seas, "PACF", cfg["name"])
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                f'<div class="caption-box">'
                f'<b>Interpretation:</b> Partial Autocorrelation Function (PACF) measuring direct correlation after controlling for intermediate lags.<br>'
                f'A sharp cutoff after lag p in the differenced series suggests an AR(p) process, whereas a damped exponential wave suggests '
                f'a Moving Average component. Use significant spikes to select candidate ARIMA orders.'
                f'</div>',
                unsafe_allow_html=True
            )

# ==============================================================================
# TAB 6: STATIONARITY (ADF)
# ==============================================================================
elif active_tab == "6. Stationarity (ADF)":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            df = load_dataset(ds_id)
            adf_df = get_stationarity_summary(df["value"], cfg["seasonal_period"])
            
            st.subheader(f"ADF unit-root test: {cfg['short_name']}")
            
            display_df = adf_df.copy()
            st.dataframe(display_df, width="stretch", hide_index=True)
            
            st.markdown(
                f'<div class="caption-box">'
                f'<b>Interpretation:</b> Augmented Dickey-Fuller (ADF) test evaluates the null hypothesis of a unit root (non-stationarity).<br>'
                f'A p-value &lt; 0.05 rejects the null hypothesis and confirms stationarity. Original series is typically non-stationary '
                f'(ADF p > 0.05), but first differencing achieves mean-stationarity (d=1, p &lt; 0.01), satisfying prerequisites for ARMA modeling.'
                f'</div>',
                unsafe_allow_html=True
            )

# ==============================================================================
# TAB 7: FORECAST COMPARISON
# ==============================================================================
elif active_tab == "7. Forecast Comparison":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            df = load_dataset(ds_id)
            t_size = get_dataset_test_size(ds_id)
            train_df, test_df = get_train_test_split(df, t_size)
            
            results = get_fitted_model_results(ds_id, t_size, tuple(selected_models))
            
            if results:
                fig = plot_forecast_comparison(
                    train_df["value"],
                    test_df["value"],
                    results,
                    cfg["name"],
                    cfg["unit"]
                )
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    f'<div class="caption-box">'
                    f'<b>Interpretation:</b> Out-of-sample forecast trajectories and shaded 95% prediction intervals evaluated over {t_size} hold-out points.<br>'
                    f'Click any model in the top legend to toggle its visibility. Observe how well each model tracks turning points, amplitude, '
                    f'and recurring seasonal peaks in comparison to actual test realizations.'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.error("No models fitted successfully. Please check model selections in sidebar.")

# ==============================================================================
# TAB 8: MODEL METRICS
# ==============================================================================
elif active_tab == "8. Model Metrics":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            t_size = get_dataset_test_size(ds_id)
            results = get_fitted_model_results(ds_id, t_size, tuple(selected_models))
            
            if results:
                st.subheader(f"Model evaluation: {cfg['short_name']}")
                
                rows = []
                for m_name, res in results.items():
                    lb_p = res["ljung_box_pvalue"]
                    lb_pass = (lb_p > 0.05) if not np.isnan(lb_p) else False
                    rows.append({
                        "Model": m_name,
                        "Specification": res["order_str"],
                        "Test RMSE": res["metrics"]["RMSE"],
                        "Test MAE": res["metrics"]["MAE"],
                        "Test MAPE (%)": res["metrics"]["MAPE"],
                        "AIC": res["aic"] if not np.isnan(res["aic"]) else "N/A",
                        "Ljung-Box p-val": f"{lb_p:.4f}" if not np.isnan(lb_p) else "N/A",
                        "Pass White Noise": "Yes" if lb_pass else "No",
                        "_lb_pass": lb_pass,
                        "_rmse": res["metrics"]["RMSE"]
                    })
                
                metrics_df = pd.DataFrame(rows)
                
                passing_models = metrics_df[metrics_df["_lb_pass"] == True]
                if not passing_models.empty:
                    best_model_name = passing_models.sort_values("_rmse").iloc[0]["Model"]
                else:
                    best_model_name = metrics_df.sort_values("_rmse").iloc[0]["Model"]
                    
                def highlight_best(row):
                    if row["Model"] == best_model_name:
                        return ["background-color: #e0f2fe; font-weight: 600; color: #0369a1"] * len(row)
                    return [""] * len(row)
                
                clean_disp = metrics_df.drop(columns=["_lb_pass", "_rmse"])
                styled_df = clean_disp.style.apply(highlight_best, axis=1)
                st.dataframe(styled_df, width="stretch", hide_index=True)
                
                s_naive_rmse = metrics_df.loc[metrics_df["Model"] == "Seasonal Naive", "_rmse"]
                if not s_naive_rmse.empty:
                    sn_val = s_naive_rmse.iloc[0]
                    best_rmse = metrics_df.loc[metrics_df["Model"] == best_model_name, "_rmse"].iloc[0]
                    if best_rmse < sn_val:
                        imp = ((sn_val - best_rmse) / sn_val) * 100
                        st.success(f"Best model **{best_model_name}** beat Seasonal Naive baseline by **{imp:.1f}%** lower RMSE.")
                    else:
                        st.warning(f"Best model did not beat Seasonal Naive baseline (Seasonal Naive RMSE: {sn_val:,.1f}).")
                
                st.markdown(
                    f'<div class="caption-box">'
                    f'<b>Interpretation:</b> Summary table comparing out-of-sample accuracy (RMSE, MAE, MAPE), complexity (AIC), and residual white noise.<br>'
                    f'The top-performing model is highlighted in azure according to the project decision rule: lowest test RMSE among models passing '
                    f'the Ljung-Box test for uncorrelated residuals (p > 0.05).'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ==============================================================================
# TAB 9: RESIDUAL DIAGNOSTICS
# ==============================================================================
elif active_tab == "9. Residual Diagnostics":
    cols = st.columns(len(active_dataset_keys))
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            t_size = get_dataset_test_size(ds_id)
            results = get_fitted_model_results(ds_id, t_size, tuple(selected_models))
            
            if results:
                valid_candidates = []
                for m_name, res in results.items():
                    lb_p = res["ljung_box_pvalue"]
                    valid_candidates.append({
                        "name": m_name,
                        "rmse": res["metrics"]["RMSE"],
                        "pass": (lb_p > 0.05) if not np.isnan(lb_p) else False
                    })
                valid_df = pd.DataFrame(valid_candidates)
                passers = valid_df[valid_df["pass"] == True]
                if not passers.empty:
                    default_m = passers.sort_values("rmse").iloc[0]["name"]
                else:
                    default_m = valid_df.sort_values("rmse").iloc[0]["name"]
                    
                model_choices = list(results.keys())
                def_idx = model_choices.index(default_m) if default_m in model_choices else 0
                chosen_m = st.selectbox(
                    f"Select Model for Diagnostics ({cfg['short_name']}):",
                    options=model_choices,
                    index=def_idx,
                    key=f"diag_sel_{ds_id}"
                )
                
                res_dict = results[chosen_m]
                residuals = res_dict["residuals"]
                lb_p = res_dict["ljung_box_pvalue"]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Residual Mean", f"{residuals.mean():.2f}")
                c2.metric("Residual Std Dev", f"{residuals.std():.2f}")
                c3.metric(
                    "Ljung-Box p-value",
                    f"{lb_p:.4f}" if not np.isnan(lb_p) else "N/A",
                    delta="White Noise" if (lb_p > 0.05) else "Autocorrelated",
                    delta_color="normal" if (lb_p > 0.05) else "inverse"
                )
                
                fig = plot_residual_diagnostics(residuals, chosen_m, cfg["short_name"])
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    f'<div class="caption-box">'
                    f'<b>Interpretation:</b> 4-quadrant diagnostic panel checking standard regression and time-series assumptions.<br>'
                    f'Adequate models should have zero-mean residuals with no volatility clustering, bell-shaped distribution aligned with '
                    f'the standard normal curve, ACF spikes falling within dotted 95% bands, and points tracking the 45-degree QQ reference line.'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ==============================================================================
# TAB 10: VERDICT
# ==============================================================================
elif active_tab == "10. Verdict":
    st.subheader("Final forecasting verdict")
    
    verdict_info = {}
    cols = st.columns(len(active_dataset_keys))
    
    for idx, ds_id in enumerate(active_dataset_keys):
        with cols[idx]:
            cfg = DATASET_CONFIGS[ds_id]
            t_size = get_dataset_test_size(ds_id)
            results = get_fitted_model_results(ds_id, t_size, tuple(selected_models))
            
            if results:
                candidate_list = []
                for m_name, res in results.items():
                    lb_p = res["ljung_box_pvalue"]
                    candidate_list.append({
                        "name": m_name,
                        "rmse": res["metrics"]["RMSE"],
                        "mae": res["metrics"]["MAE"],
                        "mape": res["metrics"]["MAPE"],
                        "spec": res["order_str"],
                        "lb_p": lb_p,
                        "lb_pass": (lb_p > 0.05) if not np.isnan(lb_p) else False
                    })
                cand_df = pd.DataFrame(candidate_list)
                passers = cand_df[cand_df["lb_pass"] == True]
                
                flag_no_pass = False
                if not passers.empty:
                    winner = passers.sort_values("rmse").iloc[0]
                else:
                    winner = cand_df.sort_values("rmse").iloc[0]
                    flag_no_pass = True
                    
                sn_row = cand_df[cand_df["name"] == "Seasonal Naive"]
                beat_baseline = False
                sn_rmse = None
                if not sn_row.empty:
                    sn_rmse = sn_row.iloc[0]["rmse"]
                    beat_baseline = winner["rmse"] < sn_rmse
                    
                verdict_info[ds_id] = {
                    "cfg": cfg,
                    "winner": winner,
                    "flag_no_pass": flag_no_pass,
                    "beat_baseline": beat_baseline,
                    "sn_rmse": sn_rmse
                }
                
                st.markdown(f"### Selected model: {cfg['short_name']}")
                st.markdown(f"#### **{winner['name']}** (`{winner['spec']}`)")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Test RMSE", f"{winner['rmse']:,.1f}")
                m2.metric("Test MAPE", f"{winner['mape']:.2f}%")
                m3.metric(
                    "Ljung-Box p-val",
                    f"{winner['lb_p']:.4f}" if not np.isnan(winner['lb_p']) else "N/A"
                )
                
                if flag_no_pass:
                    lb_msg = (
                        f"⚠️ **Note on Residuals:** None of the fitted models achieved a Ljung-Box p-value > 0.05, "
                        f"indicating subtle persistent autocorrelation in the training residuals. "
                        f"**{winner['name']}** was selected as it achieved the lowest out-of-sample RMSE ({winner['rmse']:,.1f})."
                    )
                else:
                    lb_msg = (
                        f"Its residuals successfully passed the Ljung-Box test for white noise (p = {winner['lb_p']:.4f} > 0.05), "
                        f"verifying that the model adequately captured the deterministic trend, calendar seasonality, and autoregressive dynamics."
                    )
                    
                baseline_msg = ""
                if sn_rmse is not None:
                    if beat_baseline:
                        pct_imp = ((sn_rmse - winner["rmse"]) / sn_rmse) * 100
                        baseline_msg = f"Importantly, it comfortably outperformed the Seasonal Naive baseline (RMSE: {sn_rmse:,.1f}) by **{pct_imp:.1f}%**."
                    else:
                        baseline_msg = f"However, it did not outperform the Seasonal Naive baseline (RMSE: {sn_rmse:,.1f}), suggesting high persistence in seasonal cycles."
                        
                explanation = (
                    f"**{winner['name']}** is chosen as the optimal classical forecasting model for **{cfg['name']}**. "
                    f"Over the hold-out test horizon of {t_size} periods, it registered a Mean Absolute Percentage Error of {winner['mape']:.2f}%. "
                    f"{lb_msg} {baseline_msg}"
                )
                
                st.info(explanation)
                
    if len(active_dataset_keys) == 2 and "ae_monthly" in verdict_info and "electricity_daily" in verdict_info:
        st.markdown("---")
        st.subheader("Why different models win")
        
        v1 = verdict_info["ae_monthly"]
        v2 = verdict_info["electricity_daily"]
        
        comp_text = f"""
        **Key Structural Contrasts Between Datasets:**
        1. **Temporal Frequency & Seasonal Periodicity:**
           - **NHS England A&E Attendances** operates at monthly frequency ($m=12$) with a strong secular upward trend driven by population aging, punctuated by extreme structural breaks (e.g. March 2020 COVID lockdowns). Models incorporating calendar covariates or flexible trend components ({v1['winner']['name']}) adapt effectively to these non-linearities.
           - **UK Electricity Demand** operates at daily granularity ($m=7$) governed primarily by sharp intra-week occupational cycles (high Monday–Friday industry load vs. weekend troughs), along with annual meteorological cycles. Models with strong seasonal memory ({v2['winner']['name']}) excel here because weekly calendar rhythms are exceptionally rigid.
        
        2. **Noise vs. Signal Ratio:**
           - Daily power demand exhibits high operational inertia and well-defined weather response curves, making short-horizon forecasting highly deterministic. In contrast, monthly A&E attendances are influenced by policy interventions, hospital capacity constraints, and epidemic waves, requiring models that avoid overfitting transitory spikes.
        
        3. **Verdict Summary:**
           - For **{v1['cfg']['short_name']}**, the recommended production model is **{v1['winner']['name']}** (RMSE: {v1['winner']['rmse']:,.1f}).
           - For **{v2['cfg']['short_name']}**, the recommended production model is **{v2['winner']['name']}** (RMSE: {v2['winner']['rmse']:,.1f}).
        """
        st.markdown(comp_text)
