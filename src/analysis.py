"""src/analysis.py
Time series statistical analysis modules:
- Moving average
- STL Decomposition (additive & multiplicative classical, plus STL)
- Seasonal Index
- ACF and PACF computation with confidence bands
- Augmented Dickey-Fuller (ADF) stationarity tests
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf, pacf, adfuller


def compute_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Computes rolling moving average."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def compute_decomposition(series: pd.Series, period: int, model_type: str = "additive"):
    """
    Computes decomposition of the time series.
    Supports:
      - 'additive': Classical additive decomposition
      - 'multiplicative': Classical multiplicative decomposition (requires strictly positive series)
      - 'stl': LOESS-based STL decomposition
    Returns dict containing observed, trend, seasonal, and resid series.
    """
    clean_series = series.dropna()
    
    if model_type == "stl":
        # STL requires robust period
        stl = STL(clean_series, period=period, robust=True)
        res = stl.fit()
        return {
            "type": "STL (LOESS)",
            "observed": clean_series,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid
        }
    
    # Classical decomposition
    if model_type == "multiplicative":
        # Check for non-positive values
        if (clean_series <= 0).any():
            # Offset slightly if needed or fall back
            clean_series = clean_series - clean_series.min() + 1.0
        res = seasonal_decompose(clean_series, model="multiplicative", period=period, extrapolate_trend="freq")
        return {
            "type": "Classical Multiplicative",
            "observed": clean_series,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid
        }
    else:
        res = seasonal_decompose(clean_series, model="additive", period=period, extrapolate_trend="freq")
        return {
            "type": "Classical Additive",
            "observed": clean_series,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid
        }


def compute_seasonal_indices(series: pd.Series, frequency: str, period: int) -> pd.DataFrame:
    """
    Computes normalized seasonal indices:
    - For monthly data (MS): by month (Jan - Dec)
    - For daily data (D): by day of the week (Monday - Sunday)
    Uses ratio-to-moving-average method.
    """
    trend = series.rolling(window=period, center=True, min_periods=1).mean()
    detrended = series / trend.replace(0, np.nan)
    
    df_temp = pd.DataFrame({"detrended": detrended, "date": series.index})
    
    if frequency == "MS":
        df_temp["period_num"] = df_temp["date"].dt.month
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df_temp["period_label"] = df_temp["period_num"].apply(lambda m: month_names[m - 1])
        grouped = df_temp.groupby(["period_num", "period_label"])["detrended"].mean().reset_index()
        # Normalize so average is 1.0 (or 100%)
        mean_val = grouped["detrended"].mean()
        grouped["index_val"] = grouped["detrended"] / mean_val
        grouped["percentage"] = (grouped["index_val"] - 1.0) * 100.0
        grouped = grouped.sort_values("period_num").reset_index(drop=True)
        return grouped
    else:
        # Daily data: day of week
        df_temp["period_num"] = df_temp["date"].dt.dayofweek
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_temp["period_label"] = df_temp["period_num"].apply(lambda d: day_names[d])
        grouped = df_temp.groupby(["period_num", "period_label"])["detrended"].mean().reset_index()
        mean_val = grouped["detrended"].mean()
        grouped["index_val"] = grouped["detrended"] / mean_val
        grouped["percentage"] = (grouped["index_val"] - 1.0) * 100.0
        grouped = grouped.sort_values("period_num").reset_index(drop=True)
        return grouped


def compute_acf_pacf(series: pd.Series, nlags: int = 36, alpha: float = 0.05):
    """
    Computes ACF, PACF and confidence intervals for a given series.
    """
    clean_series = series.dropna()
    n = len(clean_series)
    max_lags = min(nlags, n // 2 - 1)
    if max_lags < 2:
        max_lags = 2
        
    acf_vals, acf_confint = acf(clean_series, nlags=max_lags, alpha=alpha)
    pacf_vals, pacf_confint = pacf(clean_series, nlags=max_lags, alpha=alpha, method="yw")
    
    # Bartlett 95% confidence interval line: +/- 1.96 / sqrt(n)
    conf_bound = 1.96 / np.sqrt(n)
    
    lags = np.arange(len(acf_vals))
    return {
        "lags": lags,
        "acf": acf_vals,
        "acf_confint": acf_confint,
        "pacf": pacf_vals,
        "pacf_confint": pacf_confint,
        "conf_bound": conf_bound,
        "n": n
    }


def perform_adf_test(series: pd.Series, name: str) -> dict:
    """
    Runs Augmented Dickey-Fuller test on the given series.
    Returns test statistics, p-value, critical values, and stationarity conclusion.
    """
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return {
            "Series": name,
            "Test Statistic": np.nan,
            "p-value": np.nan,
            "Lags Used": 0,
            "Observations": len(clean_series),
            "1% Critical": np.nan,
            "5% Critical": np.nan,
            "10% Critical": np.nan,
            "Stationary (p < 0.05)": "Insufficient data"
        }
        
    res = adfuller(clean_series, autolag="AIC")
    stat = res[0]
    pval = res[1]
    usedlag = res[2]
    nobs = res[3]
    crit = res[4]
    
    is_stationary = "Yes" if pval < 0.05 else "No"
    
    return {
        "Series": name,
        "Test Statistic": round(float(stat), 4),
        "p-value": round(float(pval), 4),
        "Lags Used": usedlag,
        "Observations": nobs,
        "1% Critical": round(float(crit["1%"]), 4),
        "5% Critical": round(float(crit["5%"]), 4),
        "10% Critical": round(float(crit["10%"]), 4),
        "Stationary (p < 0.05)": is_stationary
    }


def get_stationarity_summary(series: pd.Series, seasonal_period: int) -> pd.DataFrame:
    """
    Performs ADF test on:
    1. Original Series
    2. First-Differenced Series (d=1)
    3. Seasonally-Differenced Series (D=1, lag=m)
    4. Both Differenced Series (d=1, D=1)
    """
    clean = series.dropna()
    diff1 = clean.diff().dropna()
    diff_seas = clean.diff(seasonal_period).dropna()
    diff_both = diff1.diff(seasonal_period).dropna()
    
    rows = [
        perform_adf_test(clean, "Original Series"),
        perform_adf_test(diff1, "First Differenced (d=1)"),
        perform_adf_test(diff_seas, f"Seasonally Differenced (lag={seasonal_period})"),
        perform_adf_test(diff_both, f"First + Seasonal Differenced (d=1, m={seasonal_period})")
    ]
    
    return pd.DataFrame(rows)
