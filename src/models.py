"""src/models.py
Classical and modern time series forecasting models, metric calculation, and residual diagnostics:
- Baselines: Naive, Seasonal Naive
- Exponential Smoothing: Simple Exponential Smoothing, Holt's Linear, Holt-Winters (Additive, Multiplicative)
- ARIMA Family (statsmodels): AR(p), MA(q), ARMA(p,q) on differenced/stationary series, ARIMA(p,d,q), SARIMA(p,d,q)(P,D,Q,m)
- Auto ARIMA (pmdarima stepwise search)
- Prophet (Facebook Prophet)
- Regression with ARIMA errors (SARIMAX with exogenous variables: UK bank holidays & COVID-19 dummy)
"""

import warnings
import numpy as np
import pandas as pd
import holidays
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from prophet import Prophet

warnings.filterwarnings("ignore")

# Unified color palette for all models across all plots
MODEL_COLORS = {
    "Actual Test": "#000000",
    "Naive": "#8c564b",
    "Seasonal Naive": "#e377c2",
    "Simple Exp Smoothing": "#bcbd22",
    "Holt's Linear": "#17becf",
    "Holt-Winters (Additive)": "#1f77b4",
    "Holt-Winters (Multiplicative)": "#aec7e8",
    "AR(p)": "#ff7f0e",
    "MA(q)": "#ffbb78",
    "ARMA(p,q)": "#2ca02c",
    "ARIMA(p,d,q)": "#98df8a",
    "SARIMA": "#d62728",
    "Auto ARIMA": "#9467bd",
    "Prophet": "#8c564b",
    "SARIMAX (Exogenous)": "#e377c2"
}

AVAILABLE_MODELS = [
    "Naive",
    "Seasonal Naive",
    "Simple Exp Smoothing",
    "Holt's Linear",
    "Holt-Winters (Additive)",
    "Holt-Winters (Multiplicative)",
    "AR(p)",
    "MA(q)",
    "ARMA(p,q)",
    "ARIMA(p,d,q)",
    "SARIMA",
    "Auto ARIMA",
    "Prophet",
    "SARIMAX (Exogenous)"
]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Computes test set RMSE, MAE, and MAPE."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    # Filter out potential NaNs
    valid = (~np.isnan(y_t)) & (~np.isnan(y_p))
    if not np.any(valid):
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    
    y_t = y_t[valid]
    y_p = y_p[valid]
    
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae = float(np.mean(np.abs(y_t - y_p)))
    
    # Avoid zero division in MAPE
    denom = np.where(y_t == 0, 1e-6, y_t)
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)
    
    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2)
    }


def compute_ljung_box_pvalue(residuals: pd.Series, lags: int = 10) -> float:
    """Computes Ljung-Box test p-value on residuals."""
    clean_res = residuals.dropna()
    if len(clean_res) <= lags + 2:
        return np.nan
    try:
        lb_res = acorr_ljungbox(clean_res, lags=[lags], return_df=True)
        return float(lb_res["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan


def create_exogenous_features(index: pd.DatetimeIndex, dataset_id: str) -> pd.DataFrame:
    """
    Creates exogenous regressors for SARIMAX:
    - UK Bank Holidays dummy (for both datasets)
    - COVID-19 period dummy (Mar 2020 to Mar 2021) for A&E
    """
    df_exog = pd.DataFrame(index=index)
    
    # UK bank holidays (England and Wales)
    uk_holidays = holidays.country_holidays("GB", subdiv="England")
    
    # Check each timestamp
    # If monthly, flag 1 if month contains at least one bank holiday
    # If daily, flag 1 if date is a bank holiday
    is_daily = (len(index) > 1 and (index[1] - index[0]).days <= 1)
    
    if is_daily:
        df_exog["bank_holiday"] = [1.0 if dt in uk_holidays else 0.0 for dt in index]
    else:
        # monthly
        holiday_in_month = []
        for dt in index:
            # check days in that month
            month_days = pd.date_range(start=dt, end=dt + pd.offsets.MonthEnd(1), freq="D")
            has_h = any(d in uk_holidays for d in month_days)
            holiday_in_month.append(1.0 if has_h else 0.0)
        df_exog["bank_holiday"] = holiday_in_month
        
    # COVID dummy
    if dataset_id == "ae_monthly":
        covid_mask = (index >= "2020-03-01") & (index <= "2021-03-31")
        df_exog["covid_dummy"] = [1.0 if c else 0.0 for c in covid_mask]
    else:
        # for electricity, also add covid lockdown period if overlaps (2021-01 to 2021-03)
        covid_mask = (index >= "2020-03-01") & (index <= "2021-04-30")
        df_exog["covid_dummy"] = [1.0 if c else 0.0 for c in covid_mask]
        
    return df_exog


def fit_and_forecast_model(
    model_name: str,
    train: pd.Series,
    test: pd.Series,
    frequency: str,
    seasonal_period: int,
    dataset_id: str,
    future_index: pd.DatetimeIndex | None = None
) -> dict:
    """
    Fits a specified forecasting model on train, generates out-of-sample forecast for test,
    calculates 95% prediction intervals, AIC (if applicable), and in-sample residuals.
    """
    # Future dates have no observed targets: do not invent values or report accuracy.
    evaluating = test is not None
    if not evaluating:
        if future_index is None or len(future_index) == 0:
            raise ValueError("Future forecasting requires a non-empty date index")
        if future_index[0] <= train.index[-1]:
            raise ValueError("Future dates must follow the final training observation")
        test = pd.Series(np.nan, index=future_index, dtype=float)
    h = len(test)
    test_index = test.index
    train_index = train.index
    
    forecast = pd.Series(index=test_index, dtype=float)
    lower_95 = pd.Series(index=test_index, dtype=float)
    upper_95 = pd.Series(index=test_index, dtype=float)
    residuals = pd.Series(index=train_index, dtype=float)
    aic = np.nan
    order_str = "-"
    
    # -------------------------------------------------------------
    # 1. NAIVE
    # -------------------------------------------------------------
    if model_name == "Naive":
        last_val = train.iloc[-1]
        forecast[:] = last_val
        residuals = train.diff().dropna()
        # Residual std for intervals
        sigma = residuals.std() if len(residuals) > 0 else 1.0
        # Naive interval grows with sqrt(h)
        steps = np.arange(1, h + 1)
        lower_95[:] = last_val - 1.96 * sigma * np.sqrt(steps)
        upper_95[:] = last_val + 1.96 * sigma * np.sqrt(steps)
        order_str = "y_{t-1}"

    # -------------------------------------------------------------
    # 2. SEASONAL NAIVE
    # -------------------------------------------------------------
    elif model_name == "Seasonal Naive":
        m = seasonal_period
        # Forecast repeats last m values
        rep_vals = train.iloc[-m:].values
        forecast_vals = [rep_vals[i % m] for i in range(h)]
        forecast[:] = forecast_vals
        residuals = (train - train.shift(m)).dropna()
        sigma = residuals.std() if len(residuals) > 0 else 1.0
        steps = np.arange(1, h + 1)
        # Seasonal naive standard error steps
        k_steps = np.floor((steps - 1) / m) + 1
        lower_95[:] = forecast.values - 1.96 * sigma * np.sqrt(k_steps)
        upper_95[:] = forecast.values + 1.96 * sigma * np.sqrt(k_steps)
        order_str = f"y_{{t-{m}}}"

    # -------------------------------------------------------------
    # 3. SIMPLE EXPONENTIAL SMOOTHING
    # -------------------------------------------------------------
    elif model_name == "Simple Exp Smoothing":
        fit_model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
        pred = fit_model.forecast(h)
        forecast[:] = pred.values
        residuals = fit_model.resid
        aic = getattr(fit_model, "aic", np.nan)
        sigma = residuals.std()
        steps = np.arange(1, h + 1)
        lower_95[:] = forecast.values - 1.96 * sigma * np.sqrt(1 + (fit_model.params.get("smoothing_level", 0.2)**2) * (steps - 1))
        upper_95[:] = forecast.values + 1.96 * sigma * np.sqrt(1 + (fit_model.params.get("smoothing_level", 0.2)**2) * (steps - 1))
        order_str = f"α={fit_model.params.get('smoothing_level', 0):.2f}"

    # -------------------------------------------------------------
    # 4. HOLT'S LINEAR TREND
    # -------------------------------------------------------------
    elif model_name == "Holt's Linear":
        fit_model = Holt(train, initialization_method="estimated").fit()
        pred = fit_model.forecast(h)
        forecast[:] = pred.values
        residuals = fit_model.resid
        aic = getattr(fit_model, "aic", np.nan)
        sigma = residuals.std()
        steps = np.arange(1, h + 1)
        lower_95[:] = forecast.values - 1.96 * sigma * np.sqrt(steps)
        upper_95[:] = forecast.values + 1.96 * sigma * np.sqrt(steps)
        order_str = f"α={fit_model.params.get('smoothing_level', 0):.2f}, β={fit_model.params.get('smoothing_trend', 0):.2f}"

    # -------------------------------------------------------------
    # 5. HOLT-WINTERS (ADDITIVE)
    # -------------------------------------------------------------
    elif model_name == "Holt-Winters (Additive)":
        m = seasonal_period
        fit_model = ExponentialSmoothing(
            train,
            seasonal_periods=m,
            trend="add",
            seasonal="add",
            initialization_method="estimated"
        ).fit()
        pred = fit_model.forecast(h)
        forecast[:] = pred.values
        residuals = fit_model.resid
        aic = getattr(fit_model, "aic", np.nan)
        sigma = residuals.std()
        steps = np.arange(1, h + 1)
        lower_95[:] = forecast.values - 1.96 * sigma * np.sqrt(steps / m + 1)
        upper_95[:] = forecast.values + 1.96 * sigma * np.sqrt(steps / m + 1)
        order_str = f"Add Trend + Add Season (m={m})"

    # -------------------------------------------------------------
    # 6. HOLT-WINTERS (MULTIPLICATIVE)
    # -------------------------------------------------------------
    elif model_name == "Holt-Winters (Multiplicative)":
        m = seasonal_period
        # Multiplicative requires non-zero positive
        train_shift = 0.0
        if (train <= 0).any():
            train_shift = abs(train.min()) + 1.0
            train_fit = train + train_shift
        else:
            train_fit = train
            
        fit_model = ExponentialSmoothing(
            train_fit,
            seasonal_periods=m,
            trend="add",
            seasonal="mul",
            initialization_method="estimated"
        ).fit()
        pred = fit_model.forecast(h) - train_shift
        forecast[:] = pred.values
        residuals = fit_model.resid
        aic = getattr(fit_model, "aic", np.nan)
        sigma = residuals.std()
        steps = np.arange(1, h + 1)
        lower_95[:] = forecast.values - 1.96 * sigma * np.sqrt(steps / m + 1)
        upper_95[:] = forecast.values + 1.96 * sigma * np.sqrt(steps / m + 1)
        order_str = f"Add Trend + Mul Season (m={m})"

    # -------------------------------------------------------------
    # 7. AR(p) - fitted on stationary/differenced series & converted back
    # -------------------------------------------------------------
    elif model_name == "AR(p)":
        # ARIMA(p, 1, 0) handles differencing to stationary scale and integrated inversion automatically
        # p=2 is suitable based on typical ACF/PACF partial cutoffs
        p_order = 2
        fit_model = ARIMA(train, order=(p_order, 1, 0)).fit()
        res_obj = fit_model.get_forecast(steps=h)
        pred = res_obj.predicted_mean
        ci = res_obj.conf_int(alpha=0.05)
        forecast[:] = pred.values
        lower_95[:] = ci.iloc[:, 0].values
        upper_95[:] = ci.iloc[:, 1].values
        residuals = fit_model.resid
        aic = fit_model.aic
        order_str = f"AR({p_order}) on diff series: ARIMA({p_order},1,0)"

    # -------------------------------------------------------------
    # 8. MA(q) - fitted on stationary/differenced series & converted back
    # -------------------------------------------------------------
    elif model_name == "MA(q)":
        # ARIMA(0, 1, q)
        q_order = 2
        fit_model = ARIMA(train, order=(0, 1, q_order)).fit()
        res_obj = fit_model.get_forecast(steps=h)
        pred = res_obj.predicted_mean
        ci = res_obj.conf_int(alpha=0.05)
        forecast[:] = pred.values
        lower_95[:] = ci.iloc[:, 0].values
        upper_95[:] = ci.iloc[:, 1].values
        residuals = fit_model.resid
        aic = fit_model.aic
        order_str = f"MA({q_order}) on diff series: ARIMA(0,1,{q_order})"

    # -------------------------------------------------------------
    # 9. ARMA(p,q) - fitted on stationary/differenced series & converted back
    # -------------------------------------------------------------
    elif model_name == "ARMA(p,q)":
        # ARIMA(p, 1, q)
        p_order, q_order = 1, 1
        fit_model = ARIMA(train, order=(p_order, 1, q_order)).fit()
        res_obj = fit_model.get_forecast(steps=h)
        pred = res_obj.predicted_mean
        ci = res_obj.conf_int(alpha=0.05)
        forecast[:] = pred.values
        lower_95[:] = ci.iloc[:, 0].values
        upper_95[:] = ci.iloc[:, 1].values
        residuals = fit_model.resid
        aic = fit_model.aic
        order_str = f"ARMA({p_order},{q_order}) on diff: ARIMA({p_order},1,{q_order})"

    # -------------------------------------------------------------
    # 10. ARIMA(p,d,q)
    # -------------------------------------------------------------
    elif model_name == "ARIMA(p,d,q)":
        # ARIMA(1, 1, 1) classical benchmark
        fit_model = ARIMA(train, order=(1, 1, 1)).fit()
        res_obj = fit_model.get_forecast(steps=h)
        pred = res_obj.predicted_mean
        ci = res_obj.conf_int(alpha=0.05)
        forecast[:] = pred.values
        lower_95[:] = ci.iloc[:, 0].values
        upper_95[:] = ci.iloc[:, 1].values
        residuals = fit_model.resid
        aic = fit_model.aic
        order_str = "ARIMA(1,1,1)"

    # -------------------------------------------------------------
    # 11. SARIMA(p,d,q)(P,D,Q,m)
    # -------------------------------------------------------------
    elif model_name == "SARIMA":
        m = seasonal_period
        # Standard SARIMA specification
        order = (1, 1, 1)
        s_order = (0, 1, 1, m)
        fit_model = SARIMAX(train, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        res_obj = fit_model.get_forecast(steps=h)
        pred = res_obj.predicted_mean
        ci = res_obj.conf_int(alpha=0.05)
        forecast[:] = pred.values
        lower_95[:] = ci.iloc[:, 0].values
        upper_95[:] = ci.iloc[:, 1].values
        residuals = fit_model.resid
        aic = fit_model.aic
        order_str = f"SARIMA(1,1,1)(0,1,1){m}"

    # -------------------------------------------------------------
    # 12. AUTO ARIMA (Hyndman-Khandakar stepwise search)
    # -------------------------------------------------------------
    elif model_name == "Auto ARIMA":
        m = seasonal_period
        # Use pmdarima auto_arima
        # Keep seasonal search efficient (m=12 for monthly, m=7 for daily)
        auto_fit = pm.auto_arima(
            train,
            seasonal=True,
            m=m,
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            max_d=2,
            max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        pred, ci = auto_fit.predict(n_periods=h, return_conf_int=True, alpha=0.05)
        forecast[:] = pred.values if hasattr(pred, "values") else pred
        lower_95[:] = ci[:, 0]
        upper_95[:] = ci[:, 1]
        residuals = pd.Series(auto_fit.resid(), index=train_index)
        aic = auto_fit.aic()
        order_str = str(auto_fit.order) + ("x" + str(auto_fit.seasonal_order) if auto_fit.seasonal_order else "")

    # -------------------------------------------------------------
    # 13. PROPHET
    # -------------------------------------------------------------
    elif model_name == "Prophet":
        df_prophet_train = pd.DataFrame({
            "ds": train.index,
            "y": train.values
        })
        m = Prophet(
            yearly_seasonality=True if frequency == "D" or len(train) > 24 else False,
            weekly_seasonality=True if frequency == "D" else False,
            daily_seasonality=False,
            interval_width=0.95
        )
        m.fit(df_prophet_train)
        
        future = pd.DataFrame({"ds": test.index})
        fcst = m.predict(future)
        
        forecast[:] = fcst["yhat"].values
        lower_95[:] = fcst["yhat_lower"].values
        upper_95[:] = fcst["yhat_upper"].values
        
        # in-sample residuals
        in_sample = m.predict(df_prophet_train)
        residuals = pd.Series(df_prophet_train["y"].values - in_sample["yhat"].values, index=train_index)
        aic = np.nan
        order_str = "Prophet (Trend+Yearly" + ("+Weekly" if frequency == "D" else "") + ")"

    # -------------------------------------------------------------
    # 14. REGRESSION WITH ARIMA ERRORS (SARIMAX with Exogenous)
    # -------------------------------------------------------------
    elif model_name == "SARIMAX (Exogenous)":
        m = seasonal_period
        full_index = train.index.union(test.index)
        exog_full = create_exogenous_features(full_index, dataset_id)
        exog_train = exog_full.loc[train.index]
        exog_test = exog_full.loc[test.index]
        
        fit_model = SARIMAX(
            train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, m),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        res_obj = fit_model.get_forecast(steps=h, exog=exog_test)
        pred = res_obj.predicted_mean
        ci = res_obj.conf_int(alpha=0.05)
        forecast[:] = pred.values
        lower_95[:] = ci.iloc[:, 0].values
        upper_95[:] = ci.iloc[:, 1].values
        residuals = fit_model.resid
        aic = fit_model.aic
        exog_vars = ", ".join(exog_train.columns)
        order_str = f"SARIMAX(1,1,1)(0,1,1){m} + [{exog_vars}]"

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Metrics
    metrics = compute_metrics(test, forecast) if evaluating else None
    lb_pvalue = compute_ljung_box_pvalue(residuals, lags=min(10, len(residuals) // 3))
    
    return {
        "model_name": model_name,
        "forecast": forecast,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "residuals": residuals,
        "metrics": metrics,
        "aic": round(aic, 2) if not np.isnan(aic) else np.nan,
        "ljung_box_pvalue": round(lb_pvalue, 4) if not np.isnan(lb_pvalue) else np.nan,
        "order_str": order_str
    }
