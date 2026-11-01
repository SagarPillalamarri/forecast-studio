# Applied Time Series Analysis (ATSA) Course Project

**New computer or extracted ZIP?** Follow [SETUP.md](SETUP.md) for the React dashboard. For GitHub upload commands, see [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md).

An interactive Streamlit comparative forecasting application built in Python for an **Applied Time Series Analysis** course project. The application compares two real-world empirical time series datasets and fits classical and modern forecasting models to determine the optimal econometric model for each series.

---

## 📊 Datasets

1. **NHS England Monthly A&E Attendances**
   - **Source:** NHS England official monthly time series collection (*A&E Attendances and Emergency Admissions*).
   - **Frequency:** Monthly (`MS`), $m=12$.
   - **Time Horizon:** August 2010 to April 2026 (189 monthly observations).
   - **Characteristics:** Pronounced secular growth, acute pandemic interruption (COVID-19 in Spring 2020), and recurring winter/summer hospital demand seasonality.

2. **UK Daily Electricity Demand (NESO)**
   - **Source:** National Energy System Operator (NESO / National Grid ESO) Historic Demand Data.
   - **Frequency:** Daily (`D`), $m=7$.
   - **Time Horizon:** 2021 to 2024 (1,461 daily observations).
   - **Characteristics:** Strong 7-day cyclical calendar effects (weekday industrial peaks vs. weekend troughs) and annual meteorological space-heating patterns.

---

## 🛠️ Project Structure

```text
ATSA_Project_Sem 7/
│
├── app.py                  # Streamlit user interface & interactive controllers
├── prepare_data.py         # Data preparation, downloading, and cleaning pipeline
├── requirements.txt        # Pinned project dependencies
├── README.md               # Setup and project documentation
│
├── data/
│   ├── ae_monthly.csv      # Cleaned NHS A&E dataset (date, value)
│   ├── electricity_daily.csv # Cleaned NESO Electricity dataset (date, value)
│   └── raw/                # Raw downloaded archives and manual cache
│
└── src/
    ├── __init__.py
    ├── data.py             # Dataset configuration dictionary & cached data loaders
    ├── analysis.py         # STL decomposition, seasonal index, ACF, PACF, ADF tests
    ├── models.py           # Forecasting models, metric calculation, & residual tests
    └── plots.py            # High-performance Plotly visualization generators
```

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python prepare_data.py
```
*Note: If automatic API downloads ever fail, the script provides clear instructions on where to download the raw files manually into `data/raw/`.*

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 🖥️ Layout & Interactive Features

- **Executive Analytics Interface:**
  - Light canvas, white summary cards, navy typography, and blue chart accents.
  - Sidebar workspace navigation groups the ten analysis views into **Explore**, **Forecast**, and **Evaluate**.
  - Dataset summary cards show actual observations, date coverage, latest values, and holdout settings.
  - NHS A&E is the initial dataset; comparison mode remains available.
- **Sidebar Controls:**
  - **View Mode Selector:** Switch between *Dataset 1 only*, *Dataset 2 only*, or *Both side by side* (`st.columns(2)`).
  - **Moving Average Slider:** Adjust smoothing window $k$ ($3$ to $30$).
  - **Test Set Size:** Pre-filled with standard hold-outs (12 months for NHS A&E, 60 days for Electricity), fully editable.
  - **Model Checkboxes:** Select specific forecasting models inside the expandable Benchmark models section.
- **Caching:** Data loading (`@st.cache_data`) and model fitting are cached so clicking plot buttons never refits models.

---

## 📈 Explanation of Analysis Tabs

1. **Raw + Moving Average (*Explore → Historical series*)**
   Overlays a centered rolling moving average on the raw time series to distinguish persistent long-term secular trends from high-frequency noise.
2. **Decomposition (`2. Decomposition`)**
   Deconstructs the series into observed, trend, seasonal, and residual components. Includes toggles for **STL (LOESS)**, **Classical Additive**, and **Classical Multiplicative** decomposition.
3. **Seasonal Index (`3. Seasonal Index`)**
   Bar chart of normalized seasonal deviations (% departure from baseline) across months (Jan–Dec for A&E) or days of the week (Mon–Sun for Electricity).
4. **Autocorrelation Function (`4. ACF`)**
   ACF correlogram across the original series, first-differenced series ($d=1$), and seasonally differenced series ($d=1, D=1$), with Bartlett 95% confidence bands.
5. **Partial Autocorrelation Function (`5. PACF`)**
   PACF correlograms identifying candidate AR orders and detecting moving average signatures.
6. **Stationarity (`6. Stationarity (ADF)`)**
   Augmented Dickey-Fuller (ADF) unit root test summary table reporting test statistics, $p$-values, critical values (1%, 5%, 10%), and stationarity conclusions.
7. **Forecast Comparison (`7. Forecast Comparison`)**
   Interactive Plotly chart comparing actual test data against out-of-sample forecasts with shaded 95% prediction intervals. Models can be toggled on/off in the legend.
8. **Model Metrics (`8. Model Metrics`)**
   Evaluates models on the test set using RMSE, MAE, MAPE (%), AIC, and Ljung-Box test $p$-values. The top-performing model is dynamically highlighted in blue.
9. **Residual Diagnostics (`9. Residual Diagnostics`)**
   Four-quadrant diagnostic panel for a chosen model (defaults to best model): standardized residuals over time, histogram with overlaid standard normal curve, residual ACF, and Normal Q-Q plot.
10. **Verdict (`10. Verdict`)**
    Declares the best model per dataset using the course decision rule, reports baseline comparison against Seasonal Naive, and provides automated plain-English econometric explanations.

---

## 🧮 Model Evaluation & Decision Rule

- **Candidate Models:**
  - *Baselines:* Naive, Seasonal Naive ($y_{t-m}$)
  - *Exponential Smoothing:* Simple Exponential Smoothing, Holt's Linear Trend, Holt-Winters Additive & Multiplicative
  - *ARIMA Family:* AR(p), MA(q), ARMA(p,q) on stationary scale, ARIMA(p,d,q), SARIMA(p,d,q)(P,D,Q,m)
  - *Auto ARIMA:* Hyndman-Khandakar stepwise heuristic search
  - *Prophet:* Generalized additive modeling with changepoint detection
  - *Regression with ARIMA Errors (SARIMAX):* Incorporating UK Bank Holidays and COVID-19 pandemic intervention dummies
- **Best Model Selection Rule:** Lowest test RMSE among models whose residuals pass the Ljung-Box test for white noise ($p > 0.05$). If none pass, selects lowest RMSE and displays an explanatory notice.

## React / Node.js interface

The custom React dashboard is in [`frontend/`](frontend/README.md). It uses the same Python analysis and forecasting functions through `api.py`. From `frontend`, run `pnpm install`, `pnpm build`, and `pnpm start`, then open http://localhost:3000. The original Streamlit interface remains available with `streamlit run app.py`.

### Quick start

From the project root in the VS Code terminal, run `./run.sh`, then open http://localhost:3000. The launcher finds Node.js automatically, starts both services, and detects an app that is already running. Press Ctrl+C in the running server terminal to stop it.

Electricity CSV values are daily sums of half-hourly MW demand readings. The shared loader converts them to daily energy in GWh using `sum × 0.5 / 1000`; source CSVs remain unchanged. Electricity charts, forecasts, RMSE, and MAE therefore use GWh.

### Future-year forecasting

In the React app, open **Forecast → Future forecast**, choose a year (for example 2027 or 2028), the final month, and a model. Generate predictions fitted on all available history. The view includes monthly tables, uncertainty bands, peak-month summaries, and CSV exports. Electricity also supports daily detail in GWh. Targets are limited to five years beyond the dataset cutoff; unknown future actuals are never scored as test data.

### Decision support

Future forecasts include a capacity-planning slider with overload flags and positive shortfall totals. **Evaluate → Forecast reliability** adds expanding-window backtests, fold error variability, worst-period performance, and accuracy relative to Seasonal Naive. See the frontend guide for definitions and test commands.
