# Forecast Studio — React interface

A React dashboard using the existing Python analysis and models. Node.js serves the built frontend and proxies the local Python JSON API. Analysis, future forecasting, capacity planning, reliability reports, and both datasets are available; compare mode shows both datasets. The Streamlit app remains at `../app.py`.

## Quick start

From the project root, run `./run.sh`. From this frontend directory, run `../run.sh`. Open http://localhost:3000.

## Run locally

From this directory, with Node.js and pnpm installed:

```sh
pnpm install
pnpm build
pnpm start
```

Open http://localhost:3000. The Node server starts `../.venv/bin/python ../api.py` automatically. Install the root Python requirements into that environment if needed. Set `ATSA_PYTHON` to use a different Python executable; set `PORT` to change the UI port. The Python API listens on 127.0.0.1:8765. Both services bind to loopback only.

For a new checkout or extracted ZIP, follow the first-time setup in `../SETUP.md`.

## Development

Run `../.venv/bin/python ../api.py` in one terminal and `pnpm dev` in another. Vite proxies `/api` to the Python service. Restart the production server after editing Python code; rebuild the frontend after editing React or CSS.

## Behavior

- Explore: history, decomposition (STL/additive/multiplicative), seasonal indices, ACF, PACF, and ADF stationarity.
- Forecast: holdout predictions, model legend toggles, and prediction intervals.
- Evaluate: metrics, selectable-model residual diagnostics, and the existing lowest-RMSE/residual-check decision rule.
- Settings apply together; cancel leaves current settings intact. Default selected models are Seasonal Naive, Holt-Winters Additive, and ARIMA.
- Fitted results are cached in memory by dataset, holdout, and model. First fits can take longer, especially Auto ARIMA and Prophet. Model failures are shown without discarding successful fits.
- Export analysis downloads the current API result as JSON. Plotly's chart toolbar exports PNG images.

No cloud hosting or upload is required. The API is intended for this local project; it has no authentication or multi-user scheduling.

## Future forecast

Open **Forecast → Future forecast**. Select a year, the final month to include, and one model, then click **Generate forecast**. This fits on the complete dataset and forecasts every intervening observation from the actual cutoff. Tables show the forecast months in the chosen year. Future predictions do not have test RMSE/MAE/MAPE because actual values are unknown; use Holdout comparison to validate the model first.

- NHS: monthly attendance predictions and model-specific 95% intervals.
- Electricity: daily GWh predictions, monthly energy totals, and downloadable daily details. Monthly-total uncertainty intervals are not inferred by adding daily bounds.
- Download monthly CSV or, for electricity, daily CSV including daily intervals.
- The date selector allows up to five years beyond each dataset's cutoff. Model intervals may be approximate; farther horizons are less reliable. The 7-day electricity seasonality does not represent every annual effect.
- The settings holdout controls apply to evaluation, not to the future horizon. Future forecasting uses its own explicit model selector.

After Python source changes, restart the running app. The launcher detects outdated Python code and prints a restart instruction.

## Capacity planning

Generate a future forecast, then move the planning-capacity slider. NHS capacity is attendances per month; electricity capacity is GWh per day. The dashboard shows affected months, total positive shortfalls, the worst period, a capacity line, and interval-based flags. Values are user assumptions, not measured operational limits or probabilities. Electricity month rows summarize daily overloads; spare capacity on other days is not subtracted from shortfalls.

## Forecast reliability

Open **Evaluate → Forecast reliability**. Choose 2–6 folds, a horizon per dataset, and up to five models; Seasonal Naive is always included. Each fold expands training history and evaluates the next non-overlapping block. Rankings use pooled RMSE across all test observations, and require every fold to succeed. The report includes MAE, MAPE, fold-RMSE standard deviation, worst-fold RMSE, baseline improvement, exact test dates, and fit failures. It assesses historical accuracy separately from the single-holdout Ljung–Box model-selection rule. Repeated selection on backtests still needs an untouched final test.

Checks from the project root: `.venv/bin/python -m unittest discover -s tests`. Capacity calculations: `node frontend/test-capacity.mjs` from the project root.
