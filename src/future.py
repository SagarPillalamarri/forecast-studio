"""Calendar-based forecasts beyond the last observation, with no fabricated actuals."""
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.data import DATASET_CONFIGS, load_dataset
from src.models import AVAILABLE_MODELS, fit_and_forecast_model


def forecast_limits(dataset):
    frame = load_dataset(dataset)
    last = frame.index[-1]
    # Five years from the actual data cutoff, not five years from today's date.
    maximum = last + pd.DateOffset(years=5)
    return {'firstForecastMonth': str((last.to_period('M') + 1)),
            'lastForecastMonth': str(maximum.to_period('M')),
            'lastObservation': last.strftime('%Y-%m-%d')}


def future_index(dataset, target):
    if not isinstance(target, str) or len(target) != 7:
        raise ValueError('Choose a target month in YYYY-MM format')
    try:
        year, month = target.split('-')
        if not year.isdigit() or not month.isdigit() or not 1 <= int(month) <= 12:
            raise ValueError()
        period = pd.Period(target, freq='M')
    except (ValueError, TypeError):
        raise ValueError('Choose a valid target month in YYYY-MM format') from None
    limits = forecast_limits(dataset)
    if target < limits['firstForecastMonth'] or target > limits['lastForecastMonth']:
        raise ValueError(f"Choose a month from {limits['firstForecastMonth']} through {limits['lastForecastMonth']}. Forecasts extend at most five years beyond the dataset.")
    cfg = DATASET_CONFIGS[dataset]
    last = load_dataset(dataset).index[-1]
    start = last + pd.tseries.frequencies.to_offset(cfg['frequency'])
    end = period.start_time if cfg['frequency'] == 'MS' else period.end_time.normalize()
    return pd.date_range(start, end, freq=cfg['frequency'])


@lru_cache(maxsize=32)
def fit_future(dataset, target, model):
    if model not in AVAILABLE_MODELS:
        raise ValueError('Choose a valid forecasting model')
    index = future_index(dataset, target)
    cfg = DATASET_CONFIGS[dataset]
    return fit_and_forecast_model(model_name=model, train=load_dataset(dataset)['value'],
                                  test=None, frequency=cfg['frequency'],
                                  seasonal_period=cfg['seasonal_period'], dataset_id=dataset,
                                  future_index=index)


def build_future(dataset, target, model):
    cfg = DATASET_CONFIGS[dataset]
    series = load_dataset(dataset)['value']
    result = fit_future(dataset, target, model)
    prediction = result['forecast']
    lower, upper = result['lower_95'], result['upper_95']
    if not np.isfinite(prediction.to_numpy()).all():
        raise ValueError('The model returned invalid forecasts. Choose a different model.')
    target_year = int(target[:4])
    visible = prediction[prediction.index.year == target_year]
    rows, daily = [], []
    for period, values in visible.groupby(visible.index.to_period('M')):
        if cfg['frequency'] == 'MS':
            stamp = values.index[0]
            row = {'Month': str(period), 'Forecast attendances': float(values.iloc[0]),
                   'Lower 95%': float(lower.loc[stamp]), 'Upper 95%': float(upper.loc[stamp])}
        else:
            row = {'Month': str(period), 'Total energy (GWh)': float(values.sum()),
                   'Average daily energy (GWh)': float(values.mean()), 'Days': len(values)}
        rows.append(row)
    if cfg['frequency'] == 'D':
        daily = [{'Date': stamp.strftime('%Y-%m-%d'), 'Forecast (GWh)': float(value),
                  'Lower 95% (GWh)': float(lower.loc[stamp]), 'Upper 95% (GWh)': float(upper.loc[stamp])}
                 for stamp, value in visible.items()]
    # Forecast lines include every intervening period even for targets several years away.
    history = series.iloc[-(36 if cfg['frequency'] == 'MS' else 180):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode='lines', name='Observed history',
                            line=dict(color='#8e9eb4', width=1.5)))
    fig.add_trace(go.Scatter(x=list(prediction.index) + list(prediction.index)[::-1],
                            y=list(upper.values) + list(lower.values)[::-1], fill='toself',
                            fillcolor='rgba(54,98,227,0.10)', line=dict(color='rgba(0,0,0,0)'),
                            hoverinfo='skip', name='95% prediction interval', showlegend=True))
    fig.add_trace(go.Scatter(x=prediction.index, y=prediction.values, mode='lines', name=model,
                            line=dict(color='#3662e3', width=2)))
    fig.add_vline(x=series.index[-1].timestamp()*1000, line_dash='dot', line_color='#9aa8bd')
    fig.update_layout(xaxis=dict(title=''), yaxis=dict(title=cfg['unit']), hovermode='x unified',
                      annotations=[dict(x=series.index[-1], y=1.05, xref='x', yref='paper', text='Last observation',
                                        showarrow=False, font=dict(size=10, color='#8793a5'))])
    selected_total = float(visible.sum())
    peak = max(rows, key=lambda row: row.get('Forecast attendances', row.get('Total energy (GWh)', 0)))
    warnings = []
    if (prediction < 0).any():
        warnings.append('This model predicts negative values. They have not been clipped; consider another model for non-negative demand.')
    if (lower < 0).any():
        warnings.append('Some lower interval bounds are negative. These are unconstrained model intervals, not physically possible demand.')
    note = ('Monthly totals sum the daily point forecasts. The chart and daily table show daily 95% intervals; '
            'monthly-total intervals are not estimated by adding daily bounds.' if daily else
            'The table reports monthly forecasts and model-specific 95% prediction intervals.')
    return {'model': model, 'specification': result['order_str'], 'target': target,
            'origin': series.index[-1].strftime('%Y-%m-%d'), 'firstForecast': prediction.index[0].strftime('%Y-%m-%d'),
            'forecastEnd': prediction.index[-1].strftime('%Y-%m-%d'), 'steps': len(prediction),
            'trainingObservations': len(series), 'monthlyRows': rows, 'dailyRows': daily,
            'selectedYear': target_year, 'selectedTotal': selected_total,
            'peakMonth': peak['Month'], 'latestForecast': float(prediction.iloc[-1]),
            'warnings': warnings, 'note': note, 'figure': fig,
            'accuracyNote': 'Future actuals are unavailable, so RMSE, MAE, and MAPE are not reported. Compare models on the holdout view before choosing one for future forecasting.'}
