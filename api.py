"""Local JSON API sharing the existing ATSA analysis and forecasting modules."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib_cache"))
import json
import hashlib
import math
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
import pandas as pd
from src.data import DATASET_CONFIGS, load_dataset, get_train_test_split
from src.analysis import compute_moving_average, compute_decomposition, compute_seasonal_indices, compute_acf_pacf, get_stationarity_summary
from src.models import AVAILABLE_MODELS, fit_and_forecast_model
from src.plots import plot_raw_moving_average, plot_decomposition, plot_seasonal_index, plot_correlation_function, plot_forecast_comparison, plot_residual_diagnostics

from src.future import build_future, forecast_limits
from src.validation import rolling_validation

_root = Path(__file__).parent
SOURCE_REVISION = hashlib.sha256(b''.join(path.read_bytes() for path in [Path(__file__)] + sorted((_root / 'src').glob('*.py')))).hexdigest()
fit_lock = threading.Lock()

@lru_cache(maxsize=96)
def fitted(dataset, holdout, name):
    cfg = DATASET_CONFIGS[dataset]
    train, test = get_train_test_split(load_dataset(dataset), holdout)
    return fit_and_forecast_model(name, train['value'], test['value'], cfg['frequency'], cfg['seasonal_period'], dataset)

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def analyze(params):
    dataset = params.get('dataset', ['ae_monthly'])[0]
    if dataset not in DATASET_CONFIGS:
        raise ValueError('Unknown dataset')
    cfg = DATASET_CONFIGS[dataset]
    df = load_dataset(dataset)
    series = df['value']
    window = int(params.get('window', ['7'])[0])
    holdout = int(params.get('holdout', [str(cfg['default_test_size'])])[0])
    if not 3 <= window <= 30 or not 1 <= holdout < len(df):
        raise ValueError('Invalid moving average or holdout setting')
    view = params.get('view', ['history'])[0]
    result = {'dataset': dataset, 'name': cfg['short_name'], 'unit': cfg['unit'], 'frequency': 'Monthly' if cfg['frequency'] == 'MS' else 'Daily', 'period': cfg['seasonal_period'], 'observations': len(df), 'start': df.index[0].strftime('%Y-%m-%d'), 'end': df.index[-1].strftime('%Y-%m-%d'), 'latest': float(series.iloc[-1]), 'mean': float(series.mean()), 'previous': float(series.iloc[-2]), 'holdout': holdout, 'figures': [], 'rows': [], 'errors': []}
    figures = []
    if view == 'history':
        figures = [plot_raw_moving_average(series, compute_moving_average(series, window), window, cfg['name'], cfg['unit'])]
    elif view == 'future':
        target = params.get('target', [''])[0]
        model = params.get('futureModel', ['Holt-Winters (Additive)'])[0]
        with fit_lock:
            future = build_future(dataset, target, model)
        figures = [future.pop('figure')]
        result['future'] = future
    elif view == 'validation':
        names = json.loads(params.get('models', ['["Seasonal Naive","Holt-Winters (Additive)"]'])[0])
        folds = int(params.get('folds', ['3'])[0])
        horizon = int(params.get('validationHorizon', ['12' if dataset == 'ae_monthly' else '30'])[0])
        with fit_lock:
            validation = rolling_validation(dataset, names, folds, horizon)
        figures = [validation.pop('figure')]
        result['validation'] = validation
    elif view == 'decomposition':
        mode = params.get('method', ['stl'])[0]
        if mode not in ['stl', 'additive', 'multiplicative']:
            raise ValueError('Invalid decomposition method')
        figures = [plot_decomposition(compute_decomposition(series, cfg['seasonal_period'], mode), cfg['name'], cfg['unit'])]
    elif view == 'seasonal':
        figures = [plot_seasonal_index(compute_seasonal_indices(series, cfg['frequency'], cfg['seasonal_period']), cfg['name'], cfg['period_label'])]
    elif view in ['acf', 'pacf']:
        figures = [plot_correlation_function(compute_acf_pacf(series), compute_acf_pacf(series.diff().dropna()), compute_acf_pacf(series.diff().diff(cfg['seasonal_period']).dropna()), view.upper(), cfg['name'])]
    elif view == 'stationarity':
        result['rows'] = get_stationarity_summary(series, cfg['seasonal_period']).to_dict('records')
    elif view in ['forecast', 'metrics', 'residuals', 'verdict']:
        names = json.loads(params.get('models', ['["Seasonal Naive","Holt-Winters (Additive)"]'])[0])
        if not isinstance(names, list) or not names or any(name not in AVAILABLE_MODELS for name in names):
            raise ValueError('Select valid benchmark models')
        results = {}
        with fit_lock:
            for name in names:
                try:
                    results[name] = fitted(dataset, holdout, name)
                except Exception as error:
                    result['errors'].append({'model': name, 'message': str(error)})
        if not results:
            raise ValueError('None of the selected models could be fitted. Try a baseline model or a different holdout.')
        for name, fit in results.items():
            p = fit['ljung_box_pvalue']
            result['rows'].append({'Model': name, 'RMSE': fit['metrics']['RMSE'], 'MAE': fit['metrics']['MAE'], 'MAPE (%)': fit['metrics']['MAPE'], 'AIC': fit['aic'], 'Ljung–Box p': p, 'Residual check': 'Pass' if p > .05 else 'Fail', 'Specification': fit['order_str']})
        valid = [row for row in result['rows'] if row['RMSE'] is not None and math.isfinite(row['RMSE'])]
        if not valid:
            raise ValueError('Selected models returned no finite accuracy metrics')
        passing = [row for row in valid if row['Residual check'] == 'Pass']
        winner = min(passing or valid, key=lambda row: row['RMSE'])
        result['winner'] = winner
        result['noPassingModel'] = not passing
        baseline = results.get('Seasonal Naive')
        if baseline and baseline['metrics']['RMSE'] > 0:
            result['baselineImprovement'] = (1 - winner['RMSE']/baseline['metrics']['RMSE']) * 100
        if view == 'forecast':
            train, test = get_train_test_split(df, holdout)
            figures = [plot_forecast_comparison(train['value'], test['value'], results, cfg['name'], cfg['unit'])]
        elif view == 'residuals':
            chosen = params.get('diagnostic', [winner['Model']])[0]
            chosen = chosen if chosen in results else winner['Model']
            result['diagnostic'] = chosen
            fit = results[chosen]
            result['residualSummary'] = {'mean': float(fit['residuals'].mean()), 'std': float(fit['residuals'].std()), 'p': fit['ljung_box_pvalue']}
            figures = [plot_residual_diagnostics(fit['residuals'], chosen, cfg['short_name'])]
    else:
        raise ValueError('Unknown analysis view')
    for fig in figures:
        result['figures'].append(json.loads(fig.to_json()))
    return clean(result)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/api/catalog':
                payload = {'datasets': [{'id': key, 'name': cfg['short_name'], 'unit': cfg['unit'], 'frequency': cfg['frequency'], 'defaultHoldout': cfg['default_test_size'], **forecast_limits(key)} for key, cfg in DATASET_CONFIGS.items()], 'models': AVAILABLE_MODELS}
            elif parsed.path == '/api/analyze':
                payload = analyze(parse_qs(parsed.query))
            elif parsed.path == '/api/health':
                payload = {'status': 'ok', 'features': ['future-forecast', 'rolling-validation'], 'revision': SOURCE_REVISION}
            else:
                self.respond(404, {'error': 'Not found'})
                return
            self.respond(200, payload)
        except ValueError as error:
            self.respond(400, {'error': str(error)})
        except Exception as error:
            self.respond(500, {'error': str(error)})
    def respond(self, status, payload):
        body = json.dumps(clean(payload), allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ['200', '304']:
            super().log_message(fmt, *args)

if __name__ == '__main__':
    print('ATSA API: http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
