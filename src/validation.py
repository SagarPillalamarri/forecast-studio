"""Expanding-window backtests using chronological, non-overlapping test blocks."""
from functools import lru_cache
import numpy as np
import plotly.graph_objects as go
from src.data import DATASET_CONFIGS, load_dataset
from src.models import AVAILABLE_MODELS, fit_and_forecast_model

@lru_cache(maxsize=192)
def fit_fold(dataset, train_end, horizon, model):
    cfg=DATASET_CONFIGS[dataset]
    series=load_dataset(dataset)['value']
    return fit_and_forecast_model(model,series.iloc[:train_end],series.iloc[train_end:train_end+horizon],
                                   cfg['frequency'],cfg['seasonal_period'],dataset)

def rolling_validation(dataset, models, folds=3, horizon=12):
    cfg=DATASET_CONFIGS[dataset]
    series=load_dataset(dataset)['value']
    if not isinstance(models,list) or not models or len(models)>5 or any(m not in AVAILABLE_MODELS for m in models):
        raise ValueError('Choose between one and five valid models for validation')
    if not 2<=folds<=6 or not 1<=horizon<=(24 if cfg['frequency']=='MS' else 90):
        raise ValueError('Use 2–6 folds and a horizon of 1–24 months or 1–90 days')
    first_end=len(series)-folds*horizon
    if first_end<max(36,3*cfg['seasonal_period']):
        raise ValueError('Not enough training history. Reduce the number of folds or forecast horizon.')
    names=list(dict.fromkeys(['Seasonal Naive',*models]))
    details=[]; errors=[]; summaries=[]; intervals=[]
    for fold in range(folds):
        end=first_end+fold*horizon
        test=series.iloc[end:end+horizon]
        intervals.append({'Fold':fold+1,'Training through':series.index[end-1].strftime('%Y-%m-%d'),
                          'Test start':test.index[0].strftime('%Y-%m-%d'),'Test end':test.index[-1].strftime('%Y-%m-%d')})
    for name in names:
        diffs=[]; pct_errors=[]; fold_errors=[]
        for fold in range(folds):
            end=first_end+fold*horizon
            actual=series.iloc[end:end+horizon].to_numpy(dtype=float)
            try:
                fit=fit_fold(dataset,end,horizon,name)
                pred=fit['forecast'].to_numpy(dtype=float)
                if len(pred)!=horizon or not np.isfinite(pred).all():
                    raise ValueError('Non-finite or incomplete predictions')
                difference=actual-pred
                rmse=float(np.sqrt(np.mean(difference**2)))
                diffs.extend(difference.tolist())
                pct_errors.extend((np.abs(difference)/np.maximum(np.abs(actual),1e-6)*100).tolist())
                fold_errors.append(rmse)
                details.append({'Model':name,**intervals[fold],'RMSE':rmse,'MAE':float(np.mean(np.abs(difference)))})
            except Exception as error:
                errors.append({'model':name,'fold':fold+1,'message':str(error)})
        summaries.append({'Model':name,'Pooled RMSE':float(np.sqrt(np.mean(np.square(diffs)))) if diffs else None,
                          'Pooled MAE':float(np.mean(np.abs(diffs))) if diffs else None,
                          'MAPE (%)':float(np.mean(pct_errors)) if pct_errors else None,
                          'RMSE spread (std)':float(np.std(fold_errors)) if fold_errors else None,
                          'Worst fold RMSE':max(fold_errors) if fold_errors else None,
                          'Completed folds':len(fold_errors),'Ranking eligible':'Yes' if len(fold_errors)==folds else 'No'})
    baseline=next(row for row in summaries if row['Model']=='Seasonal Naive')
    for row in summaries:
        row['vs baseline (%)']=(1-row['Pooled RMSE']/baseline['Pooled RMSE'])*100 if row['Ranking eligible']=='Yes' and baseline['Ranking eligible']=='Yes' and baseline['Pooled RMSE'] else None
    eligible=[r for r in summaries if r['Ranking eligible']=='Yes']
    winner=min(eligible,key=lambda r:r['Pooled RMSE']) if eligible else None
    fig=go.Figure()
    colors=['#8697b3','#3662e3','#309989','#bd8655','#8062ad','#de7881']
    for i,name in enumerate(names):
        rows=[r for r in details if r['Model']==name]
        fig.add_trace(go.Scatter(x=list(range(1,folds+1)),y=[next((r['RMSE'] for r in rows if r['Fold']==i),None) for i in range(1,folds+1)],mode='lines+markers',name=name,
                                line=dict(color=colors[i%len(colors)],width=2),connectgaps=False))
    fig.update_layout(xaxis=dict(title='Backtest fold',dtick=1),yaxis=dict(title=f"RMSE ({cfg['unit']})"),hovermode='x unified')
    return {'summary':summaries,'details':details,'folds':intervals,'horizon':horizon,'foldCount':folds,
            'firstTrainingSize':first_end,'winner':winner,'errors':errors,'figure':fig,
            'note':'Each fold refits on an expanding training window and predicts the next block without using its actual values. Test blocks do not overlap. Rankings require all folds to succeed and use pooled RMSE; incomplete models are excluded. Seasonal Naive is always included as a baseline.'}
