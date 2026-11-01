"""Regression checks for date handling, aggregation, and future/holdout separation."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from api import analyze, clean
from src.future import future_index, build_future, fit_future
from src.data import load_dataset
from src.models import fit_and_forecast_model
import json

class FutureForecastTests(unittest.TestCase):
    def test_monthly_origin_and_target(self):
        dates = future_index('ae_monthly', '2028-02')
        self.assertEqual(str(dates[0].date()), '2026-05-01')
        self.assertEqual(str(dates[-1].date()), '2028-02-01')
        self.assertEqual(len(dates), 22)

    def test_daily_calendar_includes_leap_day_and_gap(self):
        dates = future_index('electricity_daily', '2028-02')
        self.assertEqual(str(dates[0].date()), '2025-01-01')
        self.assertEqual(str(dates[-1].date()), '2028-02-29')
        self.assertEqual(len(dates), (pd.Timestamp('2028-02-29')-pd.Timestamp('2024-12-31')).days)

    def test_target_validation(self):
        for dataset,target in [('ae_monthly','2026-04'),('ae_monthly','2031-05'),('electricity_daily','2030-01'),('electricity_daily','2027-13'),('electricity_daily','2027-2'),('ae_monthly','bad')]:
            with self.subTest(dataset=dataset,target=target), self.assertRaises(ValueError):
                future_index(dataset,target)

    def test_future_has_no_accuracy_metrics_or_test_actuals(self):
        series=load_dataset('ae_monthly')['value']
        dates=future_index('ae_monthly','2027-01')
        fit=fit_and_forecast_model('Seasonal Naive',series,None,'MS',12,'ae_monthly',future_index=dates)
        self.assertIsNone(fit['metrics'])
        np.testing.assert_array_equal(fit['forecast'].to_numpy(),series.iloc[-12:].to_numpy()[:len(dates)])
        payload=analyze({'dataset':['ae_monthly'],'view':['future'],'target':['2027-01'],'futureModel':['Seasonal Naive']})
        self.assertNotIn('winner',payload)
        self.assertEqual(payload['rows'],[])
        self.assertEqual(payload['future']['monthlyRows'][0]['Month'],'2027-01')
        self.assertEqual(len(payload['future']['monthlyRows']),1)
        json.dumps(clean(payload),allow_nan=False)

    def test_daily_totals_and_intervals(self):
        payload=build_future('electricity_daily','2028-02','Seasonal Naive')
        self.assertEqual([r['Days'] for r in payload['monthlyRows']],[31,29])
        days=pd.DataFrame(payload['dailyRows'])
        feb=days[days['Date'].str.startswith('2028-02')]
        self.assertAlmostEqual(payload['monthlyRows'][1]['Total energy (GWh)'],feb['Forecast (GWh)'].sum())
        self.assertNotIn('Lower 95%',payload['monthlyRows'][1])
        self.assertTrue((days['Lower 95% (GWh)']<=days['Forecast (GWh)']).all())
        self.assertTrue((days['Upper 95% (GWh)']>=days['Forecast (GWh)']).all())
        self.assertLess(days['Forecast (GWh)'].max(),2000)

    def test_holdout_still_has_accuracy(self):
        payload=analyze({'dataset':['ae_monthly'],'view':['metrics'],'models':['["Seasonal Naive"]']})
        self.assertAlmostEqual(payload['rows'][0]['RMSE'],64254.33,places=2)

    def test_common_models_forecast_both_frequencies(self):
        for dataset in ['ae_monthly','electricity_daily']:
            for model in ['Holt-Winters (Additive)','ARIMA(p,d,q)']:
                with self.subTest(dataset=dataset,model=model):
                    fit=fit_future(dataset,'2027-03',model)
                    self.assertIsNone(fit['metrics'])
                    self.assertTrue(np.isfinite(fit['forecast']).all())
                    self.assertEqual(fit['forecast'].index[0],future_index(dataset,'2027-03')[0])

if __name__=='__main__':
    unittest.main()
