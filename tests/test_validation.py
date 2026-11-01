import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.validation import rolling_validation, fit_fold

class RollingValidationTests(unittest.TestCase):
    def test_expanding_windows_and_nonoverlap(self):
        series=pd.Series(np.arange(80)+100.,index=pd.date_range('2010-01-01',periods=80,freq='MS'))
        seen=[]
        def fit(model,train,test,frequency,seasonal_period,dataset):
            self.assertLess(train.index[-1],test.index[0])
            self.assertEqual(len(test),4)
            seen.append((len(train),test.index[0],test.index[-1]))
            return {'forecast':test-2}
        fit_fold.cache_clear()
        with patch('src.validation.load_dataset',return_value=pd.DataFrame({'value':series})),patch('src.validation.fit_and_forecast_model',side_effect=fit):
            result=rolling_validation('ae_monthly',['Naive'],3,4)
        self.assertEqual([x[0] for x in seen[:3]],[68,72,76])
        self.assertLess(seen[0][2],seen[1][1])
        self.assertEqual(result['summary'][0]['Pooled RMSE'],2)
        self.assertEqual(result['summary'][0]['RMSE spread (std)'],0)
        fit_fold.cache_clear()

    def test_failed_fold_cannot_win(self):
        series=pd.Series(np.ones(80)*100,index=pd.date_range('2010-01-01',periods=80,freq='MS'))
        def fit(dataset,end,horizon,model):
            if model=='Naive' and end==72:
                raise ValueError('deliberate test failure')
            return {'forecast':series.iloc[end:end+horizon]-(2 if model=='Seasonal Naive' else 0)}
        with patch('src.validation.load_dataset',return_value=pd.DataFrame({'value':series})),patch('src.validation.fit_fold',side_effect=fit):
            result=rolling_validation('ae_monthly',['Naive'],3,4)
        row=next(r for r in result['summary'] if r['Model']=='Naive')
        self.assertEqual(row['Completed folds'],2)
        self.assertEqual(row['Ranking eligible'],'No')
        self.assertIsNone(row['vs baseline (%)'])
        self.assertEqual(result['winner']['Model'],'Seasonal Naive')
        self.assertEqual(len(result['errors']),1)

    def test_invalid_settings(self):
        for models,folds,horizon in [([],3,12),(['bad'],3,12),(['Naive'],1,12),(['Naive'],3,25)]:
            with self.subTest(models=models,folds=folds,horizon=horizon),self.assertRaises(ValueError):
                rolling_validation('ae_monthly',models,folds,horizon)

    def test_insufficient_history(self):
        frame=pd.DataFrame({'value':np.ones(40)},index=pd.date_range('2020-01-01',periods=40,freq='MS'))
        with patch('src.validation.load_dataset',return_value=frame),self.assertRaises(ValueError):
            rolling_validation('ae_monthly',['Naive'],3,12)

    def test_real_baseline_both_datasets(self):
        for dataset,horizon in [('ae_monthly',6),('electricity_daily',7)]:
            result=rolling_validation(dataset,['Naive'],2,horizon)
            self.assertEqual(result['summary'][0]['Completed folds'],2)
            self.assertEqual(len(result['details']),4)
            self.assertTrue(np.isfinite(result['winner']['Pooled RMSE']))

if __name__=='__main__':unittest.main()
