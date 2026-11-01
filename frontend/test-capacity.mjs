import assert from 'node:assert/strict';
import {calculateCapacity} from './src/capacity.mjs';
const monthly={dataset:'ae_monthly',future:{monthlyRows:[
 {Month:'2027-01','Forecast attendances':120,'Lower 95%':110,'Upper 95%':130},
 {Month:'2027-02','Forecast attendances':80,'Lower 95%':70,'Upper 95%':110},
 {Month:'2027-03','Forecast attendances':100,'Lower 95%':90,'Upper 95%':100}
]}};
const summary=calculateCapacity(monthly,100);
assert.equal(summary.months,1);assert.equal(summary.excess,20);assert.equal(summary.worst,20);
assert.equal(summary.rows[0].Status,'Entire interval above');
assert.equal(summary.rows[1].Status,'Interval reaches above');
assert.equal(summary.rows[2].Status,'Within interval bounds');
assert.equal(calculateCapacity(monthly,200).months,0);
const daily={dataset:'electricity_daily',future:{monthlyRows:[{Month:'2028-02'}],dailyRows:[
 {Date:'2028-02-28','Forecast (GWh)':150,'Lower 95% (GWh)':140,'Upper 95% (GWh)':160},
 {Date:'2028-02-29','Forecast (GWh)':50,'Lower 95% (GWh)':40,'Upper 95% (GWh)':60}
]}};
const energy=calculateCapacity(daily,100);
assert.equal(energy.excess,50);assert.equal(energy.rows[0]['Days above capacity'],1);
assert.equal(energy.worst,50);assert.equal(energy.months,1);
assert.throws(()=>calculateCapacity(monthly,0));assert.throws(()=>calculateCapacity(monthly,NaN));
console.log('Capacity checks passed: thresholds, interval flags, no offsetting spare capacity, daily aggregation.');
