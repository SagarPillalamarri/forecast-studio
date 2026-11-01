// Capacity is a user planning assumption; intervals are not probabilities.
export function calculateCapacity(data,capacity){
 if(!Number.isFinite(capacity)||capacity<=0)throw new Error('Capacity must be positive and finite');
 const electricity=data.dataset==='electricity_daily';
 const points=electricity?data.future.dailyRows:data.future.monthlyRows;
 const valueKey=electricity?'Forecast (GWh)':'Forecast attendances';
 const lowerKey=electricity?'Lower 95% (GWh)':'Lower 95%';
 const upperKey=electricity?'Upper 95% (GWh)':'Upper 95%';
 const rows=data.future.monthlyRows.map(month=>{
  const values=electricity?points.filter(p=>p.Date.startsWith(month.Month)):[month];
  const excess=values.reduce((sum,p)=>sum+Math.max(0,p[valueKey]-capacity),0);
  const count=values.filter(p=>p[valueKey]>capacity).length;
  const interval=values.some(p=>p[upperKey]>capacity);
  const above=values.some(p=>p[lowerKey]>capacity);
  const status=above?'Entire interval above':count?'Forecast above':interval?'Interval reaches above':'Within interval bounds';
  return electricity?{Month:month.Month,'Days above capacity':count,'Additional energy (GWh)':excess,'Peak daily forecast (GWh)':Math.max(...values.map(v=>v[valueKey])),Status:status}:{Month:month.Month,'Forecast attendances':month[valueKey],'Additional capacity needed':excess,'Utilization (%)':month[valueKey]/capacity*100,Status:status};
 });
 return {rows,months:rows.filter(r=>r.Status==='Forecast above'||r.Status==='Entire interval above').length,
  excess:rows.reduce((sum,r)=>sum+(r['Additional capacity needed']??r['Additional energy (GWh)']),0),
  worst:Math.max(0,...points.map(p=>p[valueKey]-capacity))};
}
