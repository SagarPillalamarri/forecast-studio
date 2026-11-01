import React, {useEffect, useRef, useState} from 'react';
import CapacityPlanner from './CapacityPlanner.jsx';
import {ArrowRight, Download, LoaderCircle, CalendarDays} from 'lucide-react';

const months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
const localYear = () => new Date().getFullYear();
const asTarget = (year,month) => `${year}-${String(month).padStart(2,'0')}`;
function downloadCSV(results, daily) {
 const records=results.flatMap(d=>(daily?d.future.dailyRows:d.future.monthlyRows).map(row=>({Dataset:d.name,Model:d.future.model,...row})));
 if(!records.length)return;
 const keys=[...new Set(records.flatMap(row=>Object.keys(row)))];
 const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';
 const text=[keys.map(quote).join(','),...records.map(row=>keys.map(k=>quote(row[k])).join(','))].join('\r\n');
 const url=URL.createObjectURL(new Blob(['\uFEFF'+text],{type:'text/csv;charset=utf-8;'}));
 const a=document.createElement('a');a.href=url;a.download=`future-${daily?'daily':'monthly'}-${results[0].future.target}.csv`;a.click();URL.revokeObjectURL(url);
}
export default function FutureForecast({catalog,dataset,compare,Plot,Table,number,date}) {
 const ids=compare?['ae_monthly','electricity_daily']:[dataset];
 const configs=ids.map(id=>catalog?.datasets.find(d=>d.id===id)).filter(Boolean);
 const ready=configs.length===ids.length&&configs.every(d=>d.firstForecastMonth&&d.lastForecastMonth);
 const minimum=ready?configs.reduce((m,c)=>c.firstForecastMonth>m?c.firstForecastMonth:m,'0000-01'):'';
 const maximum=ready?configs.reduce((m,c)=>c.lastForecastMonth<m?c.lastForecastMonth:m,'9999-12'):'';
 const [year,setYear]=useState(localYear()),[month,setMonth]=useState(12),[model,setModel]=useState('Holt-Winters (Additive)');
 const [results,setResults]=useState([]),[loading,setLoading]=useState(false),[error,setError]=useState('');
 const [daily,setDaily]=useState(false);
 const activeRequest=useRef(null);
 const target=asTarget(year,month);
 const years=ready?Array.from({length:Number(maximum.slice(0,4))-Number(minimum.slice(0,4))+1},(_,i)=>Number(minimum.slice(0,4))+i):[];
 useEffect(()=>{
  if(!ready)return;
  if(target<minimum){setYear(Number(minimum.slice(0,4)));setMonth(Number(minimum.slice(5)));}
  if(target>maximum){setYear(Number(maximum.slice(0,4)));setMonth(Number(maximum.slice(5)));}
 },[minimum,maximum,target,ready]);
 useEffect(()=>{
  activeRequest.current?.abort();setResults([]);setError('');setLoading(false);setDaily(false);
  return()=>activeRequest.current?.abort();
 },[dataset,compare,target,model]);
 const generate=async()=>{
  activeRequest.current?.abort();const controller=new AbortController();activeRequest.current=controller;
  setLoading(true);setError('');setResults([]);
  try {
   const data=await Promise.all(ids.map(async id=>{
    const response=await fetch('/api/analyze?'+new URLSearchParams({dataset:id,view:'future',target,futureModel:model}),{signal:controller.signal});
    const payload=await response.json();
    if(!response.ok)throw new Error(payload.error||'Future forecast failed');
    if(!payload.future)throw new Error('Restart the app with ./run.sh to enable future forecasting.');
    if(id==='electricity_daily'&&payload.unit!=='Daily Energy (GWh)')throw new Error('Restart the app to load the corrected GWh units.');
    return payload;
   }));
   if(!controller.signal.aborted)setResults(data);
  }catch(e){if(!controller.signal.aborted)setError(e.message);}
  finally{if(!controller.signal.aborted)setLoading(false);}
 };
 return <div className="future-workspace">
  <section className="future-controls" aria-label="Future forecast settings">
   <div className="future-controls-heading"><span className="future-calendar"><CalendarDays size={19}/></span><div><h2>Choose a date beyond the data</h2><p>Fit on the complete series, then forecast through your selected month.</p></div></div>
   <div className="future-fields"><label>Forecast year<select aria-label="Forecast year" value={year} onChange={e=>setYear(Number(e.target.value))} disabled={!ready}>{years.map(y=><option key={y} value={y}>{y}{y===localYear()?' · this year':''}</option>)}</select></label><label>Through month<select aria-label="Through month" value={month} onChange={e=>setMonth(Number(e.target.value))} disabled={!ready}>{months.map((label,i)=><option key={label} value={i+1} disabled={asTarget(year,i+1)<minimum||asTarget(year,i+1)>maximum}>{label}</option>)}</select></label><label className="future-model">Forecast model<select aria-label="Future forecast model" value={model} onChange={e=>setModel(e.target.value)} disabled={!catalog}>{catalog?.models.map(name=><option key={name}>{name}</option>)}</select></label><button className="button primary" onClick={generate} disabled={!ready||loading||target<minimum||target>maximum}>{loading?<LoaderCircle className="spin" size={15}/>:<ArrowRight size={15}/>}Generate forecast</button></div>
   <div className="forecast-origins">{configs.map(c=><span key={c.id}><b>{c.name}:</b> observations end {date(c.lastObservation)}.</span>)}</div>
   <p className="field-help">The horizon starts after the last observation, including any gap before the selected year. Up to five years beyond each dataset. The holdout window does not limit future forecasts.</p>
   {!ready&&<p className="validation">{catalog?'Restart the app and refresh to load the future-forecast controls.':'Loading dataset dates…'}</p>}
  </section>
  {error&&<div className="error" role="alert"><strong>Could not generate the forecast.</strong><p>{error}</p></div>}
  {loading&&<div className="loading-panel" role="status"><LoaderCircle className="spin" size={24}/><h3>Forecasting through {months[month-1]} {year}</h3><p>Training {model} on all observed data. No future actual values are used.</p></div>}
  {!loading&&!results.length&&!error&&<div className="future-empty"><CalendarDays size={25}/><h3>Plan the year ahead. Or the next one.</h3><p>Choose 2027, 2028, or another available year. Generate a forecast to explore its monthly predictions and uncertainty.</p></div>}
  {!!results.length&&<>
   <div className="future-results-header"><div><span className="overline">PREDICTIONS, NOT OBSERVATIONS</span><h2>{year} outlook · through {months[month-1]}</h2></div><button className="button secondary" onClick={()=>downloadCSV(results,false)}><Download size={14}/>Monthly CSV</button></div>
   <div className={'analysis-grid '+(compare?'compare-grid':'')}>{results.map(d=>{
    const f=d.future;const last=f.monthlyRows.at(-1);const electricity=d.dataset==='electricity_daily';
    return <section className="analysis-panel" key={d.dataset}>
     <div className="panel-heading"><div><h2>{d.name}</h2><p>{f.model} <span>·</span> {f.specification}</p></div><span className="period-chip">Through {date(f.forecastEnd)}</span></div>
     <div className="future-summary"><div><span>Forecast for {months[month-1]} {year}</span><strong>{number(electricity?last['Total energy (GWh)']:last['Forecast attendances'],electricity?1:0)}<small>{electricity?'GWh':'attendances'}</small></strong></div><div><span>Full forecast horizon</span><strong>{number(f.steps)}<small>{electricity?'days':'months'}</small></strong></div><div><span>Highest predicted month in shown period</span><strong>{date(f.peakMonth+'-01')}</strong></div></div>
     <CapacityPlanner data={d} Plot={Plot} Table={Table} number={number}/>
     <div className="future-table-heading"><div><h3>{electricity&&daily?'Daily predictions':'Month-by-month predictions'} · {year}</h3><p>{electricity?'Monthly energy totals sum daily point forecasts.':'Monthly attendance forecasts with 95% model intervals.'}</p></div>{electricity&&<label className="compare-toggle"><input type="checkbox" checked={daily} onChange={e=>setDaily(e.target.checked)}/><span className="switch"/>Daily details</label>}</div>
     <Table rows={electricity&&daily?f.dailyRows:f.monthlyRows} unit={d.unit}/>
     {electricity&&daily&&<button className="button text-button daily-download" onClick={()=>downloadCSV([d],true)}><Download size={13}/>Download daily CSV</button>}
     <div className="future-notes"><p>{f.note}</p><p>{f.accuracyNote}</p><p>Longer horizons are more uncertain; interval methods vary by model and some are approximate.{electricity&&' Electricity models use a 7-day seasonal cycle; not all capture annual seasonality.'}</p>{f.warnings.map(note=><p className="validation" key={note}>{note}</p>)}</div>
     <div className="chart-footnote"><span className="source-dot"/>Trained on {number(f.trainingObservations)} observations through {date(f.origin)}<span className="footnote-source">Source: {electricity?'NESO':'NHS England'}</span></div>
    </section>;
   })}</div>
  </>}
 </div>;
}
