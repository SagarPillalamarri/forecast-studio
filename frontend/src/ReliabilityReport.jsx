import React,{useEffect,useRef,useState} from 'react';
import {ArrowRight,LoaderCircle} from 'lucide-react';
export default function ReliabilityReport({catalog,dataset,compare,models,Plot,Table,number}){
 const [folds,setFolds]=useState(3),[horizons,setHorizons]=useState({ae_monthly:12,electricity_daily:30});
 const [chosen,setChosen]=useState(models.slice(0,5));
 const [results,setResults]=useState([]),[loading,setLoading]=useState(false),[error,setError]=useState('');
 const request=useRef(null);const ids=compare?['ae_monthly','electricity_daily']:[dataset];
 useEffect(()=>{request.current?.abort();setResults([]);setLoading(false);setError('');return()=>request.current?.abort();},[dataset,compare,folds,horizons,chosen]);
 const run=async()=>{
  request.current?.abort();const controller=new AbortController();request.current=controller;setLoading(true);setError('');setResults([]);
  try{
   const data=await Promise.all(ids.map(async id=>{
    const res=await fetch('/api/analyze?'+new URLSearchParams({dataset:id,view:'validation',models:JSON.stringify(chosen),folds,validationHorizon:horizons[id]}),{signal:controller.signal});const p=await res.json();
    if(!res.ok)throw new Error(p.error||'Validation failed');if(!p.validation)throw new Error('Restart ./run.sh to load rolling validation.');return p;
   }));if(!controller.signal.aborted)setResults(data);
  }catch(e){if(!controller.signal.aborted)setError(e.message);}finally{if(!controller.signal.aborted)setLoading(false);}
 };
 return <section className="reliability-workspace">
  <div className="future-controls"><div className="future-controls-heading"><div><h2>Test across several historical periods</h2><p>Expanding training windows. Equal, non-overlapping test blocks. Every model is refitted at every cutoff.</p></div></div>
   <div className="future-fields"><label>Backtest folds<select aria-label="Backtest folds" value={folds} onChange={e=>setFolds(Number(e.target.value))}>{[2,3,4,5,6].map(n=><option key={n}>{n}</option>)}</select></label>{ids.map(id=><label key={id}>{id==='ae_monthly'?'Horizon per fold (months)':'Horizon per fold (days)'}<input aria-label={id==='ae_monthly'?'NHS validation horizon':'Electricity validation horizon'} type="number" min="1" max={id==='ae_monthly'?24:90} value={horizons[id]} onChange={e=>setHorizons({...horizons,[id]:e.target.value===''?'':Number(e.target.value)})}/></label>)}<button className="button primary" disabled={loading||!chosen.length||ids.some(id=>!Number.isInteger(horizons[id])||horizons[id]<1||horizons[id]>(id==='ae_monthly'?24:90))} onClick={run}>{loading?<LoaderCircle className="spin" size={15}/>:<ArrowRight size={15}/>}Run backtest</button></div>
   <details className="validation-models"><summary>Validation models · {chosen.length} selected</summary><p className="field-help">Choose up to five models. Seasonal Naive is always included as the reference. Complex models can take longer.</p><div className="model-options">{catalog?.models.map(name=><label key={name}><input type="checkbox" checked={chosen.includes(name)} disabled={!chosen.includes(name)&&chosen.length>=5} onChange={e=>setChosen(e.target.checked?[...chosen,name]:chosen.filter(m=>m!==name))}/>{name}</label>)}</div></details>
  </div>
  {error&&<div className="error" role="alert">{error}</div>}
  {loading&&<div className="loading-panel" role="status"><LoaderCircle className="spin" size={24}/><h3>Testing {folds} historical forecast origins</h3><p>Every fold fits only on earlier observations. Results are cached for repeated settings.</p></div>}
  {!loading&&!results.length&&!error&&<div className="future-empty"><h3>Does the model hold up over time?</h3><p>Run a backtest to compare accuracy, variability, and worst-period performance. Results are historical evidence, not guarantees of future accuracy.</p></div>}
  <div className={'analysis-grid '+(compare?'compare-grid':'')}>{results.map(d=>{const v=d.validation;return <section className="analysis-panel" key={d.dataset}>
   <div className="panel-heading"><div><h2>{d.name} · Forecast reliability</h2><p>{folds} folds · {v.horizon} {d.frequency==='Monthly'?'months':'days'} per fold · {v.firstTrainingSize} initial training observations</p></div></div>
   <div className="validation-verdict"><span className="overline">LOWEST POOLED RMSE · COMPLETE MODELS ONLY</span><h3>{v.winner?.Model||'No model completed every fold'}</h3>{v.winner&&<p>Pooled RMSE: <b>{number(v.winner['Pooled RMSE'],2)} {d.dataset==='electricity_daily'?'GWh':'attendances'}</b>. {v.winner['vs baseline (%)']!=null&&`${number(v.winner['vs baseline (%)'],1)}% lower RMSE than Seasonal Naive (negative means worse).`}</p>}</div>
   <Plot figure={d.figures[0]} view="validation"/>
   <div className="future-table-heading"><div><h3>Accuracy & consistency</h3><p>RMSE spread is the standard deviation of fold RMSEs. Lower spread means less variation across these periods.</p></div></div>
   <Table rows={v.summary} winner={v.winner?.Model} unit={d.unit}/>
   <details className="fold-details"><summary>Show test dates and per-fold results</summary><Table rows={v.folds}/><Table rows={v.details} unit={d.unit}/></details>
   {v.errors.length>0&&<details className="fit-errors"><summary>{v.errors.length} fold fits failed</summary>{v.errors.map((e,i)=><p key={i}>{e.model} · fold {e.fold}: {e.message}</p>)}</details>}
   <div className="future-notes"><p>{v.note}</p><p>This ranking measures historical accuracy across folds. It does not apply the single-holdout verdict’s Ljung–Box eligibility rule. Model choices tuned using these results need a separate untouched final test for an unbiased evaluation.</p></div>
  </section>;})}</div>
 </section>;
}
