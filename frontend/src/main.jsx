import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity, ArrowUpRight, ArrowDownRight, ArrowRight, ChartNoAxesCombined, Layers3, SlidersHorizontal, ChevronDown, Check, Download, X, Menu, RotateCcw, CircleHelp, Settings2, FlaskConical, LoaderCircle} from 'lucide-react';
import './style.css';
import FutureForecast from './FutureForecast.jsx';
import ReliabilityReport from './ReliabilityReport.jsx';

const groups = [
  {name:'Explore', icon:ChartNoAxesCombined, views:[['history','Historical series'],['decomposition','Decomposition'],['seasonal','Seasonal patterns'],['acf','Autocorrelation'],['pacf','Partial autocorrelation'],['stationarity','Stationarity']]},
  {name:'Forecast', icon:Layers3, views:[['forecast','Holdout comparison'],['future','Future forecast']]},
  {name:'Evaluate', icon:FlaskConical, views:[['metrics','Model results'],['residuals','Residual diagnostics'],['verdict','Final verdict'],['validation','Forecast reliability']]}
];
const explanations = {
 history:['A closer look at the series','The blue line shows recorded observations. The green moving average smooths short-term variation to make the underlying trend easier to see. Adjust its window in Analysis settings.'],
 decomposition:['Separate the signal','Decomposition separates observed values into trend, seasonality, and residuals. Compare additive and multiplicative methods to examine whether seasonal variation scales with the series.'],
 seasonal:['Find the recurring pattern','Seasonal indices show percentage departures from the average seasonal level. Monthly data is grouped by month; daily data by weekday.'],
 acf:['How long does the series remember?','Autocorrelation measures relationships across lags. Compare the original, first-differenced, and seasonally differenced series; the bands mark approximate 95% confidence limits.'],
 pacf:['Identify direct lag relationships','Partial autocorrelation removes the effect of intervening lags. Significant spikes can help identify candidate autoregressive orders.'],
 stationarity:['Check the modeling assumptions','The ADF test evaluates a unit-root null hypothesis. A p-value below 0.05 provides evidence against a unit root; it does not establish every form of stationarity.'],
 forecast:['Read the holdout forecast','Models are fitted on training data and evaluated against held-out observations. Click legend entries to isolate a forecast. Shaded bands show model-specific 95% prediction intervals where available.'],
 metrics:['Accuracy with a residual check','Selection prioritizes the lowest test RMSE among models with a Ljung–Box p-value above 0.05. If none pass, the lowest-RMSE model is selected with a caveat.'],
 residuals:['Look for what the model missed','Review the residual time plot, distribution, autocorrelation, and Q–Q plot. A Ljung–Box p-value above 0.05 means the test did not detect autocorrelation at the tested lags.'],
 verdict:['The decision rule','The selected model has the lowest holdout RMSE among models passing the residual check. This recommendation is specific to the selected dataset, models, and holdout.']
};
const number = (n, digits=0) => n == null ? '—' : new Intl.NumberFormat('en-GB',{maximumFractionDigits:digits}).format(n);
const short = n => n == null ? '—' : new Intl.NumberFormat('en-GB',{notation:'compact',maximumFractionDigits:2}).format(n);
const date = s => new Date(s+'T00:00:00').toLocaleDateString('en-GB',{month:'short',year:'numeric'});
let plotlyPromise;
const loadPlotly = () => plotlyPromise ||= import('plotly.js-dist-min').then(m => m.default);

function Plot({figure, view}) {
 const el = useRef(null);
 const [error,setError] = useState('');
 useEffect(() => {
  let alive=true, observer;
  loadPlotly().then(async Plotly => {
   if (!alive || !el.current) return;
   const colors=['#3662e3','#309989','#bd8655','#8062ad','#de7881','#7792b8','#55857d','#bf9c3c'];
   const forecastNames = figure.data.filter(t=>t.name && !t.name.includes('(95% CI)') && !['Historical Train','Test Actuals'].includes(t.name)).map(t=>t.name);
   const traces = figure.data.map(t => {
    if(view!=='forecast') return t;
    if(t.name==='Historical Train') return {...t,line:{...t.line,color:'#a7b1c1'}};
    if(t.name==='Test Actuals') return {...t,line:{...t.line,color:'#23344d'},marker:{...t.marker,color:'#23344d'}};
    const name=t.name?.replace(' (95% CI)','');
    const color=colors[Math.max(0,forecastNames.indexOf(name))%colors.length];
    return {...t,legendgroup:name,...(t.fill==='toself'?{fillcolor:color}:{line:{...t.line,color},marker:{...t.marker,color}})};
   });
   const complex=['decomposition','acf','pacf','residuals'].includes(view);
   const layout={...figure.layout, title:{text:''}, autosize:true, height:complex ? (view==='decomposition'?700:600) : 390, font:{family:'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',color:'#7b8494',size:11}, paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{l:65,r:25,t:complex?45:25,b:view==='forecast'?100:65},legend:{orientation:'h',x:0,y:-.17,xanchor:'left',yanchor:'top',font:{size:11,color:'#606b7c'},bgcolor:'transparent',borderwidth:0,groupclick:'togglegroup'},hoverlabel:{bgcolor:'#172337',font:{color:'#fff',size:12}},...(!complex?{xaxis:{...figure.layout.xaxis,gridcolor:'#f3f4f7',showline:false,tickfont:{size:10},title:{text:''}},yaxis:{...figure.layout.yaxis,gridcolor:'#eef0f4',zeroline:false,title:{...figure.layout.yaxis?.title,font:{size:10}},tickfont:{size:10}}}:{})};
   await Plotly.react(el.current,traces,layout,{responsive:true,displaylogo:false,displayModeBar:'hover',modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{filename:'forecast-studio',format:'png',scale:2}});
   if (!alive) return;
   observer=new ResizeObserver(()=>{if(el.current) Plotly.Plots.resize(el.current);});observer.observe(el.current);
  }).catch(e => {if(alive)setError('Chart could not load. '+e.message);});
  return()=>{alive=false;observer?.disconnect();if(el.current)loadPlotly().then(p=>p.purge(el.current));};
 },[figure,view]);
 return error?<div className="error">{error}</div>:<div className="plot" ref={el}/>;
}
function Table({rows,winner,unit}) {
 if(!rows.length)return null;
 const columns=Object.keys(rows[0]);
 return <div className="table-scroll"><table><thead><tr>{columns.map(c=><th key={c}>{c}{['RMSE','MAE','Pooled RMSE','Pooled MAE','RMSE spread (std)','Worst fold RMSE'].includes(c)&&unit==='Daily Energy (GWh)'?' (GWh)':''}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={i} className={Boolean(winner)&&row.Model===winner?'best-row':''}>{columns.map((c,j)=><td key={c}>{c==='Status'?<span className={'tag '+(row[c]==='Forecast above'||row[c]==='Entire interval above'?'capacity-danger':row[c]==='Interval reaches above'?'capacity-watch':'green')}>{row[c]}</span>:c==='Residual check'?<span className={'tag '+(row[c]==='Pass'?'green':'neutral')}>{row[c]==='Pass'&&<Check size={11}/>} {row[c]}</span>:typeof row[c]==='number'?number(row[c],unit==='Attendances'&&['Forecast attendances','Lower 95%','Upper 95%'].includes(c)?0:c==='Ljung–Box p'||c==='p-value'||c==='Test Statistic'?4:2):row[c]??'—'}{j===0 && Boolean(winner)&&row.Model===winner && <span className="best-label">Selected</span>}</td>)}</tr>)}</tbody></table></div>;
}
function App(){
 const [catalog,setCatalog]=useState(null),[catalogError,setCatalogError]=useState('');
 const [dataset,setDataset]=useState('ae_monthly'),[compare,setCompare]=useState(false),[view,setView]=useState('history');
 const [windowSize,setWindowSize]=useState(7),[holdouts,setHoldouts]=useState({ae_monthly:12,electricity_daily:60});
 const [models,setModels]=useState(['Seasonal Naive','Holt-Winters (Additive)','ARIMA(p,d,q)']);
 const [method,setMethod]=useState('stl'),[diagnostic,setDiagnostic]=useState('');
 const [settings,setSettings]=useState(false),[mobileNav,setMobileNav]=useState(false);
 const [data,setData]=useState([]),[loading,setLoading]=useState(true),[error,setError]=useState(''),[retry,setRetry]=useState(0),[help,setHelp]=useState(false);
 const [draft,setDraft]=useState(null);
 const group=groups.find(g=>g.views.some(v=>v[0]===view));
 const title=group.views.find(v=>v[0]===view)[1];
 useEffect(()=>{let alive=true;fetch('/api/catalog').then(async r=>{const p=await r.json();if(!r.ok)throw new Error(p.error||'Unable to load datasets');return p;}).then(p=>{if(alive){setCatalog(p);setCatalogError('');}}).catch(e=>{if(alive)setCatalogError(e.message);});return()=>{alive=false;};},[retry]);
 useEffect(()=>{
  if(['future','validation'].includes(view)){setData([]);setLoading(false);setError('');return;}
  const controller=new AbortController();setLoading(true);setError('');setData([]);
  const ids=compare?['ae_monthly','electricity_daily']:[dataset];
  Promise.all(ids.map(id=>fetch('/api/analyze?'+new URLSearchParams({dataset:id,view,window:windowSize,holdout:holdouts[id],models:JSON.stringify(models),method,diagnostic}),{signal:controller.signal}).then(async r=>{const p=await r.json();if(!r.ok)throw new Error(p.error||'Analysis failed');if(p.dataset==='electricity_daily'&&p.unit!=='Daily Energy (GWh)')throw new Error('The Python service is outdated. Stop the running app with Ctrl+C, then run ./run.sh again to load the GWh conversion.');return p;}))).then(p=>{if(!controller.signal.aborted)setData(p);}).catch(e=>{if(e.name!=='AbortError'&&!controller.signal.aborted)setError(e.message);}).finally(()=>{if(!controller.signal.aborted)setLoading(false);});
  return()=>controller.abort();
 },[dataset,compare,view,windowSize,holdouts,models,method,diagnostic,retry]);
 const openSettings=()=>{setDraft({window:windowSize,holdouts:{...holdouts},models:[...models]});setSettings(true);};
 const apply=()=>{setWindowSize(Number(draft.window));setHoldouts(draft.holdouts);setModels(draft.models);setSettings(false);};
 useEffect(()=>{const onKey=e=>{if(e.key==='Escape'){setSettings(false);setMobileNav(false);setHelp(false);}};document.addEventListener('keydown',onKey);return()=>document.removeEventListener('keydown',onKey);},[]);
 const exportData=()=>{const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`atsa-${view}.json`;a.click();URL.revokeObjectURL(url);};
 const go=v=>{setView(v);setDiagnostic('');setMobileNav(false);};
 useEffect(()=>{
  if(!settings&&!help&&!mobileNav)return;
  const previous=document.activeElement;
  const oldOverflow=document.body.style.overflow;
  document.body.style.overflow='hidden';
  const trap=e=>{
   if(e.key!=='Tab')return;
   const target=document.querySelector('[role=dialog]') || document.querySelector('.sidebar.is-open');
   if(!target)return;
   const items=[...target.querySelectorAll('button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled)')];
   if(!items.length)return;
   if(e.shiftKey&&document.activeElement===items[0]){e.preventDefault();items.at(-1).focus();}
   else if(!e.shiftKey&&document.activeElement===items.at(-1)){e.preventDefault();items[0].focus();}
  };
  document.addEventListener('keydown',trap);
  return()=>{document.body.style.overflow=oldOverflow;document.removeEventListener('keydown',trap);previous?.focus();};
 },[settings,help,mobileNav]);
 const current=data[0];
 return <div className="app-shell">
  {mobileNav&&<button className="nav-scrim" aria-label="Close navigation" onClick={()=>setMobileNav(false)}/>}
  <aside className={'sidebar '+(mobileNav?'is-open':'')}>
   <a className="brand" href="#" onClick={e=>{e.preventDefault();go('history');}}><span className="brand-symbol"><Activity size={22} strokeWidth={2}/></span><span>forecast<span className="brand-light"> / studio</span></span></a>
   <div className="project-label"><span className="project-icon">A</span><div><strong>ATSA Project</strong><small>Semester 7 · Research workspace</small></div></div>
   <div className="nav-caption">WORKSPACE</div>
   <nav aria-label="Analysis navigation">{groups.map(g=><div className="nav-group" key={g.name}><button className={'nav-parent '+(group.name===g.name?'active':'')} onClick={()=>go(g.views[0][0])}><g.icon size={18}/><span>{g.name}</span><ChevronDown size={13} className={group.name===g.name?'':'collapsed'}/></button>{group.name===g.name&&g.views.length>1&&<div className="subnav">{g.views.map(([id,label])=><button key={id} className={view===id?'selected':''} onClick={()=>go(id)}><span className="nav-dot"/>{label}</button>)}</div>}</div>)}</nav>
   <div className="sidebar-bottom"><button className="side-settings" onClick={openSettings}><Settings2 size={17}/>Analysis settings<ArrowUpRight size={14}/></button><div className="source-note"><span className="source-dot"/> Official source datasets<small>NHS England & NESO</small></div><div className="sidebar-footer"><span>Applied Time Series Analysis</span><button aria-label="About this workspace" onClick={()=>setHelp(true)}><CircleHelp size={16}/></button></div></div>
  </aside>
  <main>
   <header className="topbar"><div className="breadcrumb"><button className="mobile-menu icon-button" aria-label="Open navigation" onClick={()=>setMobileNav(true)}><Menu size={20}/></button><span>Workspace</span><span className="crumb-slash">/</span><strong>{group.name}</strong></div><div className="topbar-right"><span className="project-chip">ATSA / 07</span><button className="icon-button" aria-label="About workspace" onClick={()=>setHelp(true)}><CircleHelp size={18}/></button></div></header>
   <div className="main-content">
    <div className="page-heading"><div><div className="overline">{group.name==='Explore'?'UNDERSTAND THE DATA':group.name==='Forecast'?'TEST THE POSSIBILITIES':'ASSESS THE EVIDENCE'}</div><h1>{title}<span className="heading-dot">.</span></h1><p>{group.name==='Explore'?'Every forecast starts with a closer look.':group.name==='Forecast'?(view==='future'?'Explore what comes after the last observation.':'See how each model follows the unseen data.'):'Find the model that earns its place.'}</p></div>{!['future','validation'].includes(view)&&<button className="button secondary export-button" disabled={!data.length||loading} onClick={exportData}><Download size={15}/>Export analysis</button>}</div>
    <section className="dataset-toolbar" aria-label="Dataset controls"><div className="dataset-tabs"><button className={dataset==='ae_monthly'?'chosen':''} onClick={()=>{setDataset('ae_monthly');setDiagnostic('');}}><span className="dataset-dot blue"/>NHS A&E<span className="frequency">Monthly</span></button><button className={dataset==='electricity_daily'?'chosen':''} onClick={()=>{setDataset('electricity_daily');setDiagnostic('');}}><span className="dataset-dot teal"/>UK Electricity<span className="frequency">Daily</span></button></div><div className="toolbar-actions"><label className="compare-toggle"><input type="checkbox" checked={compare} onChange={e=>setCompare(e.target.checked)}/><span className="switch"/>Compare</label><span className="toolbar-divider"/><button className="button text-button" onClick={openSettings}><SlidersHorizontal size={15}/>Settings</button></div></section>
    {catalogError&&<div className="error" role="alert">{catalogError}<button onClick={()=>setRetry(n=>n+1)}>Retry</button></div>}
    {view==='future'?<FutureForecast catalog={catalog} dataset={dataset} compare={compare} Plot={Plot} Table={Table} number={number} date={date}/>:view==='validation'?<ReliabilityReport catalog={catalog} dataset={dataset} compare={compare} models={models} Plot={Plot} Table={Table} number={number}/>:error?<div className="error error-panel" role="alert"><strong>We couldn’t complete this analysis.</strong><p>{error}</p><button className="button secondary" onClick={()=>setRetry(n=>n+1)}><RotateCcw size={14}/>Try again</button><button className="button text-button" onClick={openSettings}>Adjust settings</button></div>:loading?<div className="loading-panel" role="status"><LoaderCircle className="spin" size={24}/><h3>{['forecast','metrics','residuals','verdict'].includes(view)?'Fitting your selected models':'Preparing the series'}</h3><p>{['forecast','metrics','residuals','verdict'].includes(view)?'Some models take a little longer. Fitted results are cached for your next view.':'Reading the dataset and calculating the analysis.'}</p><div className="loading-line"/></div>:<>
     <div className={'metric-strip '+(compare?'comparison-metrics':'')}>
      {compare?data.map(d=><div className="metric" key={d.dataset}><span className="metric-title">{d.name}</span><strong>{number(d.observations)}<span className="metric-inline">observations</span></strong><small>{date(d.start)} — {date(d.end)}</small></div>):<><div className="metric"><span className="metric-title">Observations</span><strong>{number(current?.observations)}</strong><small>{current&&date(current.start)} — {current&&date(current.end)}</small></div><div className="metric"><span className="metric-title">Latest observation</span><strong>{current?.dataset==='electricity_daily'?<>{number(current.latest,1)}<span className="metric-inline">GWh</span></>:short(current?.latest)}</strong><small><span className="change">{current?.latest>=current?.previous?<ArrowUpRight size={12}/>:<ArrowDownRight size={12}/>} {number(Math.abs((current?.latest/current?.previous-1)*100),1)}%</span> from previous period</small></div><div className="metric"><span className="metric-title">Average per {current?.frequency==='Monthly'?'month':'day'}</span><strong>{current?.dataset==='electricity_daily'?<>{number(current.mean,1)}<span className="metric-inline">GWh</span></>:short(current?.mean)}</strong><small>{current?.unit}</small></div><div className="metric"><span className="metric-title">Holdout window</span><strong>{current?.holdout}<span className="metric-inline">{current?.frequency==='Monthly'?'months':'days'}</span></strong><small>{models.length} models selected</small></div></>}
     </div>
     <div className={'analysis-grid '+(compare?'compare-grid':'')}>{data.map(d=><section className="analysis-panel" key={d.dataset}>
      <div className="panel-heading"><div><h2>{compare?d.name:view==='history'?'Observed series & moving average':title}</h2><p>{d.name} <span>·</span> {d.frequency.toLowerCase()} observations</p></div><div className="panel-tools">{view==='decomposition'?<select aria-label="Decomposition method" value={method} onChange={e=>setMethod(e.target.value)}><option value="stl">STL (LOESS)</option><option value="additive">Classical additive</option><option value="multiplicative">Classical multiplicative</option></select>:view==='residuals'?<select aria-label="Diagnostic model" value={diagnostic||d.diagnostic} onChange={e=>setDiagnostic(e.target.value)}>{d.rows.map(r=><option key={r.Model}>{r.Model}</option>)}</select>:<span className="period-chip">{date(d.start)} — {date(d.end)}</span>}</div></div>
      {view==='verdict'?<div className="verdict"><span className="tag green"><Check size={13}/>Selected model</span><h3>{d.winner.Model}</h3><p className="specification">{d.winner.Specification}</p><div className="verdict-metrics"><div><span>Test RMSE{d.dataset==='electricity_daily'?' (GWh)':''}</span><strong>{number(d.winner.RMSE,2)}</strong></div><div><span>Test MAPE</span><strong>{number(d.winner['MAPE (%)'],2)}%</strong></div><div><span>Ljung–Box p</span><strong>{number(d.winner['Ljung–Box p'],4)}</strong></div></div><p>{d.noPassingModel?'No selected model passed the residual check. This model has the lowest holdout RMSE, but residual autocorrelation remains a concern.':'This model has the lowest holdout RMSE among the selected models that pass the Ljung–Box residual check.'}</p>{d.baselineImprovement!=null&&<p>{d.baselineImprovement>0?`${number(d.baselineImprovement,1)}% lower RMSE than the Seasonal Naive baseline.`:'The selected model does not improve on the Seasonal Naive baseline.'}</p>}</div>:<>
       {view==='residuals'&&<div className="residual-summary"><span>Residual mean <b>{number(d.residualSummary?.mean,2)}</b></span><span>Standard deviation <b>{number(d.residualSummary?.std,2)}</b></span><span>Ljung–Box p <b>{number(d.residualSummary?.p,4)}</b></span></div>}
       {d.figures.map((f,i)=><Plot key={i} figure={f} view={view}/>)}
       {['metrics','stationarity'].includes(view)&&<Table rows={d.rows} winner={d.winner?.Model} unit={d.unit}/>}
      </>}
      {d.errors.length>0&&<details className="fit-errors"><summary>{d.errors.length} model{d.errors.length>1?'s':''} could not be fitted</summary>{d.errors.map(e=><p key={e.model}><strong>{e.model}:</strong> {e.message}</p>)}</details>}
      <div className="chart-footnote"><span className="source-dot"/>{view==='history'?`Centered ${windowSize}-period moving average`:['forecast','metrics','residuals','verdict'].includes(view)?`Evaluated on ${d.holdout} held-out observations`:`Seasonal period: ${d.period}`}<span className="footnote-source">Source: {d.dataset==='ae_monthly'?'NHS England':'NESO'}</span></div>
     </section>)}</div>
     <section className="insight"><div className="insight-icon"><ChartNoAxesCombined size={20}/></div><div><span className="overline">READING THE ANALYSIS</span><h3>{explanations[view][0]}</h3><p>{explanations[view][1]}</p></div>{view==='history'&&<button onClick={()=>go('decomposition')}>Explore decomposition<ArrowRight size={15}/></button>}</section>
     <div className="next-views"><div><span className="overline">CONTINUE EXPLORING</span><p>The next question is yours.</p></div><div>{(group.name==='Explore'?[['future','Forecast a future year'],['metrics','Review model results']]:[['history','Back to the series'],['verdict','View final verdict']]).map(([id,label])=><button key={id} onClick={()=>go(id)}>{label}<ArrowUpRight size={16}/></button>)}</div></div>
    </>}
    <footer className="page-footer"><span>Forecast Studio <span className="footer-slash">/</span> Applied Time Series Analysis</span><span>Built on observed data.</span></footer>
   </div>
  </main>
  {settings&&draft&&<div className="modal-backdrop" onClick={()=>setSettings(false)}><section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={e=>e.stopPropagation()}><div className="dialog-header"><div><span className="overline">WORKSPACE CONTROLS</span><h2 id="settings-title">Analysis settings</h2></div><button autoFocus className="icon-button" aria-label="Close settings" onClick={()=>setSettings(false)}><X size={20}/></button></div><div className="dialog-body"><div className="settings-section"><h3>Series & holdout</h3><label className="range-label" htmlFor="window">Moving average window <strong>{draft.window} periods</strong></label><input id="window" type="range" min="3" max="30" value={draft.window} onChange={e=>setDraft({...draft,window:e.target.value})}/><p className="field-help">A longer window reveals broader trends.</p><div className="holdout-fields">{[['ae_monthly','NHS A&E','months',3,36],['electricity_daily','UK Electricity','days',7,180]].map(([id,label,unit,min,max])=><label key={id}>{label} holdout<div className="number-field"><input type="number" min={min} max={max} value={draft.holdouts[id]} onChange={e=>setDraft({...draft,holdouts:{...draft.holdouts,[id]:e.target.value===''?'':Number(e.target.value)}})}/><span>{unit}</span></div><small>{min}–{max} {unit}</small></label>)}</div></div><div className="settings-section"><div className="section-label"><h3>Benchmark models</h3><span>{draft.models.length} selected</span></div><p className="field-help">Complex models may take longer on the first run.</p><div className="model-options">{catalog?.models.map(name=><label key={name}><input type="checkbox" checked={draft.models.includes(name)} onChange={e=>setDraft({...draft,models:e.target.checked?[...draft.models,name]:draft.models.filter(n=>n!==name)})}/>{name}</label>)}</div>{!draft.models.length&&<p className="validation">Select at least one model.</p>}</div></div><div className="dialog-footer"><button className="button secondary" onClick={()=>setSettings(false)}>Cancel</button><button className="button primary" disabled={!catalog||!draft.models.length||!(draft.holdouts.ae_monthly>=3&&draft.holdouts.ae_monthly<=36&&draft.holdouts.electricity_daily>=7&&draft.holdouts.electricity_daily<=180)} onClick={apply}>Apply settings<ArrowRight size={15}/></button></div></section></div>}
  {help&&<div className="modal-backdrop" onClick={()=>setHelp(false)}><section className="about-dialog" role="dialog" aria-modal="true" aria-labelledby="about-title" onClick={e=>e.stopPropagation()}><button autoFocus className="icon-button" aria-label="Close information" onClick={()=>setHelp(false)}><X size={18}/></button><Activity size={28}/><h2 id="about-title">Forecast Studio</h2><p>A comparative forecasting workspace for Applied Time Series Analysis. Explore NHS England A&E attendances and NESO electricity demand, then evaluate models on held-out observations.</p><p>All charts and metrics are calculated from your project’s datasets and Python analysis functions.</p></section></div>}
 </div>;
}
createRoot(document.getElementById('root')).render(<App/>);
