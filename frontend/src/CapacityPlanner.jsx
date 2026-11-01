import React,{useMemo,useState} from 'react';
import {calculateCapacity} from './capacity.mjs';

export default function CapacityPlanner({data,Plot,Table,number}){
 const electricity=data.dataset==='electricity_daily';
 const [capacity,setCapacity]=useState(electricity?700:2400000);
 const unit=electricity?'GWh / day':'attendances / month';
 const summary=useMemo(()=>calculateCapacity(data,capacity),[data,capacity]);
 const figure=useMemo(()=>({...data.figures[0],layout:{...data.figures[0].layout,shapes:[...(data.figures[0].layout.shapes||[]),{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:capacity,y1:capacity,line:{color:'#b67b47',width:2,dash:'dash'}}],annotations:[...(data.figures[0].layout.annotations||[]),{xref:'paper',x:0.01,yref:'y',y:capacity,text:`Planning capacity: ${number(capacity)} ${electricity?'GWh/day':'attendances/month'}`,showarrow:false,yshift:12,font:{size:10,color:'#a27244'}}]}}),[data,capacity]);
 return <div className="capacity-planner">
  <div className="capacity-heading"><div><span className="overline">CAPACITY PLANNING</span><h3>Where could demand exceed your limit?</h3><p>This is a planning assumption, not a measured operational capacity.</p></div><strong>{number(capacity)} <small>{unit}</small></strong></div>
  <label className="capacity-range">Set planning capacity<input aria-label={`${data.name} planning capacity`} type="range" min={electricity?100:500000} max={electricity?1500:5000000} step={electricity?10:25000} value={capacity} onChange={e=>setCapacity(Number(e.target.value))}/></label>
  <div className="future-summary"><div><span>Months with point-forecast overload</span><strong>{summary.months}<small>of {summary.rows.length} shown</small></strong></div><div><span>{electricity?'Total additional daily energy needed':'Total additional monthly capacity needed'}</span><strong>{number(summary.excess,electricity?1:0)}<small>{electricity?'GWh':'attendances'}</small></strong></div><div><span>{electricity?'Worst daily shortfall':'Worst monthly shortfall'}</span><strong>{number(summary.worst,electricity?1:0)}<small>{electricity?'GWh':'attendances'}</small></strong></div></div>
  <Plot figure={figure} view="future"/>
  <Table rows={summary.rows} unit={data.unit}/>
  <div className="future-notes"><p>“Forecast above” means the point forecast exceeds capacity. “Interval reaches above” means only an upper interval bound crosses it. These are interval comparisons, not overload probabilities. Totals count positive shortfalls only, without offsetting spare capacity in other periods. {electricity&&'Electricity capacity is checked daily; monthly rows summarize those daily checks.'}</p></div>
 </div>;
}
