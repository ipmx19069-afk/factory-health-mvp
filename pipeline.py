"""Deterministic KPI and bounded investigation engine. Demo data is synthetic."""
import hashlib, math
import pandas as pd
VERSION='1.0.0'
KEY=['plant','line','timestamp']
NUM=['planned_min','run_min','ideal_cycle_sec','total_count','good_count','energy_kwh']
COLS=['event_id',*KEY,'ingested_at','product',*NUM,'reason','source','vibration_mm_s','temperature_c']
PLANTS=[f'P{i:02}' for i in range(1,7)]
def clean(raw):
    missing=set(COLS)-set(raw.columns)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    df=raw.copy().reset_index(drop=True); df['_row']=df.index+2
    reasons={i:[] for i in df.index}
    for c in ['timestamp','ingested_at']: df[c]=pd.to_datetime(df[c],utc=True,errors='coerce')
    for c in NUM+['vibration_mm_s','temperature_c']: df[c]=pd.to_numeric(df[c],errors='coerce')
    for i,r in df.iterrows():
        def bad(s): reasons[i].append(s)
        if any(pd.isna(r[c]) or str(r[c]).strip()=='' for c in ['event_id','plant','line','product','reason','source']): bad('missing_dimension')
        for c,allowed in [('plant',PLANTS),('line',['L1','L2']),('product',['pump','valve','flow_control']),('source',['MES','manual'])]:
            if r[c] not in allowed: bad('unknown_'+c)
        if pd.isna(r.timestamp) or pd.isna(r.ingested_at): bad('invalid_timestamp')
        elif r.ingested_at<r.timestamp+pd.Timedelta(minutes=15): bad('ingestion_before_interval_end')
        if any(not math.isfinite(r[c]) for c in NUM): bad('missing_or_nonfinite_measure')
        else:
            if not 0<r.planned_min<=15: bad('planned_range')
            if not 0<=r.run_min<=r.planned_min: bad('run_range')
            if r.ideal_cycle_sec<=0: bad('cycle_range')
            if not 0<=r.good_count<=r.total_count: bad('count_order')
            if r.total_count<0 or any(r[c]!=int(r[c]) for c in ['total_count','good_count']): bad('count_integer')
            if r.energy_kwh<0: bad('negative_energy')
            if r.total_count*r.ideal_cycle_sec>r.run_min*60+1e-6: bad('performance_above_one')
        if pd.notna(r.timestamp) and (r.timestamp.minute%15 or r.timestamp.second or r.timestamp.microsecond): bad('off_grid_interval')
    eligible=df.loc[[i for i in df.index if not reasons[i]]]
    for _,g in eligible.groupby(KEY,dropna=False):
        if len(g)>1:
            if len(g.drop(columns=['_row','ingested_at']).drop_duplicates())==1:
                keep=g.sort_values(['ingested_at','_row']).index[0]
                for i in g.index:
                    if i!=keep: reasons[i].append('exact_duplicate')
            else:
                for i in g.index: reasons[i].append('conflicting_duplicate')
    eligible=df.loc[[i for i in df.index if not reasons[i]]]
    for i in eligible.index[eligible.event_id.duplicated(keep=False)]: reasons[i].append('event_id_collision')
    df['disposition']=[';'.join(reasons[i]) or 'accepted' for i in df.index]
    a=df[df.disposition=='accepted'].copy()
    a['late_arrival']=(a.ingested_at-a.timestamp).dt.total_seconds()>1800
    a['sensor_missing']=a[['vibration_mm_s','temperature_c']].isna().any(axis=1)
    a['ideal_total_min']=a.total_count*a.ideal_cycle_sec/60
    a['ideal_good_min']=a.good_count*a.ideal_cycle_sec/60
    a['availability_loss_min']=a.planned_min-a.run_min
    a['speed_loss_min']=a.run_min-a.ideal_total_min
    a['quality_loss_min']=a.ideal_total_min-a.ideal_good_min
    return a,df[df.disposition!='accepted'].copy(),df[['_row','event_id',*KEY,'disposition']]
def kpis(df):
    cols=['planned_min','run_min','ideal_total_min','ideal_good_min','total_count','good_count','energy_kwh','availability_loss_min','speed_loss_min','quality_loss_min']
    s={c:float(df[c].sum()) for c in cols}
    def div(a,b): return s[a]/s[b] if s[b] else None
    return {**s,'availability':div('run_min','planned_min'),'performance':div('ideal_total_min','run_min'),'quality':div('ideal_good_min','ideal_total_min'),'yield':div('good_count','total_count'),'oee':div('ideal_good_min','planned_min'),'energy_per_good':div('energy_kwh','good_count')}
def coverage(df,plants,start,end):
    expected=len(plants)*2*len(pd.date_range(start,end,freq='15min',inclusive='left'))
    n=len(df[(df.timestamp>=start)&(df.timestamp<end)&df.plant.isin(plants)])
    return min(n/expected,1) if expected else None
def scoped(df,role,plant='P01'):
    if role in ['Executive','Analyst']: return df.copy()
    if role in ['Plant manager','Operator']:
        out=df[df.plant==plant]
        return out[out.line=='L1'].copy() if role=='Operator' else out.copy()
    raise PermissionError('Unknown role')
def replay(df,elapsed_seconds,speed=4):
    cut=min(df.ingested_at.min()+pd.Timedelta(seconds=max(elapsed_seconds,0)*60*speed),df.ingested_at.max())
    return df[df.ingested_at<=cut].copy(),cut,cut>=df.ingested_at.max()
def investigate(df,question,asof=None,data_coverage=1.0):
    q=question.lower()
    blocked=['ignore previous','system prompt','api key','password','bypass','disable','restart','shut down','setpoint','plc write','employee','operator name']
    if any(x in q for x in blocked): return {'status':'refused','summary':'Only aggregate evidence review is supported. Operational changes require qualified human authorization.','evidence':[],'action':'Escalate to the plant manager.'}
    if not any(x in q for x in ['oee','loss','quality','energy','performance','investigat','why','availability','compare']): return {'status':'unsupported','summary':'Ask about OEE, availability, performance, quality, energy, or losses.','evidence':[],'action':'Choose a supported investigation.'}
    if df.empty: return {'status':'insufficient','summary':'No accepted records in this authorized window.','evidence':[],'action':'Check ingestion and selected window.'}
    asof=pd.Timestamp(asof) if asof is not None else df.ingested_at.max()
    age=(asof-(df.timestamp.max()+pd.Timedelta(minutes=15))).total_seconds()/60
    if age>60 or data_coverage<.9: return {'status':'insufficient','summary':f'Data quality gate: coverage {data_coverage:.1%}, age {max(age,0):.0f} min. Diagnosis withheld.','evidence':[],'action':'Validate missing or stale records with the data steward.'}
    cols=['availability_loss_min','speed_loss_min','quality_loss_min']; loss=df[cols].sum(); driver=loss.idxmax()
    labels={'availability_loss_min':'availability','speed_loss_min':'speed','quality_loss_min':'quality'}; label=labels[driver]
    worst=df.sort_values(driver,ascending=False).head(5)
    evidence=[{'id':r.event_id,'plant':r.plant,'line':r.line,'timestamp':r.timestamp.isoformat(),'loss_min':round(float(r[driver]),3),'reason':str(r.reason)[:80]} for _,r in worst.iterrows()]
    k=kpis(df)
    summary=f"OEE is {k['oee']:.1%}. Largest loss is {label}: {loss[driver]:.1f} min. This is a prioritization hypothesis, not a verified root cause."
    if 'energy' in q: summary=f"Energy per good unit is {k['energy_per_good']:.3f} kWh. Product mix affects comparison. "+summary
    actions={'availability':'Review stoppage logs and maintenance history with the shift lead.','speed':'Check cycle master, changeovers and schedule with the production engineer.','quality':'Ask the quality lead to review lot inspection and process parameters.'}
    return {'status':'hypothesis','driver':label,'summary':summary,'evidence':evidence,'action':actions[label],'limitations':'Reason codes are unverified annotations. No individual attribution or machine control. Sensor correlation is not causation.','version':VERSION}
def file_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()

