from pathlib import Path
import time,json,hashlib,hmac,os
import pandas as pd
import streamlit as st
from pipeline import clean,kpis,scoped,replay,investigate,coverage
from model import classify
ROOT=Path(__file__).parent
DATA=ROOT/'data' if (ROOT/'data').is_dir() else ROOT
st.set_page_config(page_title='Cascade Factory Visibility',page_icon='🏭',layout='wide')
st.title('Factory health and performance')
st.caption('Cascade Industrial Manufacturing case prototype. All operational data is synthetic. UTC clocks. Read-only equipment access.')
@st.cache_data
def load(which):
    raw=pd.read_csv(DATA/which)
    return clean(raw)

def secret_users():
    try: return dict(st.secrets.get('users',{}))
    except FileNotFoundError: return {}
users=secret_users()
if users:
    with st.sidebar.form('login'):
        name=st.text_input('User'); password=st.text_input('Password',type='password'); submit=st.form_submit_button('Sign in')
    if submit:
        record=users.get(name,{})
        digest=hashlib.sha256(password.encode()).hexdigest()
        if record and hmac.compare_digest(digest,record.get('password_sha256','')): st.session_state.principal={'user':name,'role':record['role'],'plant':record.get('plant','P01')}
        else: st.sidebar.error('Sign-in failed')
    if 'principal' not in st.session_state: st.info('Sign in to view your assigned dashboard.'); st.stop()
    principal=st.session_state.principal; role=principal['role']; home=principal['plant']
    st.sidebar.write(f"Signed in: {principal['user']} / {role}")
    if st.sidebar.button('Sign out'):
        st.session_state.clear(); st.rerun()
else:
    st.sidebar.warning('Public demo role simulation. This selector is not authentication. Synthetic records only.')
    role=st.sidebar.selectbox('Demo persona',['Executive','Plant manager','Operator','Analyst'])
    home=st.sidebar.selectbox('Assigned demo plant',[f'P{i:02}' for i in range(1,7)])
mode=st.sidebar.radio('Dataset',['Six plants','One plant MVP'])
a,rejected,audit=load('six_plants_raw.csv' if mode=='Six plants' else 'one_plant_raw.csv')
a=scoped(a,role,home); rejected=scoped(rejected,role,home); audit=scoped(audit,role,home)
if a.empty: st.warning('No data for this assignment. Choose P01 for the one-plant demo.'); st.stop()
allowed=sorted(a.plant.unique()); selected=st.sidebar.multiselect('Plants in view',allowed,default=allowed)
a=a[a.plant.isin(selected)]; rejected=rejected[rejected.plant.isin(selected)]; audit=audit[audit.plant.isin(selected)]
if a.empty: st.info('Select at least one plant.'); st.stop()
lines=st.sidebar.multiselect('Lines',sorted(a.line.unique()),default=sorted(a.line.unique()))
a=a[a.line.isin(lines)]
if a.empty: st.info('Select a line.'); st.stop()
scope_key=(role,home,mode,tuple(selected),tuple(lines))
if st.session_state.get('scope_key')!=scope_key:
    for key in ['queue','investigation','investigation_context']: st.session_state.pop(key,None)
    st.session_state.scope_key=scope_key
day=st.sidebar.slider('Historical day in October 2026',1,28,28)
st.sidebar.caption('Seeded examples: P01 L1 days 11-12 stoppage, P02 L2 days 18-19 quality, P03 L1 days 22-23 speed. These labels are simulation design, not evaluated predictions.')
play=st.sidebar.toggle('Simulated live replay',value=False)
st.sidebar.caption('Replay reveals fixed records by ingestion time. No PLC feed or background cloud service. Every 1 second represents 4 simulation minutes.')
if 'replay_start' not in st.session_state: st.session_state.replay_start=time.time()
if st.sidebar.button('Restart replay'): st.session_state.replay_start=time.time()
window=st.sidebar.slider('Lookback hours',1,168,24)
@st.fragment(run_every='5s' if play else None)
def dashboard():
    visible,asof,finished=replay(a,time.time()-st.session_state.replay_start) if play else (a,a.ingested_at.max(),True)
    end=asof.floor('15min') if play else pd.Timestamp('2026-10-01T00:00:00Z')+pd.Timedelta(days=day)
    if not play: asof=end+pd.Timedelta(minutes=20)
    start=end-pd.Timedelta(hours=window)
    visible=visible[visible.ingested_at<=asof]
    df=visible[(visible.timestamp>=start)&(visible.timestamp<end)]
    if df.empty: st.info('Waiting for the first accepted interval.'); return
    # Operator has one line, so calculate its expected grid separately.
    expected=len(selected)*len(lines)*window*4
    cov=min(len(df)/expected,1)
    lag=max((asof-(df.timestamp.max()+pd.Timedelta(minutes=15))).total_seconds()/60,0)
    st.caption(f"{'SIMULATED LIVE' if play else 'HISTORICAL SIMULATION'} / As of {asof.isoformat()} / Window {start.isoformat()} to {end.isoformat()} / {'Replay complete' if play and finished else 'Replay active' if play else 'Snapshot'}")
    k=kpis(df); cs=st.columns(5)
    for col,label,key in zip(cs,['OEE','Availability','Performance','Quality weighted by cycle','First pass yield'],['oee','availability','performance','quality','yield']):
        col.metric(label,'N/A' if k[key] is None else f'{k[key]:.1%}')
    st.write(f"Coverage **{cov:.1%}** · Latest interval age **{lag:.0f} min** · Energy **{k['energy_per_good']:.3f} kWh/good unit**")
    if cov<.9 or lag>60: st.warning('Incomplete or stale window. KPIs use accepted intervals only. Do not interpret the result as full-plant performance.')
    overview,agent,actions,quality=st.tabs(['Performance','AI investigation','Review queue','Data trust'])
    with overview:
        if role=='Executive': st.subheader('Where should leadership focus?')
        elif role=='Operator': st.subheader('Current line losses and next review')
        else: st.subheader('Plant and line performance')
        summary=[]
        for (plant,line),g in df.groupby(['plant','line']):
            v=kpis(g); summary.append({'plant':plant,'line':line,'OEE %':round(v['oee']*100,2),'lost availability min':v['availability_loss_min'],'lost speed min':v['speed_loss_min'],'lost quality min':v['quality_loss_min'],'good units':v['good_count']})
        st.dataframe(pd.DataFrame(summary).round(1),hide_index=True,width='stretch')
        trend=df.assign(hour_utc=df.timestamp.dt.floor('h').dt.strftime('%m-%d %H:%MZ')).groupby('hour_utc').apply(lambda g:kpis(g)['oee'],include_groups=False)
        st.line_chart(trend.rename('OEE'),height=230)
        losses=pd.Series({'Availability':k['availability_loss_min'],'Speed':k['speed_loss_min'],'Quality':k['quality_loss_min']})
        st.bar_chart(losses.rename('Lost minutes'),height=220)
        if cov>=.9 and lag<=60 and k['oee']<.6:
            st.warning('Review recommended: OEE below the illustrative 60% threshold. Validate the evidence before taking action.')
        st.caption('The sum of three losses plus ideal good minutes equals planned minutes. Cross-plant ranking is directional until mix, schedule and data coverage align.')
        if role=='Executive':
            with st.expander('Illustrative business case assumptions'):
                incidents=st.number_input('Incidents per year across all six plants',1,100,12)
                avoid=st.slider('Avoidable share recovered',0.,1.,.2,.05); realize=st.slider('Realization factor',0.,1.,.5,.05)
                annual=incidents*180000*avoid*realize; net=annual-60000
                st.write(f'Illustrative annual incident benefit: USD {annual:,.0f}. Annual run cost assumption: USD 60,000. Net run-rate benefit: USD {net:,.0f}.')
                st.caption('These are assumptions, not measured savings. USD 180,000 is the case estimate of cost already incurred per incident, not guaranteed recoverable cost.')
    with agent:
        st.subheader('Investigate the accepted evidence')
        st.caption('Local ML loss classifier plus deterministic KPI tools. No language model, paid API, or external data transfer. Causes remain unverified.')
        question=st.text_input('Question','Why is OEE low in this window?')
        if st.button('Investigate'):
            result=investigate(df,question,asof,cov)
            if result['status']=='hypothesis':
                vals=[k[x] for x in ['availability_loss_min','speed_loss_min','quality_loss_min']]; total=sum(vals)
                result['ml']=classify([v/total for v in vals]) if total else {'label':'uncertain'}
            st.session_state['investigation']=result
            st.session_state['investigation_context']=(tuple(selected),str(start),str(end),role)
        result=st.session_state.get('investigation')
        if result and st.session_state.get('investigation_context')==(tuple(selected),str(start),str(end),role):
            st.write(result['summary']); st.write(result['action'])
            if 'ml' in result: st.write('ML supporting hypothesis:',result['ml'])
            if result['evidence']: st.dataframe(result['evidence'],hide_index=True)
            if 'limitations' in result: st.caption(result['limitations'])
            if result['status']=='hypothesis' and st.button('Draft review item'):
                item={'id':f"R{len(st.session_state.get('queue',[]))+1:03}",'plant_scope':','.join(selected),'role':role,'status':'Draft','action':result['action'],'evidence_ids':[x['id'] for x in result['evidence']],'created_utc':pd.Timestamp.now(tz='UTC').isoformat()}
                st.session_state.setdefault('queue',[]).append(item); st.success('Draft added for human review.')
    with actions:
        st.caption('Session-only review demonstration. Download before closing. Production requires durable workflow storage and identity. No work order or equipment command is sent.')
        queue=st.session_state.get('queue',[])
        if queue:
            st.dataframe(queue)
            if role in ['Plant manager','Analyst']:
                pick=st.selectbox('Review item',[x['id'] for x in queue]); state=st.selectbox('Human review decision',['Approved for manual follow-up','Rejected','Closed'])
                if st.button('Record review'):
                    for item in queue:
                        if item['id']==pick: item['status']=state; item['reviewer_role']=role; item['reviewed_utc']=pd.Timestamp.now(tz='UTC').isoformat()
                    st.success('Review recorded in this session.')
            st.download_button('Download review log',json.dumps(queue,indent=2),'review_log.json','application/json')
        else: st.info('Investigate an issue and draft a review item.')
    with quality:
        st.write(f'Authorized raw rows: {len(audit):,}. Authorized quarantined rows: {len(rejected):,}. Late accepted rows in window: {int(df.late_arrival.sum()):,}. Missing optional sensor rows: {int(df.sensor_missing.sum()):,}.')
        st.caption('Quarantine counts cover the authorized dataset, not just the selected KPI time window. Missing sensors do not invalidate OEE. No zero-fill or count imputation.')
        if role=='Analyst':
            st.dataframe(rejected.head(100),hide_index=True)
            st.download_button('Download authorized audit',audit.to_csv(index=False),'cleaning_audit.csv','text/csv')
        st.download_button('Download authorized accepted window',df.to_csv(index=False),'accepted_window.csv','text/csv')
dashboard()

