"""Deterministic review of already-authorized records;no clinical inference.

Callers own patient/tenant/consent authorization. This module never fetches
records,reads documents,converts units,or changes the chart.
"""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import re

MAX_RECORDS=200


def _time(value):
    try:
        stamp=datetime.fromisoformat(value)
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (TypeError,ValueError):return None


def _data(entry):
    value=entry.get('data',{})
    return json.loads(value) if isinstance(value,str) else value


def review(entries,as_of=None):
    at=as_of or datetime.now(timezone.utc)
    if at.tzinfo is None:raise ValueError('Review timestamp requires a timezone.')
    rows=sorted(entries,key=lambda row:(_time(row.get('recorded_at')) or datetime.min.replace(tzinfo=timezone.utc),row['id']))[-MAX_RECORDS:]
    timeline=[];groups=defaultdict(list);excluded={'draft':0,'future':0,'invalid_time':0}
    for row in rows:
        data=_data(row);stamp=_time(row.get('recorded_at'));flags=[]
        if stamp is None:flags.append('invalid_time')
        elif stamp>at:flags.append('future_record')
        classification='unapproved_draft' if row['status']=='draft' else 'clinician_approved_record' if row['status']=='approved' else 'reported_record'
        timeline.append({'entry_id':row['id'],'revision':row.get('revision',1),'kind':row['kind'],'label':row['label'],'recorded_at':row['recorded_at'],'status':row['status'],'classification':classification,'source':data.get('source','Not documented'),'age_days':max(0,(at-stamp).days) if stamp else None,'review_flags':flags})
        if row['kind'] not in ('lab','vital'):continue
        if row['status']=='draft':excluded['draft']+=1;continue
        if stamp is None:excluded['invalid_time']+=1;continue
        if stamp>at:excluded['future']+=1;continue
        value=data.get('value');unit=data.get('unit')
        if type(value) not in (int,float) or not math.isfinite(value) or not isinstance(unit,str) or not unit:continue
        key=(row['kind'],row['label'],unit,data.get('source','Not documented'))
        groups[key].append({'entry_id':row['id'],'revision':row.get('revision',1),'recorded_at':row['recorded_at'],'value':value,'reference_low':data.get('reference_low'),'reference_high':data.get('reference_high')})
    trends=[]
    for (kind,label,unit,source),points in groups.items():
        by_time=defaultdict(set)
        for point in points:by_time[_time(point['recorded_at'])].add(point['value'])
        conflict=any(len(values)>1 for values in by_time.values())
        dates=sorted(by_time);delta=None;flags=[]
        if conflict:flags.append('same_time_conflicting_values')
        elif len(dates)>1:
            delta=next(iter(by_time[dates[-1]]))-next(iter(by_time[dates[0]]))
            if not math.isfinite(delta):delta=None;flags.append('arithmetic_range')
        ranges={(p['reference_low'],p['reference_high']) for p in points}
        if len(ranges)>1:flags.append('reference_range_changed')
        trends.append({'kind':kind,'label':label,'unit':unit,'source':source,'points':points,'delta_first_to_last':delta,'review_flags':flags,'comparison_blocked':conflict or 'arithmetic_range' in flags})
        if conflict:
            ids={point['entry_id'] for point in points if len(by_time[_time(point['recorded_at'])])>1}
            for item in timeline:
                if item['entry_id'] in ids:item['review_flags'].append('same_time_conflicting_values')
    return {'as_of':at.isoformat(),'timeline':timeline,'trends':trends,'records_truncated':len(entries)>MAX_RECORDS,'record_limit':MAX_RECORDS,'excluded_from_trends':excluded,'limitations':'Entered-record arithmetic only. Same kind,label,unit and source;no unit conversion,diagnosis,causal explanation or verification of record accuracy. Draft/future observations excluded from trends. Missing records are not negative findings.'}


def provider_context(entries,question,as_of=None,max_chars=16000):
    """Select whole non-draft,past records without silently truncating facts."""
    at=as_of or datetime.now(timezone.utc)
    if at.tzinfo is None or not 1000<=max_chars<=18000:raise ValueError('Invalid context bounds.')
    terms=set(re.findall(r'[a-z0-9]{3,}',question.lower()[:1000]))
    flags={row['entry_id']:row['review_flags'] for row in review(entries[-MAX_RECORDS:],at)['timeline']}
    candidates=[]
    for row in entries[-MAX_RECORDS:]:
        stamp=_time(row.get('recorded_at'))
        if row['status']=='draft' or stamp is None or stamp>at:continue
        data=_data(row)
        haystack=(row['kind']+' '+row['label']+' '+str(data.get('text',''))).lower()
        score=sum(term in haystack for term in terms)
        record={k:row[k] for k in ('id','kind','label','recorded_at','status')}
        record.update(revision=row.get('revision',1),data=data,review_flags=flags.get(row['id'],[]),record_classification='clinician_approved_record' if row['status']=='approved' else 'reported_record')
        record['snapshot_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        candidates.append((score,stamp,row['id'],record))
    result={'scope':'authorized_chart_entries_only','as_of':at.isoformat(),'selection':'At most200recent records;lexical question overlap,then recency;not clinical relevance validation. Whole records only;draft/future/undated records excluded. Omitted facts are unknown,not absent.','records':[],'records_considered':min(len(entries),MAX_RECORDS),'older_records_omitted':len(entries)>MAX_RECORDS,'records_omitted':len(entries)}
    for _,_,_,record in sorted(candidates,key=lambda item:item[:3],reverse=True):
        if len(result['records'])==20:break
        result['records'].append(record);result['records_omitted']=len(entries)-len(result['records'])
        if len(json.dumps(result))>max_chars:result['records'].pop();result['records_omitted']=len(entries)-len(result['records'])
    return result
