"""Content-free bounded development measurements;not capacity or clinical validation."""
import math

SCHEMA='''CREATE TABLE IF NOT EXISTS dev_ai_generation_metrics (
 generation_id TEXT PRIMARY KEY,retrieval_ms REAL,total_ms REAL,first_delta_ms REAL,
 provider_ms REAL,prompt_tokens INTEGER,completion_tokens INTEGER,total_tokens INTEGER
);'''


def usage_values(usage):
    """Accept only complete internally consistent provider counts;unknown stays NULL."""
    values={key:getattr(usage,key,None) for key in ('prompt_tokens','completion_tokens','total_tokens')}
    if any(type(value) is not int or not 0<=value<=10000000 for value in values.values()):return {}
    if values['total_tokens']!=values['prompt_tokens']+values['completion_tokens']:return {}
    return values


def summary(rows):
    result={'sample_count':len(rows),'generation_usage_reports':0,'generation_tokens':None,'estimated_cost':None,'timings':{}}
    counts=[row['total_tokens'] for row in rows if type(row.get('total_tokens')) is int]
    result['generation_usage_reports']=len(counts)
    if counts:result['generation_tokens']=sum(counts)
    for key in ('retrieval_ms','total_ms','first_delta_ms','provider_ms'):
        values=sorted(float(row[key]) for row in rows if type(row.get(key)) in (float,int) and math.isfinite(row[key]) and 0<=row[key]<=3600000)
        result['timings'][key]={'samples':len(values),**{label:(round(values[max(0,math.ceil(len(values)*percent)-1)],3) if values else None) for label,percent in (('p50',.5),('p95',.95),('p99',.99))}}
    result['scope']='Nearest-rank observed timings for at most200currently authorized generations;small samples are not production percentiles. First delta is measured after retrieval;total includes application work and stream consumption. Tokens are actual provider reports only;missing/cancelled streams can have unknown usage. Embeddings,patient-chat,structured assessments and checker usage are excluded. No approved price schedule;cost is unknown. No prompt,text,patient identifiers or external telemetry.'
    return result
