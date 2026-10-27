"""Opt-in read-only live retrieval evaluation on the six reviewed CDC topics.

This measures catalog routing/source integrity,not clinical reasoning or safety.
No patient context,model answer generation,collection writes or fixture providers.
"""
import argparse
import json
import os
from pathlib import Path
from curated_evidence import search,fetch_sources,URLS,MODEL,DIMENSIONS,collection

CASES=(
    ('antibiotics','Do antibiotics work on viruses?',URLS[0],'adult'),
    ('stroke','What are signs and symptoms of stroke?',URLS[3],'adult'),
    ('diabetes','What are signs and symptoms of diabetes?',URLS[4],'adult'),
    ('asthma','What are asthma symptoms?',URLS[5],'pediatric'),
    ('missing','violet gyroscopic calibrator xylophone',None,'adult'),
)


def run():
    if not all(os.getenv(key) for key in ('LLM_API_KEY','LLM_BASE_URL','QDRANT_URL','QDRANT_API_KEY')):
        raise ValueError('Secure live provider bindings required;no fallback or request.')
    sources={item['url']:item for item in fetch_sources()}
    cases=[]
    for identifier,query,expected,population in CASES:
        hits=search(query,top_k=6,population=population)
        integrity=all(hit['url'] in sources and hit['content_sha256']==sources[hit['url']]['content_sha256'] and hit['text'] in sources[hit['url']]['text'] for hit in hits)
        applicability=all(hit['population'] in (population,'general') for hit in hits)
        rank=next((index for index,item in enumerate(hits,1) if item['url']==expected),None) if expected else None
        passed=(rank is not None if expected else not hits) and integrity and applicability
        cases.append({'case_id':identifier,'query':query,'population':population,'expected_source':expected,'expected_source_rank':rank,'source_integrity':integrity,'population_filter_passed':applicability,'status':'passed' if passed else 'failed','returned_chunk_ids':[item['chunk_id'] for item in hits],'returned_urls':[item['url'] for item in hits]})
    topic=[case for case in cases if case['expected_source']]
    return {'status':'passed' if all(case['status']=='passed' for case in cases) else 'failed','uses_mock_services':False,'read_only':True,'collection':collection(),'embedding_model':MODEL,'actual_dimensions':DIMENSIONS,'catalog_source_recall_at_6':sum(case['expected_source_rank'] is not None for case in topic)/len(topic),'catalog_source_mrr':sum(1/case['expected_source_rank'] if case['expected_source_rank'] else 0 for case in topic)/len(topic),'clinical_accuracy_evaluated':False,'coverage_limit':'Four source-routing questions plus one missing-evidence case;not comprehensive or expert clinical validation.','cases':cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:result=run()
    except Exception as error:result={'status':'blocked','uses_mock_services':False,'error_type':type(error).__name__,'http_status':getattr(getattr(error,'response',None),'status_code',None)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='cases'}))
    return int(result['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
