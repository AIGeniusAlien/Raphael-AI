"""Bounded development connectivity checks. Never prints credentials or raw errors.

`--live` permits at most one tiny inference and one tiny streaming request.
Qdrant checks are read-only: no cluster/collection creation or recreation.
"""
import argparse
import json
import os

BINDINGS=('LLM_API_KEY','LLM_BASE_URL','LLM_MODEL','QDRANT_URL','QDRANT_API_KEY')


def failure(error):
    return {'status':'failed','error_type':type(error).__name__,'http_status':getattr(error,'status_code',None)}


def check_services(live=False):
    result={'runtime_presence':{n:bool(os.getenv(n)) for n in BINDINGS},'live_requested':live}
    result['openai']={'status':'not_run' if not live else 'blocked_missing_runtime_binding'}
    result['qdrant']={'status':'not_run' if not live else 'blocked_missing_runtime_binding'}
    if not live:return result
    if os.getenv('LLM_BASE_URL') and os.getenv('LLM_API_KEY'):
        from openai import OpenAI
        try:
            with OpenAI(base_url=os.environ['LLM_BASE_URL'],api_key=os.environ['LLM_API_KEY'],timeout=25,max_retries=0) as client:
                model=os.getenv('LLM_MODEL','gpt-4o-mini')
                response=client.chat.completions.create(model=model,messages=[{'role':'user','content':'Reply exactly: Raphael synthetic connectivity check.'}],max_completion_tokens=24)
                result['openai']={'status':'passed','inference_nonempty':bool(response.choices[0].message.content),'returned_model':response.model}
                with client.chat.completions.create(model=model,messages=[{'role':'user','content':'Reply exactly: Raphael synthetic check.'}],max_completion_tokens=16,stream=True) as stream:
                    pieces=sum(bool(event.choices and event.choices[0].delta.content) for event in stream)
                result['openai']['stream_nonempty']=pieces>0
        except Exception as error:result['openai']=failure(error)
    if os.getenv('QDRANT_URL'):
        from qdrant_client import QdrantClient
        try:
            client=QdrantClient(url=os.environ['QDRANT_URL'],api_key=os.getenv('QDRANT_API_KEY') or None,timeout=20)
            try:
                names={c.name for c in client.get_collections().collections};target=os.getenv('QDRANT_COLLECTION','raphael_evidence_v1')
                result['qdrant']={'status':'passed','target_collection':target,'target_exists':target in names,'embedding_model_configured':os.getenv('EMBED_MODEL','sentence-transformers/all-MiniLM-L6-v2'),'expected_dimensions':int(os.getenv('EMBED_DIM','384')),'no_mutations':True}
                if target in names:
                    vectors=client.get_collection(target).config.params.vectors
                    result['qdrant']['actual_vectors']=vectors.model_dump() if hasattr(vectors,'model_dump') else {name:v.model_dump() for name,v in vectors.items()}
                    result['qdrant']['point_count']=client.count(target,exact=True).count
                    result['qdrant']['document_count']='not_counted'
            finally:client.close()
        except Exception as error:result['qdrant']=failure(error)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true');args=parser.parse_args()
    result=check_services(args.live);print(json.dumps(result,indent=2))
    return 0 if not args.live or all(result[n]['status']=='passed' for n in ('openai','qdrant')) else 1


if __name__=='__main__':raise SystemExit(main())
