"""Bounded CDC public-text corpus and real hosted-embedding Qdrant retrieval.

Explicit CLI ingestion only; no background ingestion or runtime fixture provider.
Original versions are retained. Never mixes model spaces in an existing collection.
"""
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
import re
import uuid
from datetime import date,datetime,timezone
from urllib.parse import urlsplit

import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_config import connection_options
from rag.hosted_embeddings import encode,MODEL,DIMENSIONS

CORPUS='cdc-antibiotic-demo-v1'
POLICY='https://www.cdc.gov/other/agencymaterials.html'
URLS=(
 'https://www.cdc.gov/antibiotic-use/about/index.html',
 'https://www.cdc.gov/antibiotic-use/hcp/clinical-care/adult-outpatient.html',
 'https://www.cdc.gov/antibiotic-use/hcp/clinical-care/pediatric-outpatient.html',
)
NOTICE='Source: CDC. Original material is available free on the linked CDC website. Raphael is not endorsed by CDC, HHS or the U.S. Government. Text excerpts retain original wording; consult the current source.'


def now():return datetime.now(timezone.utc).isoformat()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def collection():return os.getenv('QDRANT_COLLECTION','raphael_evidence_v1')
def client():
    url=os.environ['QDRANT_URL'];parsed=urlsplit(url)
    if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError('HTTPS Qdrant destination required.')
    return QdrantClient(url=url,api_key=os.environ['QDRANT_API_KEY'],timeout=25,**connection_options(url))


class BodyText(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.depth=0;self.skip=0;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=='div':
            if self.depth:self.depth+=1
            elif 'cdc-dfe-body' in dict(attrs).get('class','').split():self.depth=1
        if self.depth:
            if tag in ('script','style'):self.skip+=1
            if tag in ('p','li','tr','h2','h3','h4','td','th'):self.parts.append(' ')
    def handle_endtag(self,tag):
        if tag in ('script','style') and self.skip:self.skip-=1
        if tag=='div' and self.depth:self.depth-=1
        if self.depth and tag in ('p','li','tr','h2','h3','h4','td','th'):self.parts.append(' ')
    def handle_data(self,text):
        if self.depth and not self.skip:self.parts.append(text)


def parse_source(url,markup,policy_hash):
    parser=BodyText();parser.feed(markup);body=' '.join(''.join(parser.parts).split())
    title=re.search(r'<h1[^>]*>(.*?)</h1>',markup,re.S|re.I)
    dated=re.search(r'<time[^>]*datetime="(\d{4}-\d{2}-\d{2})',markup,re.I)
    if not title or not dated or len(body)<100 or any(x in body.lower() for x in ('copyright ©','all rights reserved')):
        raise ValueError('Source body/date/title or reuse review is incomplete.')
    published=dated.group(1)
    if date.fromisoformat(published)>datetime.now(timezone.utc).date():raise ValueError('Future source date.')
    return {'url':url,'title':html.unescape(re.sub('<[^>]+>','',title.group(1))).strip(),'source_date':published,
            'text':body,'content_sha256':digest(body),'source_html_sha256':digest(markup),'retrieved_at':now(),
            'license':'CDC public-domain government text;exceptions reviewed;no images/logos/third-party works',
            'license_url':POLICY,'license_policy_sha256':policy_hash,'attribution_notice':NOTICE,
            'org':'CDC','jurisdiction':'United States','source_type':'official_clinical_guidance_summary' if '/hcp/' in url else 'official_patient_education',
            'evidence_quality':'Official CDC guidance/education;not independently GRADE-rated or a systematic review','evidence_strength':None,'tier':'B' if '/hcp/' in url else 'D'}


def fetch_sources():
    with httpx.Client(timeout=25,follow_redirects=False,headers={'User-Agent':'Raphael-development-evidence/1.0'}) as web:
        def fetch(url):
            with web.stream('GET',url) as response:
                response.raise_for_status()
                if response.status_code!=200 or 'text/html' not in response.headers.get('content-type',''):raise ValueError('Expected direct CDC HTML response.')
                data=bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data)>2_000_000:raise ValueError('Evidence page exceeds limit.')
                return data.decode('utf-8')
        policy=fetch(POLICY)
        if 'public domain' not in policy.lower() or 'endorsement' not in policy.lower():raise ValueError('CDC reuse policy must be reviewed.')
        return [parse_source(url,fetch(url),digest(policy)) for url in URLS]


def chunks(body,size=1100,overlap=150):
    start=0
    while start<len(body):
        end=min(start+size,len(body));yield body[start:end]
        if end==len(body):break
        start=end-overlap


def inventory(c):
    total=c.count(collection(),exact=True).count
    if total>10000:raise ValueError('Collection exceeds bounded development inventory;manual review required.')
    points=[];offset=None
    while True:
        batch,offset=c.scroll(collection_name=collection(),limit=100,offset=offset,with_payload=True,with_vectors=False)
        points.extend(batch)
        if offset is None:break
    for point in points:
        p=point.payload or {}
        if p.get('embedding_model')!=MODEL or p.get('embedding_dimensions')!=DIMENSIONS:
            raise ValueError('Existing collection contains another or untagged embedding space;preserve it and select a separate collection.')
    return points


def ensure_compatible(c,probe,allow_create=False):
    if probe.shape!=(1,DIMENSIONS) or not np.isfinite(probe).all() or np.linalg.norm(probe)==0:raise ValueError('Actual dimension probe failed.')
    names={item.name for item in c.get_collections().collections}
    if collection() not in names:
        if not allow_create:raise ValueError('Evidence collection is absent.')
        c.create_collection(collection_name=collection(),vectors_config=qm.VectorParams(size=DIMENSIONS,distance=qm.Distance.COSINE))
    vectors=c.get_collection(collection()).config.params.vectors
    if not hasattr(vectors,'size') or vectors.size!=DIMENSIONS or str(vectors.distance).lower()!='cosine':raise ValueError('Existing collection schema is incompatible;no recreation performed.')
    points=inventory(c)
    if allow_create:
        # Managed Qdrant can require indexes for all filtered payload fields.
        indexed=c.get_collection(collection()).payload_schema
        for field in ('corpus','doc_id','source_version','embedding_model','chunk_id'):
            if field not in indexed:c.create_payload_index(collection_name=collection(),field_name=field,field_schema=qm.PayloadSchemaType.KEYWORD,wait=True)
        if 'is_current' not in indexed:c.create_payload_index(collection_name=collection(),field_name='is_current',field_schema=qm.PayloadSchemaType.BOOL,wait=True)
    return points


def ingest(sources):
    probe=encode(['Raphael nonclinical dimension compatibility probe'])
    c=client()
    try:
        before=ensure_compatible(c,probe,allow_create=True);known={str(point.id):point.payload for point in before};written=0;skipped=0
        for source in sources:
            doc_id=digest(source['url']);version=digest(source['content_sha256']+'|'+source['source_date'])
            records=[]
            for ordinal,body in enumerate(chunks(source['text'])):
                chunk_id=digest(doc_id+'|'+version+'|'+str(ordinal)+'|'+body);point_id=str(uuid.uuid5(uuid.NAMESPACE_URL,chunk_id))
                payload={**source,'text':body,'doc_id':doc_id,'source_version':version,'chunk_id':chunk_id,'ordinal':ordinal,'corpus':CORPUS,'is_current':True,'embedding_model':MODEL,'embedding_dimensions':DIMENSIONS}
                if point_id in known:
                    if known[point_id]!=payload:
                        # Retrieval time can differ on an otherwise identical verified version.
                        if any(known[point_id].get(k)!=v for k,v in payload.items() if k not in ('retrieved_at','source_html_sha256')):raise ValueError('Existing point payload differs;manual version review required.')
                    skipped+=1;continue
                records.append((point_id,payload))
            for start in range(0,len(records),32):
                batch=records[start:start+32];vectors=encode([p['text'] for _,p in batch])
                c.upsert(collection_name=collection(),points=[qm.PointStruct(id=identifier,vector=vector.tolist(),payload=payload) for (identifier,payload),vector in zip(batch,vectors)],wait=True);written+=len(batch)
            c.set_payload(collection_name=collection(),payload={'is_current':False},points=qm.Filter(must=[qm.FieldCondition(key='corpus',match=qm.MatchValue(value=CORPUS)),qm.FieldCondition(key='doc_id',match=qm.MatchValue(value=doc_id))],must_not=[qm.FieldCondition(key='source_version',match=qm.MatchValue(value=version))]),wait=True)
        after=inventory(c);active=[p.payload for p in after if p.payload.get('corpus')==CORPUS and p.payload.get('is_current')]
        return {'collection':collection(),'actual_dimensions':probe.shape[1],'embedding_model':MODEL,'vectors_before':len(before),'vectors_after':len(after),'active_vectors':len(active),'active_documents':len({p['doc_id'] for p in active}),'vectors_written':written,'vectors_deduplicated':skipped}
    finally:c.close()


def current_filter():
    return qm.Filter(must=[qm.FieldCondition(key='corpus',match=qm.MatchValue(value=CORPUS)),qm.FieldCondition(key='is_current',match=qm.MatchValue(value=True)),qm.FieldCondition(key='embedding_model',match=qm.MatchValue(value=MODEL))])


def search(query,top_k=6):
    if not isinstance(query,str) or not 1<=len(query)<=1000 or not 1<=top_k<=12:raise ValueError('Bounded evidence query required.')
    vector=encode([query]);c=client()
    try:
        ensure_compatible(c,vector)
        hits=c.search(collection_name=collection(),query_vector=vector[0].tolist(),query_filter=current_filter(),limit=top_k,score_threshold=.35,with_payload=True)
        stop={'the','and','for','with','what','this','that','from','have','does','should','patient','synthetic','explain','please','about','only','could','would','there','their','information','provide','source','evidence','uncertainty','clinical','review','sources'}
        terms=set(re.findall(r'\b[a-z]{3,}\b',query.lower()))-stop;out=[]
        for hit in hits:
            payload=hit.payload or {}
            if not terms.intersection(re.findall(r'\b[a-z]{3,}\b',payload['text'].lower())):continue
            age=(datetime.now(timezone.utc).date()-date.fromisoformat(payload['source_date'])).days
            out.append({**payload,'score':float(hit.score),'score_kind':'cosine_similarity_not_clinical_confidence','retrieval_method':'qdrant_hosted_embedding','freshness_status':'older_than_two_years' if age>730 else 'dated_within_two_years','citation_path':'/api/dev/intelligence/evidence/'+payload['chunk_id']})
        return out
    finally:c.close()


def citation(chunk_id):
    if not re.fullmatch('[0-9a-f]{64}',chunk_id):return None
    c=client()
    try:
        filters=current_filter();filters.must.append(qm.FieldCondition(key='chunk_id',match=qm.MatchValue(value=chunk_id)))
        rows,_=c.scroll(collection_name=collection(),scroll_filter=filters,limit=1,with_payload=True,with_vectors=False)
        return rows[0].payload if rows else None
    finally:c.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ingest',action='store_true',required=True);args=parser.parse_args()
    try:
        sources=fetch_sources();first=ingest(sources);second=ingest(sources)
        print(json.dumps({'first':first,'repeat':second,'sources':[{k:s[k] for k in ('url','title','source_date','content_sha256','license','license_url','license_policy_sha256','retrieved_at')} for s in sources]},indent=2))
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__,'http_status':getattr(error,'status_code',None)}));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
