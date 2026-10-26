"""Explicit, bounded hosted encoder; never silently substitutes embedding spaces."""
import os
from urllib.parse import urlsplit
import numpy as np
from openai import OpenAI

MODEL = 'text-embedding-3-small'
DIMENSIONS = 384


def encode(texts):
    if not texts or len(texts)>64 or any(not isinstance(t,str) or not t.strip() or len(t)>12000 for t in texts):
        raise ValueError('Bounded nonempty embedding inputs required.')
    base=os.getenv('LLM_BASE_URL','');url=urlsplit(base)
    if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Hosted embeddings require a configured HTTPS provider.')
    with OpenAI(base_url=base,api_key=os.environ['LLM_API_KEY'],timeout=30,max_retries=0) as client:
        result=client.embeddings.create(model=MODEL,dimensions=DIMENSIONS,input=texts)
    if result.model!=MODEL or sorted(item.index for item in result.data)!=list(range(len(texts))):
        raise ValueError('Unexpected embedding model or response indices.')
    values=np.array([item.embedding for item in sorted(result.data,key=lambda item:item.index)],dtype=np.float32)
    if values.shape!=(len(texts),DIMENSIONS) or not np.isfinite(values).all() or np.any(np.linalg.norm(values,axis=1)==0):
        raise ValueError('Invalid actual embedding vectors.')
    return values
