"""Bounded single-process development request budgets;not a distributed WAF."""
from collections import OrderedDict,deque
import hashlib
import threading
import time
from fastapi import HTTPException

class RequestBudget:
    def __init__(self,limit=300,expensive_limit=12,window=60,max_clients=10000,clock=time.monotonic):
        self.limit=limit;self.expensive_limit=expensive_limit;self.window=window;self.max_clients=max_clients;self.clock=clock;self.clients=OrderedDict();self.lock=threading.Lock()
    def check(self,request):
        # Do not trust caller-supplied Forwarded/X-Forwarded-For headers.
        host=request.client.host if request.client else 'unknown'
        key=hashlib.sha256(host.encode()).digest();at=self.clock()
        expensive=request.method=='POST' and any(request.url.path.endswith(p) for p in ('/stream','/assessment','/draft-assist'))
        with self.lock:
            while self.clients and next(iter(self.clients.values()))[2]<=at-self.window:self.clients.popitem(last=False)
            entry=self.clients.get(key)
            if entry is None:
                if len(self.clients)>=self.max_clients:raise HTTPException(429,'Development request capacity reached.',headers={'Retry-After':str(self.window)})
                entry=(deque(),deque(),at)
            general,model,_=entry
            for queue in (general,model):
                while queue and queue[0]<=at-self.window:queue.popleft()
            self.clients[key]=(general,model,at);self.clients.move_to_end(key)
            if len(general)>=self.limit or (expensive and len(model)>=self.expensive_limit):raise HTTPException(429,'Development request limit reached.Try again later.',headers={'Retry-After':str(self.window)})
            general.append(at)
            if expensive:model.append(at)
