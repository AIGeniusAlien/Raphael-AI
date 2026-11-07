"""Durable bounded synthetic background jobs. No arbitrary tools or clinical actions.

Workers run only explicitly allowlisted, read-only inventory operations. A
bounded worker invocation is not an installed scheduler or continuous service.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import time
from typing import Literal
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from sqlalchemy import text
from database import execute_sqlite_script
from dev_account import session_active
from dev_portal import _require_synthetic

SCHEMA = '''
CREATE TABLE IF NOT EXISTS dev_jobs (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL,
 owner_id INTEGER NOT NULL REFERENCES dev_users(id),
 request_id TEXT NOT NULL,
 payload_hash TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('evidence_inventory','requirements_snapshot')),
 state TEXT NOT NULL CHECK(state IN ('queued','running','retry_wait','succeeded','dead_letter','cancelled')),
 priority INTEGER NOT NULL CHECK(priority BETWEEN 0 AND 5),
 attempts INTEGER NOT NULL DEFAULT 0,
 max_attempts INTEGER NOT NULL CHECK(max_attempts BETWEEN 1 AND 3),
 available_at REAL NOT NULL,
 lease_until REAL,
 lease_token_hash TEXT,
 result_json TEXT,
 last_error_code TEXT,
 created_at REAL NOT NULL,
 updated_at REAL NOT NULL,
 UNIQUE(tenant,owner_id,request_id)
);
CREATE INDEX IF NOT EXISTS dev_jobs_claim ON dev_jobs(tenant,state,available_at,priority,id);
CREATE TABLE IF NOT EXISTS dev_job_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 job_id INTEGER NOT NULL REFERENCES dev_jobs(id),
 event TEXT NOT NULL,
 attempt INTEGER NOT NULL,
 created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS dev_job_events_job ON dev_job_events(job_id,id);
CREATE TRIGGER IF NOT EXISTS dev_job_events_immutable_update BEFORE UPDATE ON dev_job_events
 BEGIN SELECT RAISE(ABORT,'Job history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS dev_job_events_immutable_delete BEFORE DELETE ON dev_job_events
 BEGIN SELECT RAISE(ABORT,'Job history is append-only'); END;
CREATE TABLE IF NOT EXISTS dev_job_worker_runs (
 worker_id TEXT PRIMARY KEY,
 tenant TEXT NOT NULL,
 started_at REAL NOT NULL,
 heartbeat_at REAL NOT NULL,
 finished_at REAL,
 processed INTEGER NOT NULL DEFAULT 0
);
'''
ROLES = {'evidence_inventory': 'clinician', 'requirements_snapshot': 'org_admin'}
TERMINAL = frozenset(('succeeded', 'dead_letter', 'cancelled'))
NOTICE = ('Synthetic read-only background inventories only. No diagnosis, prescribing, clinical orders, '
          'messages, model calls, network fetches or code execution. Browser closure does not delete jobs. '
          'A separate worker must run; no permanent scheduler or 24/7 service is implied.')


def init_jobs_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c:
        execute_sqlite_script(c, SCHEMA)


class Enqueue(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: UUID
    kind: Literal['evidence_inventory', 'requirements_snapshot']
    delay_seconds: StrictInt = Field(default=0, ge=0, le=3600)
    priority: StrictInt = Field(default=0, ge=0, le=5)
    max_attempts: StrictInt = Field(default=3, ge=1, le=3)


def _event(c, identifier, event, attempt, at):
    c.execute(text('INSERT INTO dev_job_events(job_id,event,attempt,created_at) VALUES(:id,:e,:a,:at)'),
              {'id': identifier, 'e': event, 'a': attempt, 'at': at})


def _audit(c, job, event):
    from datetime import datetime, timezone
    c.execute(text('INSERT INTO dev_audit(tenant,actor_id,event,resource_type,resource_id,created_at) VALUES(:t,:u,:e,:r,:id,:at)'),
              {'t': job['tenant'], 'u': job['owner_id'], 'e': 'background_job_' + event,
               'r': 'background_job', 'id': job['id'], 'at': datetime.now(timezone.utc).isoformat()})


def _owner_active(c, job):
    return bool(c.execute(text('SELECT 1 FROM dev_users WHERE id=:u AND tenant=:t AND role=:r AND active=1'),
                          {'u': job['owner_id'], 't': job['tenant'], 'r': ROLES.get(job['kind'], '')}).scalar())


def _owned(c, identifier, user):
    row = c.execute(text('SELECT * FROM dev_jobs WHERE id=:id AND tenant=:t AND owner_id=:u'),
                    {'id': identifier, 't': user['tenant'], 'u': user['id']}).mappings().first()
    if row is None or ROLES.get(row['kind']) != user['role']:
        raise HTTPException(404, 'Background job not found.')
    return row


def _public(c, row):
    fields = ('id', 'kind', 'state', 'priority', 'attempts', 'max_attempts', 'available_at',
              'lease_until', 'last_error_code', 'created_at', 'updated_at')
    events = c.execute(text('SELECT event,attempt,created_at FROM dev_job_events WHERE job_id=:id ORDER BY id LIMIT 32'),
                       {'id': row['id']}).mappings().all()
    return {**{key: row[key] for key in fields}, 'result': json.loads(row['result_json']) if row['result_json'] else None,
            'events': [dict(e) for e in events], 'clinical_validation': False}


def install_jobs(app, ctx):
    router = APIRouter(prefix='/api/dev/jobs', tags=['bounded synthetic background jobs'],
                       dependencies=[Depends(ctx.same_origin_json)])

    def operator(user=Depends(ctx.user)):
        if user['role'] not in ('clinician', 'org_admin'):
            raise HTTPException(403, 'A permitted workflow operator role is required.')
        return user

    def active(c, request, user):
        if not session_active(c, request, user):
            raise HTTPException(401, 'Session revoked; job was not changed.')

    @router.get('/status')
    def status(user=Depends(operator)):
        at = time.time()
        with ctx.engine.begin() as c:
            counts = dict(c.execute(text('SELECT state,count(*) FROM dev_jobs WHERE tenant=:t AND owner_id=:u AND kind=:k GROUP BY state'),
                                    {'t': user['tenant'], 'u': user['id'], 'k': next(kind for kind, role in ROLES.items() if role == user['role'])}).all())
            worker = c.execute(text('SELECT heartbeat_at,finished_at FROM dev_job_worker_runs WHERE tenant=:t ORDER BY heartbeat_at DESC LIMIT 1'),
                               {'t': user['tenant']}).mappings().first()
            ctx.audit(c, user, 'review_background_job_status', 'background_job')
        return {'own_jobs_by_state': counts,
                'fresh_worker_heartbeat': bool(worker and worker['finished_at'] is None and worker['heartbeat_at'] >= at-15),
                'last_worker_heartbeat': worker['heartbeat_at'] if worker else None,
                'supported_kinds': [kind for kind, role in ROLES.items() if role == user['role']],
                'notice': NOTICE, 'persistent_scheduler_configured': False}

    @router.post('')
    def enqueue(body: Enqueue, request: Request, user=Depends(operator)):
        if ROLES[body.kind] != user['role']:
            raise HTTPException(403, 'This job kind is not permitted for the current role.')
        digest = hashlib.sha256(json.dumps(body.model_dump(mode='json'), sort_keys=True).encode()).hexdigest()
        at = time.time()
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE'); active(c, request, user)
            old = c.execute(text('SELECT * FROM dev_jobs WHERE tenant=:t AND owner_id=:u AND request_id=:r'),
                            {'t': user['tenant'], 'u': user['id'], 'r': str(body.request_id)}).mappings().first()
            if old:
                if old['payload_hash'] != digest:
                    raise HTTPException(409, 'Submission identifier was reused with a different job.')
                return dict(_public(c, old), deduplicated=True)
            if c.execute(text('SELECT count(*) FROM dev_jobs WHERE tenant=:t AND owner_id=:u'),
                         {'t': user['tenant'], 'u': user['id']}).scalar() >= 500:
                raise HTTPException(413, 'Development job history limit reached; no job was added.')
            identifier = c.execute(text("INSERT INTO dev_jobs(tenant,owner_id,request_id,payload_hash,kind,state,priority,max_attempts,available_at,created_at,updated_at) VALUES(:t,:u,:r,:h,:k,'queued',:p,:m,:due,:at,:at)"),
                                   {'t': user['tenant'], 'u': user['id'], 'r': str(body.request_id), 'h': digest,
                                    'k': body.kind, 'p': body.priority, 'm': body.max_attempts, 'due': at+body.delay_seconds, 'at': at}).lastrowid
            _event(c, identifier, 'enqueued', 0, at); ctx.audit(c, user, 'enqueue_background_job', 'background_job', identifier)
            return dict(_public(c, _owned(c, identifier, user)), deduplicated=False)

    @router.get('')
    def listing(user=Depends(operator)):
        with ctx.engine.begin() as c:
            rows = c.execute(text('SELECT * FROM dev_jobs WHERE tenant=:t AND owner_id=:u AND kind=:k ORDER BY id DESC LIMIT 100'),
                             {'t': user['tenant'], 'u': user['id'], 'k': next(kind for kind, role in ROLES.items() if role == user['role'])}).mappings().all()
            ctx.audit(c, user, 'list_background_jobs', 'background_job')
            return {'items': [_public(c, r) for r in rows], 'limit': 100, 'notice': NOTICE}

    @router.get('/{identifier}')
    def read(identifier: int, user=Depends(operator)):
        with ctx.engine.begin() as c:
            row = _owned(c, identifier, user); ctx.audit(c, user, 'read_background_job', 'background_job', identifier)
            return _public(c, row)

    @router.post('/{identifier}/cancel')
    def cancel(identifier: int, request: Request, user=Depends(operator)):
        at = time.time()
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE'); active(c, request, user); row = _owned(c, identifier, user)
            if row['state'] in TERMINAL:
                if row['state'] != 'cancelled':
                    raise HTTPException(409, 'A completed job cannot be cancelled.')
            else:
                c.execute(text("UPDATE dev_jobs SET state='cancelled',lease_until=NULL,lease_token_hash=NULL,updated_at=:at WHERE id=:id"), {'id': identifier, 'at': at})
                _event(c, identifier, 'cancelled_by_owner', row['attempts'], at)
                ctx.audit(c, user, 'cancel_background_job', 'background_job', identifier)
            return _public(c, _owned(c, identifier, user))

    app.include_router(router)


def claim(engine, tenant, *, at=None, lease_seconds=30):
    """One transactional lease; expired attempts are fenced and bounded."""
    _require_synthetic(engine)
    at = time.time() if at is None else at
    if not isinstance(tenant, str) or not 1 <= len(tenant) <= 100 or not 1 <= lease_seconds <= 300:
        raise ValueError('Bounded tenant and lease required.')
    with engine.begin() as c:
        c.exec_driver_sql('BEGIN IMMEDIATE')
        expired = c.execute(text("SELECT * FROM dev_jobs WHERE tenant=:t AND state='running' AND lease_until<=:at ORDER BY id LIMIT 100"), {'t': tenant, 'at': at}).mappings().all()
        for row in expired:
            target = 'dead_letter' if row['attempts'] >= row['max_attempts'] else 'retry_wait'
            c.execute(text('UPDATE dev_jobs SET state=:s,available_at=:due,lease_until=NULL,lease_token_hash=NULL,last_error_code=:e,updated_at=:at WHERE id=:id'),
                      {'s': target, 'at': at, 'due': at+min(60, 2**row['attempts']) if target == 'retry_wait' else at, 'e': 'lease_expired', 'id': row['id']})
            _event(c, row['id'], 'lease_expired', row['attempts'], at); _audit(c, row, target)
        # At most100 candidates per invocation; malformed/deactivated owners
        # are cancelled, not silently reassigned to another actor.
        rows = c.execute(text("SELECT * FROM dev_jobs WHERE tenant=:t AND state IN ('queued','retry_wait') AND available_at<=:at ORDER BY priority DESC,id LIMIT 100"), {'t': tenant, 'at': at}).mappings().all()
        for row in rows:
            if not _owner_active(c, row):
                c.execute(text("UPDATE dev_jobs SET state='cancelled',last_error_code='owner_authorization_changed',updated_at=:at WHERE id=:id"), {'at': at, 'id': row['id']})
                _event(c, row['id'], 'owner_authorization_changed', row['attempts'], at); _audit(c, row, 'cancelled')
                continue
            token = secrets.token_urlsafe(32); digest = hashlib.sha256(token.encode()).hexdigest(); attempt = row['attempts']+1
            c.execute(text("UPDATE dev_jobs SET state='running',attempts=:a,lease_until=:until,lease_token_hash=:h,updated_at=:at WHERE id=:id"),
                      {'id': row['id'], 'a': attempt, 'until': at+lease_seconds, 'h': digest, 'at': at})
            _event(c, row['id'], 'claimed', attempt, at); _audit(c, row, 'claimed')
            return dict(row, state='running', attempts=attempt, lease_until=at+lease_seconds, token=token)
    return None


def heartbeat(engine, job, *, at=None, lease_seconds=30):
    at = time.time() if at is None else at
    if not 1 <= lease_seconds <= 300: raise ValueError('Bounded lease required.')
    with engine.begin() as c:
        result = c.execute(text("UPDATE dev_jobs SET lease_until=:until,updated_at=:at WHERE id=:id AND state='running' AND lease_token_hash=:h AND lease_until>:at"),
                           {'until': at+lease_seconds, 'at': at, 'id': job['id'], 'h': hashlib.sha256(job['token'].encode()).hexdigest()})
        return result.rowcount == 1


class JobUnavailable(Exception):
    pass


def execute_job(engine, job):
    """No dynamic handler names, URLs, commands, prompts or patient arguments."""
    if job['kind'] == 'evidence_inventory':
        with engine.connect() as c:
            if not _owner_active(c, job): raise JobUnavailable('owner_authorization_changed')
            sources = dict(c.execute(text('SELECT status,count(*) FROM dev_evidence_sources WHERE tenant=:t GROUP BY status'), {'t': job['tenant']}).all())
            chunks = c.execute(text("SELECT count(*) FROM dev_evidence_chunks ch JOIN dev_evidence_sources s ON s.id=ch.source_id WHERE s.tenant=:t AND s.status='approved'"), {'t': job['tenant']}).scalar()
        return {'source_versions_by_status': sources, 'approved_chunks': chunks,
                'source_verification': 'clinician_attestation_only', 'clinical_validation': False}
    if job['kind'] == 'requirements_snapshot':
        try:
            raw = (Path(__file__).resolve().parent/'docs/ENGINEERING_LEDGER.json').read_bytes()
            if len(raw) > 8*1024*1024: raise ValueError()
            ledger = json.loads(raw)
            return {'requirement_count': ledger['requirement_count'], 'state_counts': ledger['state_counts'],
                    'catalog_sha256': hashlib.sha256(raw).hexdigest(), 'clinical_validation': False,
                    'scope': 'Repository catalog snapshot;not implementation or clinical acceptance.'}
        except (OSError, ValueError, KeyError, TypeError):
            raise JobUnavailable('requirements_catalog_unavailable') from None
    raise JobUnavailable('unsupported_job_kind')


def finish(engine, job, *, result=None, error_code=None, at=None):
    """Fencing prevents stale/cancelled workers from publishing results."""
    at = time.time() if at is None else at
    if error_code is not None and error_code not in ('worker_operation_failed', 'requirements_catalog_unavailable', 'owner_authorization_changed', 'unsupported_job_kind'):
        error_code = 'worker_operation_failed'
    encoded = json.dumps(result, sort_keys=True) if error_code is None else None
    if encoded is not None and len(encoded) > 16000: raise ValueError('Bounded result required.')
    with engine.begin() as c:
        c.exec_driver_sql('BEGIN IMMEDIATE')
        row = c.execute(text("SELECT * FROM dev_jobs WHERE id=:id AND state='running' AND lease_token_hash=:h AND lease_until>:at"),
                        {'id': job['id'], 'h': hashlib.sha256(job['token'].encode()).hexdigest(), 'at': at}).mappings().first()
        if row is None: return False
        if not _owner_active(c, row):
            state, error_code, encoded = 'cancelled', 'owner_authorization_changed', None
        elif error_code:
            state = 'dead_letter' if row['attempts'] >= row['max_attempts'] else 'retry_wait'
        else: state = 'succeeded'
        due = at+min(60, 2**row['attempts']) if state == 'retry_wait' else at
        c.execute(text('UPDATE dev_jobs SET state=:s,available_at=:due,lease_until=NULL,lease_token_hash=NULL,result_json=:r,last_error_code=:e,updated_at=:at WHERE id=:id'),
                  {'s': state, 'due': due, 'r': encoded, 'e': error_code, 'at': at, 'id': row['id']})
        _event(c, row['id'], state, row['attempts'], at); _audit(c, row, state)
        return True


def run_once(engine, tenant):
    job = claim(engine, tenant)
    if job is None: return False
    try: result = execute_job(engine, job)
    except JobUnavailable as exc:
        finish(engine, job, error_code=str(exc)); return True
    except Exception:
        finish(engine, job, error_code='worker_operation_failed'); return True
    finish(engine, job, result=result); return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url', required=True)
    parser.add_argument('--tenant', required=True)
    parser.add_argument('--max-jobs', type=int, default=1)
    parser.add_argument('--run-seconds', type=int, default=0, help='Bounded polling window0..300;not a permanent scheduler.')
    args = parser.parse_args(argv)
    if not 1 <= args.max_jobs <= 100 or not 0 <= args.run_seconds <= 300 or not 1 <= len(args.tenant) <= 100:
        parser.error('Use1..100jobs,0..300seconds and a bounded tenant.')
    from sqlalchemy.engine import make_url
    from protected_database import application_engine
    try:
        url = make_url(args.database_url)
        if url.drivername != 'sqlite' or url.username or url.password or url.host or url.query or not url.database or not Path(url.database).is_file() or Path(url.database).is_symlink():
            raise ValueError()
    except Exception:
        parser.error('Use an existing dedicated synthetic SQLite file URL without credentials,queries or symlinks.')
    engine = None
    worker = secrets.token_hex(16); processed = 0; start = time.time(); deadline = time.monotonic()+args.run_seconds
    try:
        engine = application_engine(args.database_url)
        _require_synthetic(engine)
        # Refuse unprepared databases rather than migrating under a live worker.
        with engine.begin() as c:
            c.execute(text('SELECT id FROM dev_jobs LIMIT 0'))
            c.execute(text('DELETE FROM dev_job_worker_runs WHERE heartbeat_at<:old'), {'old': start-86400})
            c.execute(text('INSERT INTO dev_job_worker_runs(worker_id,tenant,started_at,heartbeat_at) VALUES(:w,:t,:at,:at)'), {'w': worker, 't': args.tenant, 'at': start})
        while True:
            with engine.begin() as c:
                c.execute(text('UPDATE dev_job_worker_runs SET heartbeat_at=:at,processed=:p WHERE worker_id=:w'), {'at': time.time(), 'p': processed, 'w': worker})
            if run_once(engine, args.tenant): processed += 1
            if processed >= args.max_jobs or time.monotonic() >= deadline: break
            time.sleep(min(1, max(0, deadline-time.monotonic())))
        print(json.dumps({'status': 'completed', 'jobs_attempted': processed, 'bounded_worker': True,
                          'persistent_scheduler_configured': False, 'provider_requests': 0}))
        return 0
    except Exception:
        print('Worker refused or failed;check synthetic scope,schema and protected-key configuration. No key or task content disclosed.')
        return 1
    finally:
        try:
            with engine.begin() as c:
                c.execute(text('UPDATE dev_job_worker_runs SET finished_at=:at,heartbeat_at=:at,processed=:p WHERE worker_id=:w'), {'at': time.time(), 'p': processed, 'w': worker})
        except Exception: pass
        if engine is not None: engine.dispose()


if __name__ == '__main__':
    raise SystemExit(main())
