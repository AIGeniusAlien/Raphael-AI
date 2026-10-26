"""Clinician-only conversations and consent-controlled synthetic documentation.

Streamed tokens are provisional drafts. Browser speech recognition is a browser
service, not diarization or an independently validated medical transcription.
"""
import json
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from sqlalchemy import text

from database import execute_sqlite_script
from dev_portal import _require_synthetic

SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_ai_threads (
 id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL, owner_id INTEGER NOT NULL,
 patient_id INTEGER, intake_id INTEGER, title TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_ai_messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id INTEGER NOT NULL,
 role TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL,
 generation_id TEXT UNIQUE, model TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_ai_generation_evidence (
 generation_id TEXT PRIMARY KEY, evidence TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_active_sessions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL,
 encounter_id INTEGER NOT NULL, owner_id INTEGER NOT NULL, consent INTEGER NOT NULL,
 state TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_transcript_segments (
 id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
 content TEXT NOT NULL, origin TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_active_notes (
 id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
 version INTEGER NOT NULL, content TEXT NOT NULL, format TEXT NOT NULL,
 origin TEXT NOT NULL, status TEXT NOT NULL, approved_by INTEGER, approved_at TEXT,
 published_note_id INTEGER,
 UNIQUE(session_id,version)
);
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def init_intelligence_schema(engine):
    _require_synthetic(engine)
    from dev_care import init_care_schema
    init_care_schema(engine)
    with engine.begin() as c:
        c.exec_driver_sql('BEGIN IMMEDIATE')
        execute_sqlite_script(c, SCHEMA)
        columns = {r[1] for r in c.exec_driver_sql('PRAGMA table_info(dev_active_notes)')}
        if 'published_note_id' not in columns:
            c.exec_driver_sql('ALTER TABLE dev_active_notes ADD COLUMN published_note_id INTEGER')
        thread_columns = {r[1] for r in c.exec_driver_sql('PRAGMA table_info(dev_ai_threads)')}
        if 'intake_id' not in thread_columns:
            c.exec_driver_sql('ALTER TABLE dev_ai_threads ADD COLUMN intake_id INTEGER')


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ThreadIn(Input):
    title: str = Field(min_length=1, max_length=200)
    patient_id: int | None = Field(default=None, gt=0)
    intake_id: int | None = Field(default=None, gt=0)


class Prompt(Input):
    content: str = Field(min_length=1, max_length=10000)


class SessionIn(Input):
    encounter_id: int = Field(gt=0)
    consent: StrictBool


class Consent(Input):
    consent: StrictBool


class Segment(Input):
    content: str = Field(min_length=1, max_length=10000)
    origin: Literal['browser_speech', 'manual_synthetic']


class NoteIn(Input):
    expected_version: int = Field(ge=0)
    format: Literal['SOAP', 'progress'] = 'SOAP'
    content: str | None = Field(default=None, min_length=1, max_length=50000)


class Version(Input):
    expected_version: int = Field(gt=0)


def openai_tokens(messages, model):
    """Actual provider SSE via SDK. Never yields manufactured runtime tokens."""
    from openai import OpenAI
    base, key = os.getenv('LLM_BASE_URL', ''), os.getenv('LLM_API_KEY', '')
    if not base or not key or key == 'EMPTY':
        raise RuntimeError('Provider not configured')
    destination = urlsplit(base)
    local_fixture = (destination.scheme == 'http' and destination.hostname in ('127.0.0.1', 'localhost', '::1')
                     and os.getenv('RAPHAEL_ALLOW_LOCAL_LLM') == '1')
    if (destination.scheme != 'https' and not local_fixture) or not destination.hostname or destination.username or destination.password or destination.query or destination.fragment:
        raise RuntimeError('Provider requires HTTPS; loopback HTTP needs an explicit development opt-in.')
    with OpenAI(base_url=base, api_key=key, timeout=30, max_retries=0) as client:
        with client.chat.completions.create(model=model, messages=messages, stream=True, max_completion_tokens=2000) as stream:
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content


def install_intelligence(app, ctx, token_provider=None, assessment_generator=None):
    router = APIRouter(prefix='/api/dev/intelligence', dependencies=[Depends(ctx.same_origin_json)])
    engine = ctx.engine

    def thread_for(c, identifier, user):
        row = c.execute(text('SELECT * FROM dev_ai_threads WHERE id=:id AND tenant=:t AND owner_id=:u'),
                        {'id': identifier, 't': user['tenant'], 'u': user['id']}).mappings().first()
        if not row:
            raise HTTPException(404, 'Conversation not found.')
        if row['patient_id']:
            ctx.patient_for(c, row['patient_id'], user)
        if row['intake_id']:
            from dev_care import authorized_intake
            authorized_intake(c, row['intake_id'], user, require_ai=True)
        return dict(row)

    def session_for(c, identifier, user):
        row = c.execute(text('SELECT * FROM dev_active_sessions WHERE id=:id AND owner_id=:u'),
                        {'id': identifier, 'u': user['id']}).mappings().first()
        if not row:
            raise HTTPException(404, 'Documentation session not found.')
        ctx.patient_for(c, row['patient_id'], user)
        return dict(row)

    @router.get('/config')
    def config(user=Depends(ctx.clinician)):
        return {'model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
                'provider_configured': bool(os.getenv('LLM_BASE_URL') and os.getenv('LLM_API_KEY') not in (None, '', 'EMPTY')),
                'speech': 'browser_capability_required', 'output': 'provisional_clinician_draft'}

    @router.get('/threads')
    def threads(q: str = '', user=Depends(ctx.clinician)):
        if len(q) > 200:
            raise HTTPException(422, 'Search is too long.')
        with engine.begin() as c:
            return [dict(r) for r in c.execute(text("SELECT * FROM dev_ai_threads WHERE tenant=:t AND owner_id=:u AND title LIKE :q AND (intake_id IS NULL OR EXISTS(SELECT 1 FROM dev_care_intakes i WHERE i.id=intake_id AND i.tenant=:t AND i.recipient_id=:u AND i.state='AUTHORIZED' AND i.allow_ai=1 AND i.expires_at>:at)) ORDER BY id DESC"),
                    {'t': user['tenant'], 'u': user['id'], 'q': '%' + q + '%', 'at': now()}).mappings()]

    @router.post('/threads')
    def create_thread(body: ThreadIn, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            if body.intake_id:
                if body.patient_id:
                    raise HTTPException(422, 'An intake thread may use only its consented snapshot, not a separate chart.')
                from dev_care import authorized_intake
                authorized_intake(c, body.intake_id, user, require_ai=True)
            if body.patient_id:
                ctx.patient_for(c, body.patient_id, user)
            identifier = c.execute(text('INSERT INTO dev_ai_threads(tenant,owner_id,patient_id,intake_id,title,created_at) VALUES (:t,:u,:p,:i,:title,:at)'),
                    {'t': user['tenant'], 'u': user['id'], 'p': body.patient_id, 'i': body.intake_id, 'title': body.title, 'at': now()}).lastrowid
            ctx.audit(c, user, 'create_ai_thread', 'ai_thread', identifier)
            return thread_for(c, identifier, user)

    @router.get('/threads/{identifier}')
    def thread(identifier: int, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            result = thread_for(c, identifier, user)
            result['messages'] = [dict(r) for r in c.execute(text('SELECT * FROM dev_ai_messages WHERE thread_id=:id ORDER BY id'), {'id': identifier}).mappings()]
            for message in result['messages']:
                stored = c.execute(text('SELECT evidence FROM dev_ai_generation_evidence WHERE generation_id=:g'), {'g': message['generation_id']}).scalar() if message['generation_id'] else None
                message['evidence'] = json.loads(stored) if stored else []
            return result

    @router.post('/threads/{identifier}/stream')
    def stream(identifier: int, body: Prompt, user=Depends(ctx.clinician)):
        if token_provider is None and not (os.getenv('LLM_BASE_URL') and os.getenv('LLM_API_KEY') not in (None, '', 'EMPTY')):
            raise HTTPException(503, 'Configure a dedicated development LLM provider to stream responses.')
        from dev_knowledge import search_evidence
        generation = secrets.token_urlsafe(24)
        model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            current_thread = thread_for(c, identifier, user)
            if c.execute(text("SELECT 1 FROM dev_ai_messages WHERE thread_id=:id AND status='streaming'"), {'id': identifier}).first():
                raise HTTPException(409, 'Stop or finish the current generation first.')
            history = [dict(r) for r in c.execute(text("SELECT role,content FROM dev_ai_messages WHERE thread_id=:id AND status='complete' ORDER BY id DESC LIMIT 20"), {'id': identifier}).mappings()][::-1]
            c.execute(text("INSERT INTO dev_ai_messages(thread_id,role,content,status,created_at) VALUES (:id,'user',:v,'complete',:at)"), {'id': identifier, 'v': body.content, 'at': now()})
            c.execute(text("INSERT INTO dev_ai_messages(thread_id,role,content,status,generation_id,model,created_at) VALUES (:id,'assistant','','streaming',:g,:m,:at)"), {'id': identifier, 'g': generation, 'm': model, 'at': now()})
            ctx.audit(c, user, 'start_ai_generation', 'ai_thread', identifier)
            context = []
            if current_thread['patient_id']:
                context = [dict(r) for r in c.execute(text('SELECT kind,label,data,recorded_at,status FROM dev_chart_entries WHERE patient_id=:p ORDER BY id DESC LIMIT 20'), {'p': current_thread['patient_id']}).mappings()]
            elif current_thread['intake_id']:
                from dev_care import authorized_intake
                intake = authorized_intake(c, current_thread['intake_id'], user, require_ai=True)
                context = {'consented_snapshot_only': intake['snapshot'], 'reported_symptoms': intake['symptoms'], 'purpose': intake['purpose']}
        try:
            evidence = search_evidence(engine, user['tenant'], body.content[:1000])
            with engine.begin() as c:
                c.execute(text('INSERT INTO dev_ai_generation_evidence(generation_id,evidence) VALUES (:g,:v)'), {'g': generation, 'v': json.dumps(evidence)})
        except Exception:
            with engine.begin() as c:
                c.execute(text("UPDATE dev_ai_messages SET status='failed' WHERE generation_id=:g"), {'g': generation})
                ctx.audit(c, user, 'ai_retrieval_failed', 'ai_thread', identifier)
            raise HTTPException(503, 'Reviewed evidence retrieval is unavailable. No provider request was made.')
        messages = [{'role': 'system', 'content': 'You are Raphael clinical decision support for synthetic development. Output is a provisional draft for clinician review. Distinguish evidence, inference, uncertainty and missing information. Never invent references. Use only the supplied excerpts for medical claims, cite exact IDs as [chunk_id]. No orders or external messages. If evidence is insufficient, abstain. Authorized synthetic chart entries (reported data, not verified findings): ' + json.dumps(context)[:20000] + '\nEVIDENCE: ' + json.dumps(evidence)}] + history + [{'role': 'user', 'content': body.content}]

        def events():
            content, status = '', 'failed'
            source = None
            def event(kind, value):
                return 'event: ' + kind + '\ndata: ' + json.dumps(value) + '\n\n'
            try:
                yield event('start', {'generation_id': generation, 'model': model, 'provisional': True})
                yield event('evidence', evidence)
                if not evidence:
                    status = 'abstained'
                    yield event('abstained', {'detail': 'No reviewed evidence retrieved. No model request was made.'})
                    return
                # Retrieval can take time: revoke/disable must prevent the initial
                # provider transmission too, not just subsequent streamed deltas.
                with engine.connect() as c:
                    thread_for(c, identifier, user)
                    if not c.execute(text('SELECT active FROM dev_users WHERE id=:u'), {'u': user['id']}).scalar():
                        raise HTTPException(403, 'Account is inactive.')
                source = iter((token_provider or openai_tokens)(messages, model))
                for delta in source:
                    with engine.connect() as c:
                        active = c.execute(text('SELECT status FROM dev_ai_messages WHERE generation_id=:g'), {'g': generation}).scalar()
                        account = c.execute(text('SELECT active FROM dev_users WHERE id=:u'), {'u': user['id']}).scalar()
                        if current_thread['intake_id']:
                            from dev_care import authorized_intake
                            try:
                                authorized_intake(c, current_thread['intake_id'], user, require_ai=True)
                            except HTTPException:
                                account = False
                    if active != 'streaming' or not account:
                        status = 'cancelled'
                        yield event('cancelled', {'generation_id': generation})
                        return
                    if not isinstance(delta, str) or len(content) + len(delta) > 50000:
                        raise ValueError('Invalid provider output')
                    content += delta
                    yield event('delta', {'text': delta})
                citations = re.findall(r'\[([^\[\]]+)\]', content)
                allowed = {e['chunk_id'] for e in evidence}
                if not citations or any(x not in allowed for x in citations):
                    raise ValueError('Unsupported citation membership')
                status = 'complete'
                yield event('complete', {'generation_id': generation, 'review_required': True, 'citation_check': 'retrieved_id_membership_only'})
            except GeneratorExit:
                status = 'cancelled'
                raise
            except Exception:
                yield event('error', {'detail': 'Generation failed or citation membership was invalid. Provisional text is not approved.'})
            finally:
                if source is not None and hasattr(source, 'close'):
                    try:
                        source.close()
                    except Exception:
                        status = 'failed'
                with engine.begin() as c:
                    c.execute(text("UPDATE dev_ai_messages SET content=:v,status=CASE WHEN status='cancelled' THEN 'cancelled' ELSE :s END WHERE generation_id=:g AND status IN ('streaming','cancelled')"), {'v': content, 's': status, 'g': generation})
                    ctx.audit(c, user, 'ai_generation_' + status, 'ai_thread', identifier)
        return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @router.post('/generations/{generation}/cancel')
    def cancel(generation: str, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            row = c.execute(text('SELECT thread_id FROM dev_ai_messages WHERE generation_id=:g'), {'g': generation}).first()
            if not row:
                raise HTTPException(404, 'Generation not found.')
            thread_for(c, row[0], user)
            c.execute(text("UPDATE dev_ai_messages SET status='cancelled' WHERE generation_id=:g AND status='streaming'"), {'g': generation})
            ctx.audit(c, user, 'cancel_ai_generation', 'ai_thread', row[0])
        return {'status': 'cancelled'}

    @router.post('/threads/{identifier}/assessment')
    def structured_assessment(identifier: int, body: Prompt, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            record = thread_for(c, identifier, user)
            history = [dict(r) for r in c.execute(text("SELECT role,content FROM dev_ai_messages WHERE thread_id=:id AND status='complete' ORDER BY id DESC LIMIT 20"), {'id': identifier}).mappings()][::-1]
            chart = []
            if record['patient_id']:
                chart = [dict(r) for r in c.execute(text('SELECT kind,label,data,status FROM dev_chart_entries WHERE patient_id=:p ORDER BY id DESC LIMIT 20'), {'p': record['patient_id']}).mappings()]
            elif record['intake_id']:
                from dev_care import authorized_intake
                intake = authorized_intake(c, record['intake_id'], user, require_ai=True)
                chart = {'consented_snapshot_only': intake['snapshot'], 'reported_symptoms': intake['symptoms']}
            ctx.audit(c, user, 'request_structured_assessment', 'ai_thread', identifier)
        if assessment_generator is None:
            raise HTTPException(503, 'Structured assessment service is not configured.')
        result = assessment_generator(case_context=json.dumps({'reported_chart': chart, 'history': history})[:50000], question=body.content, mode='general', tenant=user['tenant'])
        from rag.assessment_schema import Assessment
        from pydantic import ValidationError
        try:
            result = Assessment.model_validate(result).model_dump()
        except ValidationError:
            raise HTTPException(502, 'Assessment response failed schema validation.')
        with engine.begin() as c:
            thread_for(c, identifier, user)
            c.execute(text("INSERT INTO dev_ai_messages(thread_id,role,content,status,model,created_at) VALUES (:id,'assistant',:v,'complete',:m,:at)"), {'id': identifier, 'v': json.dumps(result), 'm': os.getenv('LLM_MODEL', 'gpt-4o-mini'), 'at': now()})
            ctx.audit(c, user, 'store_structured_assessment', 'ai_thread', identifier)
        return result

    @router.post('/patients/{patient_id}/sessions')
    def create_session(patient_id: int, body: SessionIn, user=Depends(ctx.clinician)):
        if not body.consent:
            raise HTTPException(403, 'Explicit recording consent is required.')
        with engine.begin() as c:
            ctx.patient_for(c, patient_id, user)
            row = c.execute(text('SELECT patient_id,status FROM dev_chart_encounters WHERE id=:id'), {'id': body.encounter_id}).first()
            if not row or row[0] != patient_id:
                raise HTTPException(404, 'Encounter not found.')
            if row[1] != 'open':
                raise HTTPException(409, 'Open the encounter before recording.')
            identifier = c.execute(text("INSERT INTO dev_active_sessions(patient_id,encounter_id,owner_id,consent,state,created_at) VALUES (:p,:e,:u,1,'active',:at)"), {'p': patient_id, 'e': body.encounter_id, 'u': user['id'], 'at': now()}).lastrowid
            ctx.audit(c, user, 'recording_consent_granted', 'active_session', identifier)
            return session_for(c, identifier, user)

    @router.get('/sessions/{identifier}')
    def session(identifier: int, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            result = session_for(c, identifier, user)
            result['segments'] = [dict(r) for r in c.execute(text('SELECT * FROM dev_transcript_segments WHERE session_id=:id ORDER BY id'), {'id': identifier}).mappings()]
            result['notes'] = [dict(r) for r in c.execute(text('SELECT * FROM dev_active_notes WHERE session_id=:id ORDER BY version'), {'id': identifier}).mappings()]
            return result

    @router.post('/sessions/{identifier}/consent')
    def consent(identifier: int, body: Consent, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            session_for(c, identifier, user)
            c.execute(text("UPDATE dev_active_sessions SET consent=:v,state=:s WHERE id=:id"), {'v': int(body.consent), 's': 'active' if body.consent else 'stopped', 'id': identifier})
            ctx.audit(c, user, 'recording_consent_granted' if body.consent else 'recording_consent_revoked', 'active_session', identifier)
        return {'consent': body.consent}

    @router.post('/sessions/{identifier}/segments')
    def segment(identifier: int, body: Segment, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            record = session_for(c, identifier, user)
            if not record['consent'] or record['state'] != 'active':
                raise HTTPException(403, 'Recording consent is not active.')
            count = c.execute(text('SELECT count(*) FROM dev_transcript_segments WHERE session_id=:id'), {'id': identifier}).scalar()
            if count >= 500:
                raise HTTPException(413, 'Documentation session limit reached.')
            c.execute(text('INSERT INTO dev_transcript_segments(session_id,content,origin,created_at) VALUES (:id,:v,:o,:at)'), {'id': identifier, 'v': body.content, 'o': body.origin, 'at': now()})
            ctx.audit(c, user, 'append_transcript', 'active_session', identifier)
        return {'stored': True, 'origin': body.origin}

    @router.post('/sessions/{identifier}/notes')
    def note(identifier: int, body: NoteIn, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            session_for(c, identifier, user)
            previous = c.execute(text('SELECT * FROM dev_active_notes WHERE session_id=:id ORDER BY version DESC LIMIT 1'), {'id': identifier}).mappings().first()
            version = previous['version'] if previous else 0
            if body.expected_version != version:
                raise HTTPException(409, 'Note changed; reload before editing.')
            if previous and previous['status'] == 'approved':
                raise HTTPException(409, 'Approved note is immutable; start a new documentation session.')
            if body.content is None and previous and previous['origin'] == 'clinician_edit':
                raise HTTPException(409, 'Preserve clinician edits. Explicitly edit or merge the newer transcript.')
            transcript = '\n'.join(c.execute(text('SELECT content FROM dev_transcript_segments WHERE session_id=:id ORDER BY id'), {'id': identifier}).scalars())
            content = body.content
            if content is None:
                if not transcript:
                    raise HTTPException(400, 'A transcript is required.')
                header = 'SYNTHETIC TRANSCRIPT TEMPLATE — NOT AI INFERENCE. Clinician review required.\n'
                content = header + ('S — Reported transcript:\n' + transcript + '\nO — Not independently observed.\nA — Not assessed.\nP — Pending clinician review.' if body.format == 'SOAP' else 'Reported interval history:\n' + transcript + '\nAssessment and plan: Pending clinician review.')
            if len(content) > 50000:
                raise HTTPException(413, 'Draft note is too long.')
            c.execute(text("INSERT INTO dev_active_notes(session_id,version,content,format,origin,status) VALUES (:id,:v,:c,:f,:o,'draft')"), {'id': identifier, 'v': version + 1, 'c': content, 'f': body.format, 'o': 'clinician_edit' if body.content is not None else 'transcript_template'})
            ctx.audit(c, user, 'version_note_draft', 'active_session', identifier)
        return {'version': version + 1, 'content': content, 'status': 'draft'}

    @router.post('/sessions/{identifier}/approve')
    def approve(identifier: int, body: Version, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            record = session_for(c, identifier, user)
            current = c.execute(text('SELECT * FROM dev_active_notes WHERE session_id=:id ORDER BY version DESC LIMIT 1'), {'id': identifier}).mappings().first()
            if not current or current['version'] != body.expected_version or current['status'] != 'draft':
                raise HTTPException(409, 'Only the current draft can be approved.')
            at = now()
            published = c.execute(text("INSERT INTO dev_notes(patient_id,author_id,content,status,created_at,approved_by,approved_at) VALUES (:p,:u,:v,'approved',:at,:u,:at)"), {'p': record['patient_id'], 'u': user['id'], 'v': current['content'], 'at': at}).lastrowid
            c.execute(text("UPDATE dev_active_notes SET status='approved',approved_by=:u,approved_at=:at,published_note_id=:p WHERE id=:id"), {'u': user['id'], 'at': at, 'id': current['id'], 'p': published})
            ctx.audit(c, user, 'approve_active_note', 'active_session', identifier)
            ctx.audit(c, user, 'publish_approved_encounter_note', 'note', published)
        return {'status': 'approved', 'version': body.expected_version}

    @router.get('/sessions/{identifier}/audit')
    def audit(identifier: int, user=Depends(ctx.clinician)):
        with engine.begin() as c:
            session_for(c, identifier, user)
            return [dict(r) for r in c.execute(text("SELECT event,created_at FROM dev_audit WHERE tenant=:t AND actor_id=:u AND resource_type='active_session' AND resource_id=:id ORDER BY id"), {'t': user['tenant'], 'u': user['id'], 'id': identifier}).mappings()]

    app.include_router(router)
