"""Server-enforced, versioned consumer-candidate boundary for synthetic previews.

This does not authorize public deployment, real PHI, clinical use, registration,
clinical AI or account-erasure compliance. Default development behavior remains
unchanged. Unknown future routes are denied in candidate scope.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse, RedirectResponse

API_VERSION = '1'
SCOPES = ('development', 'consumer_candidate')
# Paths are full matches, never prefix permissions. Existing route-level role,
# patient/tenant ownership, consent, origin, session and request budgets remain.
CANDIDATE_ROUTES = (
    ('GET', r'/api/dev/session'), ('POST', r'/api/dev/logout'),
    ('GET', r'/api/dev/patients'), ('GET', r'/api/dev/patients/[1-9][0-9]*'),
    ('GET', r'/api/dev/patients/[1-9][0-9]*/(?:entries|notes|documents)'),
    ('GET', r'/api/dev/documents/[1-9][0-9]*/content'),
    ('GET', r'/api/dev/account'), ('POST', r'/api/dev/account/profile'),
    ('GET', r'/api/dev/account/(?:storage|sessions|export|deletion-requests|mfa)'),
    ('POST', r'/api/dev/account/sessions/[a-f0-9]+/revoke'),
    ('POST', r'/api/dev/account/deletion-requests'),
    ('POST', r'/api/dev/account/deletion-requests/[1-9][0-9]*/cancel'),
    ('GET', r'/api/dev/feedback/reports'), ('POST', r'/api/dev/feedback/reports'),
    ('GET', r'/api/dev/feedback/reports/[1-9][0-9]*'),
)
PUBLIC_GET = frozenset(('/healthz', '/readyz', '/api/dev/release-policy'))
STATIC_ASSETS = frozenset(('/static/release-preview.html', '/static/release-preview.js',
                           '/static/release-preview.css'))
ACCOUNT_CONTINUITY = frozenset(('/api/dev/session', '/api/dev/logout', '/api/dev/account',
                              '/api/dev/account/sessions', '/api/dev/account/storage',
                              '/api/dev/account/mfa', '/api/dev/account/deletion-requests',
                              '/api/dev/account/profile'))
ACCOUNT_CONTROL_PATTERNS = (r'/api/dev/account/sessions/[a-f0-9]+/revoke',
                            r'/api/dev/account/deletion-requests/[1-9][0-9]*/cancel')
CAPABILITIES = {
    'own_linked_records': True, 'own_approved_notes': True,
    'own_document_download': True, 'own_account_preferences': True,
    'own_record_export': True, 'own_feedback': True,
    'registration': False, 'external_password_recovery': False,
    'clinical_assessment': False, 'clinical_generation': False,
    'radiology': False, 'ambient_recording': False, 'clinical_tools': False,
    'external_clinical_communications': False, 'administration': False,
    'complete_account_erasure': False,
}


@dataclass(frozen=True)
class ReleasePolicy:
    scope: str = 'development'
    pause_file: Path | None = None

    @classmethod
    def from_environment(cls, environment=None):
        env = os.environ if environment is None else environment
        scope = env.get('RAPHAEL_RELEASE_SCOPE', 'development')
        if scope not in SCOPES:
            raise ValueError('Unknown Raphael release scope; startup refused.')
        if scope == 'consumer_candidate' and (env.get('RAPHAEL_DEV_PORTAL') != '1' or
                                               env.get('RAPHAEL_DEV_SYNTHETIC_ONLY') != '1'):
            raise ValueError('Consumer candidate currently requires an isolated synthetic portal; public or clinical deployment is not authorized.')
        pause = env.get('RAPHAEL_RELEASE_PAUSE_FILE')
        path = None
        if pause:
            if scope != 'consumer_candidate':
                raise ValueError('Operator pause files apply only to consumer candidate scope.')
            if os.name != 'posix':
                raise ValueError('Operator pause file requires verified POSIX private-file semantics.')
            path = Path(pause).absolute()
            parent = path.parent
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError('Operator pause configuration requires an existing private directory.')
            info = parent.stat()
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError('Operator pause directory must be owner-only.')
        return cls(scope, path)

    def paused(self):
        if self.pause_file is None:
            return False
        try:
            parent = self.pause_file.parent.lstat()
            if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
                return True
        except OSError:
            return True
        try:
            self.pause_file.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        # Any presence, including symlinks or malformed files, pauses access.
        # Never read file contents or disclose its path.
        return True

    def manifest(self):
        capabilities = dict(CAPABILITIES) if self.scope == 'consumer_candidate' else None
        version = hashlib.sha256(json.dumps({'api_version': API_VERSION, 'scope': self.scope,
            'capabilities': capabilities, 'candidate_routes': CANDIDATE_ROUTES,
            'static_assets': sorted(STATIC_ASSETS), 'public_get': sorted(PUBLIC_GET),
            'account_continuity': sorted(ACCOUNT_CONTINUITY),
            'account_control_patterns': ACCOUNT_CONTROL_PATTERNS,
            'anonymous_login': ['POST', '/api/dev/login'], 'root_redirect': '/static/release-preview.html',
            'subject_role': 'patient', 'policy_schema': 'release-boundary-v2'}, sort_keys=True).encode()).hexdigest()
        return {'scope': self.scope, 'api_version': API_VERSION, 'policy_sha256': version,
                'operator_paused': self.paused(), 'capabilities': capabilities,
                'public_distribution_authorized': False, 'clinical_production_authorized': False,
                'data_scope': 'synthetic-development-only' if self.scope == 'consumer_candidate' else 'development; not public or clinical authorization',
                'notice': 'Candidate scope is a synthetic engineering preview, not a public release or clinical authorization. Provisioned patient accounts only; registration, external recovery and complete erasure remain unavailable.'}


class ReleaseBoundary:
    def __init__(self, app, policy, context):
        self.app, self.policy, self.context = app, policy, context

    async def __call__(self, scope, receive, send):
        if self.policy.scope != 'consumer_candidate' or scope['type'] not in ('http', 'websocket'):
            return await self.app(scope, receive, send)
        if scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
            return
        original_send = send
        async def secured_send(message):
            if message['type'] == 'http.response.start':
                headers = MutableHeaders(scope=message)
                headers['Cache-Control'] = 'no-store'
                headers['Permissions-Policy'] = 'microphone=(), camera=(), geolocation=()'
                headers['X-Content-Type-Options'] = 'nosniff'
                headers['Referrer-Policy'] = 'no-referrer'
                headers['X-Frame-Options'] = 'DENY'
                headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'self'"
            await original_send(message)
        send = secured_send
        request = Request(scope, receive=receive)
        path, method = scope['path'], scope['method']
        async def answer(response):
            response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY',
                'Permissions-Policy': 'microphone=(), camera=(), geolocation=()',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'self'"})
            await response(scope, receive, send)
        def reject(code=403, detail='This capability is not enabled for the consumer candidate scope.'):
            return JSONResponse({'detail': detail, 'release_scope': self.policy.scope}, status_code=code)
        # Reject ambiguous wire forms, rather than relying on decoding behavior
        # across proxies/ASGI servers. This allowlist uses ASCII canonical paths.
        raw = scope.get('raw_path', path.encode()).split(b'?', 1)[0]
        if raw != path.encode() or '\\' in path or '//' in path or '..' in path.split('/'):
            return await answer(reject())
        if path in PUBLIC_GET and method == 'GET':
            return await self.app(scope, receive, send)
        if request.headers.get('x-raphael-api-version', API_VERSION) != API_VERSION:
            return await answer(reject(426, 'Unsupported client API version; update the client.'))
        if path == '/' and method == 'GET':
            return await answer(RedirectResponse('/static/release-preview.html'))
        if path in STATIC_ASSETS and method == 'GET':
            return await self.app(scope, receive, send)
        if path == '/api/dev/login' and method == 'POST':
            return await self.app(scope, receive, send)
        if not any(method == verb and re.fullmatch(pattern, path) for verb, pattern in CANDIDATE_ROUTES):
            return await answer(reject())
        try:
            user = await run_in_threadpool(self.context.user, request)
            if user['role'] != 'patient' and not (method == 'POST' and path == '/api/dev/logout'):
                return await answer(reject(403, 'Consumer candidate access is restricted to provisioned patient accounts.'))
        except HTTPException as exc:
            return await answer(reject(exc.status_code, str(exc.detail)))
        except Exception:
            return await answer(reject(503, 'Session verification is temporarily unavailable; no records returned.'))
        continuity = path in ACCOUNT_CONTINUITY or any(re.fullmatch(pattern, path) for pattern in ACCOUNT_CONTROL_PATTERNS)
        if self.policy.paused() and not continuity:
            return await answer(reject(503, 'Record and feedback access is temporarily paused by the operator. Account and sign-out controls remain available.'))
        # Recheck immediately before releasing protected response bytes. A role,
        # tenant or session changed by an operator/another request must not turn
        # the initial patient check into a clinician/other-tenant response.
        # Sign-out and session-revoke acknowledgments contain no record data and
        # intentionally invalidate their own authorization credential.
        revocation_ack = method == 'POST' and (path == '/api/dev/logout' or
            re.fullmatch(r'/api/dev/account/sessions/[a-f0-9]+/revoke', path))
        if revocation_ack:
            return await self.app(scope, receive, send)
        expected_identity = (user['id'], user['tenant'], user['role'])
        withheld = False
        async def authorized_send(message):
            nonlocal withheld
            if withheld:
                return
            if message['type'] == 'http.response.start':
                try:
                    current = await run_in_threadpool(self.context.user, request)
                    if (current['id'], current['tenant'], current['role']) != expected_identity or current['role'] != 'patient':
                        raise HTTPException(403, 'Authorization changed; refresh before accessing records.')
                except HTTPException as exc:
                    withheld = True
                    await answer(reject(exc.status_code, 'Session or authorization changed; protected response withheld.'))
                    return
                except Exception:
                    withheld = True
                    await answer(reject(503, 'Session verification is temporarily unavailable; protected response withheld.'))
                    return
            await send(message)
        return await self.app(scope, receive, authorized_send)


def install_release_policy(app, context=None, policy=None):
    policy = policy or ReleasePolicy.from_environment()
    if policy.scope not in SCOPES:
        raise ValueError('Unknown Raphael release scope; startup refused.')
    if policy.scope == 'consumer_candidate':
        if context is None or os.getenv('RAPHAEL_DEV_PORTAL') != '1' or os.getenv('RAPHAEL_DEV_SYNTHETIC_ONLY') != '1':
            raise ValueError('Consumer candidate requires the authenticated isolated synthetic portal context.')
    if getattr(app.state, 'release_policy', None) is not None:
        raise ValueError('Release policy must be installed once before startup.')
    app.state.release_policy = policy
    @app.get('/api/dev/release-policy', include_in_schema=False)
    def release_manifest():
        return JSONResponse(policy.manifest(), headers={'Cache-Control': 'no-store'})
    app.add_middleware(ReleaseBoundary, policy=policy, context=context)
    return policy
