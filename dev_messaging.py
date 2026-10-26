"""Synthetic internal inbox, explicit draft approval, receipts and safe attachments.

No external transports are implemented or invoked. Receipt status records a
portal client's explicit acknowledgement, not external delivery or clinical care.
"""
import hashlib
import io
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from pypdf import PdfReader
from sqlalchemy import text

from database import execute_sqlite_script

LIMIT = 2 * 1024 * 1024
class AttachmentLimitMiddleware:
    """Bound multipart bytes before FastAPI parses/spools an upload."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST" or not re.fullmatch(r"/api/dev/messages/[0-9]+/attachments", scope.get("path", "")):
            return await self.app(scope, receive, send)
        cap = LIMIT + 65536  # Small bounded multipart envelope in addition to file cap.
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    length = int(value)
                except ValueError:
                    return await JSONResponse({"detail": "Invalid request length."}, 400)(scope, receive, send)
                if length < 0:
                    return await JSONResponse({"detail": "Invalid request length."}, 400)(scope, receive, send)
                if length > cap:
                    return await JSONResponse({"detail": "Attachment request exceeds its bounded 2 MiB file envelope."}, 413)(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > cap:
                return await JSONResponse({"detail": "Attachment request exceeds its bounded 2 MiB file envelope."}, 413)(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body, delivered = b"".join(chunks), False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        return await self.app(scope, replay, send)


SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_read_receipts (
 user_id INTEGER NOT NULL REFERENCES dev_users(id),
 conversation_id INTEGER NOT NULL REFERENCES dev_conversations(id),
 last_read_id INTEGER NOT NULL,
 read_at TEXT NOT NULL,
 PRIMARY KEY(user_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS dev_message_drafts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 conversation_id INTEGER NOT NULL REFERENCES dev_conversations(id),
 author_id INTEGER NOT NULL REFERENCES dev_users(id),
 content TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('draft','approved','sent')),
 source TEXT NOT NULL,
 due_at TEXT,
 created_at TEXT NOT NULL,
 approved_at TEXT,
 sent_message_id INTEGER REFERENCES dev_messages(id)
);
CREATE TABLE IF NOT EXISTS dev_message_attachments (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 message_id INTEGER NOT NULL REFERENCES dev_messages(id),
 uploader_id INTEGER NOT NULL REFERENCES dev_users(id),
 filename TEXT NOT NULL,
 media_type TEXT NOT NULL,
 byte_count INTEGER NOT NULL,
 sha256 TEXT NOT NULL,
 content BLOB NOT NULL,
 created_at TEXT NOT NULL
);
"""


def init_messaging_schema(engine):
    with engine.begin() as c:
        execute_sqlite_script(c, SCHEMA)


def now():
    return datetime.now(timezone.utc).isoformat()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DraftIn(Input):
    content: str = Field(min_length=1, max_length=20000)
    due_at: AwareDatetime | None = None
    urgency: Literal["routine"] = "routine"


class AssistIn(Input):
    instruction: str = Field(min_length=1, max_length=5000)


class ReceiptIn(Input):
    last_read_id: int = Field(gt=0)


def install_messaging(app, ctx, draft_generator=None):
    router = APIRouter(prefix="/api/dev", dependencies=[Depends(ctx.request_guard)], tags=["synthetic internal messaging"])
    app.add_middleware(AttachmentLimitMiddleware)

    def participant(user=Depends(ctx.user)):
        if user["role"] not in ("patient", "clinician"):
            raise HTTPException(403, "Patient or clinician role required.")
        return user

    def owned_draft(c, draft_id, user):
        row = c.execute(text("SELECT * FROM dev_message_drafts WHERE id=:id"), {"id": draft_id}).mappings().first()
        if not row:
            raise HTTPException(404, "Draft not found.")
        ctx.conversation_for(c, row["conversation_id"], user)
        if row["author_id"] != user["id"]:
            raise HTTPException(404, "Draft not found.")
        return dict(row)

    def message_for(c, message_id, user):
        row = c.execute(text("SELECT * FROM dev_messages WHERE id=:id"), {"id": message_id}).mappings().first()
        if not row:
            raise HTTPException(404, "Message not found.")
        ctx.conversation_for(c, row["conversation_id"], user)
        return dict(row)

    @router.get("/inbox")
    def inbox(user=Depends(participant)):
        with ctx.engine.begin() as c:
            where = "p.tenant=:tenant AND " + ("p.patient_user_id=:uid" if user["role"] == "patient" else "v.clinician_id=:uid")
            rows = c.execute(text("SELECT v.*,p.name AS patient_name, "
                "(SELECT count(*) FROM dev_messages m WHERE m.conversation_id=v.id AND m.sender_id!=:uid "
                "AND m.id>COALESCE((SELECT r.last_read_id FROM dev_read_receipts r WHERE r.user_id=:uid AND r.conversation_id=v.id),0)) AS unread_count, "
                "(SELECT count(*) FROM dev_message_drafts d WHERE d.conversation_id=v.id AND d.author_id=:uid AND d.status!='sent' "
                "AND d.due_at IS NOT NULL AND d.due_at<=:at) AS due_followups "
                "FROM dev_conversations v JOIN dev_patients p ON p.id=v.patient_id WHERE " + where + " ORDER BY v.id DESC LIMIT 200"),
                {"tenant": user["tenant"], "uid": user["id"], "at": now()}).mappings().all()
            # Thread ownership does not outlive chart authorization or revocation.
            rows=[row for row in rows if ctx.can_access_patient(c,row['patient_id'],user)]
            items = [dict(row, channel="internal_portal", external_delivery_available=False) for row in rows]
            ctx.audit(c, user, "read_inbox", "conversation")
            return {"threads": items, "unread_count": sum(row["unread_count"] for row in rows),
                    "due_followups": sum(row["due_followups"] for row in rows),
                    "channel": "internal_portal", "external_delivery_available": False,
                    "urgent_policy": "Routine synthetic tests only. Do not use this portal for emergencies or real patient care."}

    @router.get("/inbox/{conversation_id}")
    def thread(conversation_id: int, user=Depends(participant)):
        with ctx.engine.begin() as c:
            conversation = ctx.conversation_for(c, conversation_id, user)
            rows = c.execute(text("SELECT m.*,u.username AS sender_username,u.role AS sender_role, "
                "EXISTS(SELECT 1 FROM dev_read_receipts r WHERE r.conversation_id=m.conversation_id AND r.user_id!=m.sender_id AND r.last_read_id>=m.id) AS read_by_recipient "
                "FROM dev_messages m JOIN dev_users u ON u.id=m.sender_id WHERE m.conversation_id=:id ORDER BY m.id"),
                {"id": conversation_id}).mappings().all()
            messages = []
            for row in rows:
                attachments = c.execute(text("SELECT id,filename,media_type,byte_count,sha256 FROM dev_message_attachments WHERE message_id=:id ORDER BY id"), {"id": row["id"]}).mappings().all()
                messages.append(dict(row, read_by_recipient=bool(row["read_by_recipient"]), delivery_status="stored_in_portal", attachments=[dict(a) for a in attachments]))
            drafts = []
            if user["role"] == "clinician":
                drafts = [dict(row) for row in c.execute(text("SELECT * FROM dev_message_drafts WHERE conversation_id=:cid AND author_id=:uid ORDER BY id"), {"cid": conversation_id, "uid": user["id"]}).mappings()]
            ctx.audit(c, user, "read_thread", "conversation", conversation_id)
            return {"conversation": dict(conversation), "messages": messages, "drafts": drafts,
                    "channel": "internal_portal", "external_delivery_available": False}

    @router.post("/conversations/{conversation_id}/read", dependencies=[Depends(ctx.same_origin_json)])
    def read(conversation_id: int, body: ReceiptIn, user=Depends(participant)):
        with ctx.engine.begin() as c:
            ctx.conversation_for(c, conversation_id, user)
            if not c.execute(text("SELECT 1 FROM dev_messages WHERE id=:mid AND conversation_id=:cid"), {"mid": body.last_read_id, "cid": conversation_id}).scalar():
                raise HTTPException(400, "Acknowledge only a message in this conversation.")
            c.execute(text("INSERT INTO dev_read_receipts(user_id,conversation_id,last_read_id,read_at) VALUES (:uid,:cid,:mid,:at) "
                "ON CONFLICT(user_id,conversation_id) DO UPDATE SET last_read_id=MAX(last_read_id,excluded.last_read_id),read_at=excluded.read_at"),
                {"uid": user["id"], "cid": conversation_id, "mid": body.last_read_id, "at": now()})
            ctx.audit(c, user, "acknowledge_thread", "conversation", conversation_id)
            return {"status": "read_in_portal", "external_delivery_confirmed": False}

    def save_draft(c, conversation_id, content, due_at, user, source):
        result = c.execute(text("INSERT INTO dev_message_drafts(conversation_id,author_id,content,status,source,due_at,created_at) "
            "VALUES (:cid,:uid,:content,'draft',:source,:due,:at)"),
            {"cid": conversation_id, "uid": user["id"], "content": content, "source": source,
             "due": due_at.astimezone(timezone.utc).isoformat() if due_at else None, "at": now()})
        ctx.audit(c, user, "create_message_draft", "message_draft", result.lastrowid)
        return owned_draft(c, result.lastrowid, user)

    @router.post("/conversations/{conversation_id}/drafts", dependencies=[Depends(ctx.same_origin_json)])
    def create_draft(conversation_id: int, body: DraftIn, user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            ctx.conversation_for(c, conversation_id, user)
            return save_draft(c, conversation_id, body.content, body.due_at, user, "manual")

    @router.post("/conversations/{conversation_id}/draft-assist", dependencies=[Depends(ctx.same_origin_json)])
    def assist(conversation_id: int, body: AssistIn, user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            ctx.conversation_for(c, conversation_id, user)
            messages = c.execute(text("SELECT m.content,u.role FROM dev_messages m JOIN dev_users u ON u.id=m.sender_id WHERE conversation_id=:cid ORDER BY m.id"), {"cid": conversation_id}).all()
        if draft_generator is None:
            raise HTTPException(503, "AI-assisted message drafting is not configured. Manual drafts remain available.")
        try:
            content = draft_generator(context="\n".join(f"{role}: {content}" for content, role in messages), instruction=body.instruction)
        except Exception:
            raise HTTPException(503, "AI-assisted message drafting service is unavailable. No draft was sent.")
        if not isinstance(content, str) or not content.strip() or len(content) > 20000:
            raise HTTPException(502, "Draft provider returned an invalid message draft.")
        with ctx.engine.begin() as c:
            ctx.conversation_for(c, conversation_id, user)
            return save_draft(c, conversation_id, content.strip(), None, user, "configured_provider")

    @router.patch("/message-drafts/{draft_id}", dependencies=[Depends(ctx.same_origin_json)])
    def edit(draft_id: int, body: DraftIn, user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            owned_draft(c, draft_id, user)
            result = c.execute(text("UPDATE dev_message_drafts SET content=:v,due_at=:due WHERE id=:id AND status='draft'"),
                {"v": body.content, "due": body.due_at.astimezone(timezone.utc).isoformat() if body.due_at else None, "id": draft_id})
            if result.rowcount != 1:
                raise HTTPException(409, "Only unapproved drafts can be edited.")
            ctx.audit(c, user, "edit_message_draft", "message_draft", draft_id)
            return owned_draft(c, draft_id, user)

    @router.post("/message-drafts/{draft_id}/approve", dependencies=[Depends(ctx.same_origin_json)])
    def approve(draft_id: int, user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            owned_draft(c, draft_id, user)
            result = c.execute(text("UPDATE dev_message_drafts SET status='approved',approved_at=:at WHERE id=:id AND status='draft'"), {"id": draft_id, "at": now()})
            if result.rowcount != 1:
                raise HTTPException(409, "Only draft messages can be approved.")
            ctx.audit(c, user, "approve_message_draft", "message_draft", draft_id)
            return owned_draft(c, draft_id, user)

    @router.post("/message-drafts/{draft_id}/send", dependencies=[Depends(ctx.same_origin_json)])
    def send(draft_id: int, user=Depends(ctx.clinician)):
        with ctx.engine.begin() as c:
            # Read approved content only after taking the write lock. A stale
            # pre-lock draft must never become the sent message.
            c.exec_driver_sql("BEGIN IMMEDIATE")
            draft = owned_draft(c, draft_id, user)
            result = c.execute(text("UPDATE dev_message_drafts SET status='sent' WHERE id=:id AND status='approved'"), {"id": draft_id})
            if result.rowcount != 1:
                raise HTTPException(409, "Explicitly approve a draft before sending it once internally.")
            result = c.execute(text("INSERT INTO dev_messages(conversation_id,sender_id,content,created_at) VALUES (:cid,:uid,:v,:at)"),
                {"cid": draft["conversation_id"], "uid": user["id"], "v": draft["content"], "at": now()})
            c.execute(text("UPDATE dev_message_drafts SET sent_message_id=:mid WHERE id=:id"), {"mid": result.lastrowid, "id": draft_id})
            ctx.audit(c, user, "send_approved_internal_draft", "message", result.lastrowid)
            return {"message_id": result.lastrowid, "status": "stored_in_portal", "external_delivery_confirmed": False}

    async def same_origin_upload(request: Request):
        origin = request.headers.get("origin")
        supplied, expected = urlsplit(origin or str(request.url)), urlsplit(str(request.url))
        if (supplied.scheme, supplied.netloc) != (expected.scheme, expected.netloc):
            raise HTTPException(403, "Cross-origin portal mutations are forbidden.")

    @router.post("/messages/{message_id}/attachments", dependencies=[Depends(same_origin_upload)])
    async def attach(message_id: int, file: UploadFile = File(...), user=Depends(participant)):
        with ctx.engine.begin() as c:
            message = message_for(c, message_id, user)
            if message["sender_id"] != user["id"]:
                raise HTTPException(403, "Attach files only to your own messages.")
        data = await file.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise HTTPException(413, "Attachment exceeds 2 MiB.")
        if not data:
            raise HTTPException(400, "Empty attachments are not supported.")
        filename = PurePosixPath((file.filename or "attachment.txt").replace("\\", "/")).name
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix in (".txt", ".md"):
            try:
                data.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise HTTPException(400, "Text attachment must be valid UTF-8.")
            media_type = "text/plain; charset=utf-8"
        elif suffix == ".pdf":
            try:
                if not data.startswith(b"%PDF-"):
                    raise ValueError()
                pdf = PdfReader(io.BytesIO(data), strict=True)
                if pdf.is_encrypted or not 1 <= len(pdf.pages) <= 200 or pdf.trailer["/Root"].get("/OpenAction"):
                    raise ValueError()
            except Exception:
                raise HTTPException(400, "PDF attachment is invalid, encrypted, has an automatic open action, or exceeds the page limit. Downloads are untrusted; no malware scan is provided.")
            media_type = "application/pdf"
        else:
            raise HTTPException(415, "Only UTF-8 .txt/.md and validated .pdf attachments are accepted.")
        with ctx.engine.begin() as c:
            message = message_for(c, message_id, user)
            if message["sender_id"] != user["id"]:
                raise HTTPException(403, "Attach files only to your own messages.")
            result = c.execute(text("INSERT INTO dev_message_attachments(message_id,uploader_id,filename,media_type,byte_count,sha256,content,created_at) "
                "VALUES (:mid,:uid,:name,:type,:size,:hash,:content,:at)"),
                {"mid": message_id, "uid": user["id"], "name": filename[:200], "type": media_type, "size": len(data),
                 "hash": hashlib.sha256(data).hexdigest(), "content": data, "at": now()})
            ctx.audit(c, user, "attach_internal_document", "message_attachment", result.lastrowid)
            return {"id": result.lastrowid, "filename": filename[:200], "byte_count": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    @router.get("/message-attachments/{attachment_id}/content")
    def attachment(attachment_id: int, user=Depends(participant)):
        with ctx.engine.begin() as c:
            row = c.execute(text("SELECT * FROM dev_message_attachments WHERE id=:id"), {"id": attachment_id}).mappings().first()
            if not row:
                raise HTTPException(404, "Attachment not found.")
            message_for(c, row["message_id"], user)
            ctx.audit(c, user, "download_internal_attachment", "message_attachment", attachment_id)
            filename = "attachment.pdf" if row["media_type"] == "application/pdf" else "attachment.txt"
            return Response(bytes(row["content"]), media_type=row["media_type"], headers={
                "Content-Disposition": f'attachment; filename="{filename}"', "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "sandbox", "Cache-Control": "no-store"})

    app.include_router(router)
