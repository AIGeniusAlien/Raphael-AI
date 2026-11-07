from sqlalchemy import text
from database import execute_sqlite_script

FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  conversation_id INTEGER,
  question TEXT,
  mode TEXT,
  model_output TEXT,
  evidence_used TEXT,
  clinician_action TEXT,     -- 'approve' | 'edit' | 'reject'
  clinician_notes TEXT,
  corrected_output TEXT
);
"""

def ensure_feedback_table(engine):
    with engine.begin() as c:
        execute_sqlite_script(c, FEEDBACK_DDL)

def insert_feedback(engine, rec: dict):
    with engine.begin() as c:
        c.execute(
            text("""
                INSERT INTO feedback(created_at, conversation_id, question, mode, model_output, evidence_used,
                                    clinician_action, clinician_notes, corrected_output)
                VALUES (:ts, :cid, :q, :m, :out, :ev, :act, :notes, :corr)
            """),
            rec
        )

def list_feedback(engine, limit: int = 50):
    with engine.begin() as c:
        rows = c.execute(text("SELECT * FROM feedback ORDER BY id DESC LIMIT :n"), {"n": limit}).mappings().all()
    return list(rows)

# Tenant/owner scoped product-improvement records are separate from legacy
# clinical feedback. Legacy rows are never retroactively assigned an owner.
PORTAL_FEEDBACK_DDL='''
CREATE TABLE IF NOT EXISTS dev_product_feedback (
 id INTEGER PRIMARY KEY AUTOINCREMENT,tenant TEXT NOT NULL,owner_id INTEGER NOT NULL REFERENCES dev_users(id),
 request_id TEXT NOT NULL,payload_hash TEXT NOT NULL,category TEXT NOT NULL,impact TEXT NOT NULL,
 title TEXT NOT NULL,detail TEXT NOT NULL,state TEXT NOT NULL CHECK(state IN ('submitted','triaged','task_created','in_progress','resolved','rejected')),priority TEXT CHECK(priority IN ('P0','P1','P2')),
 revision INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
 UNIQUE(tenant,owner_id,request_id)
);
CREATE TABLE IF NOT EXISTS dev_feedback_tasks (
 id INTEGER PRIMARY KEY AUTOINCREMENT,feedback_id INTEGER NOT NULL UNIQUE REFERENCES dev_product_feedback(id),
 requirement_id TEXT NOT NULL,assigned_to INTEGER NOT NULL REFERENCES dev_users(id),
 state TEXT NOT NULL CHECK(state IN ('planned','in_progress','done')),evidence_reference TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_feedback_actions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,feedback_id INTEGER NOT NULL REFERENCES dev_product_feedback(id),
 actor_id INTEGER NOT NULL REFERENCES dev_users(id),request_id TEXT NOT NULL,payload_hash TEXT NOT NULL,
 action TEXT NOT NULL,note TEXT NOT NULL,result_revision INTEGER NOT NULL,created_at TEXT NOT NULL,
 UNIQUE(feedback_id,request_id)
);
'''

def ensure_portal_feedback_tables(engine):
    from dev_portal import _require_synthetic
    _require_synthetic(engine)
    with engine.begin() as c:execute_sqlite_script(c,PORTAL_FEEDBACK_DDL)
