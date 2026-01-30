from sqlalchemy import text

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
        for stmt in FEEDBACK_DDL.strip().split(";\n"):
            s = stmt.strip().rstrip(";")
            if s:
                c.execute(text(s))

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
