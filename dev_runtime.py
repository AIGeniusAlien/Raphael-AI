"""Versioned local-development schema preparation and non-destructive SQLite backup.

This module intentionally supports the synthetic SQLite workflow only. It does
not advertise PostgreSQL migrations or a production backup/recovery service.
"""
import argparse
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import sqlite3

from sqlalchemy import create_engine, text

from database import execute_sqlite_script


COMPONENTS = (
    ("portal", "dev_portal", "init_portal_schema"),
    ("chart", "dev_chart", "init_chart_schema"),
    ("messaging", "dev_messaging", "init_messaging_schema"),
    ("imaging", "dev_imaging", "init_imaging_schema"),
    ("knowledge", "dev_knowledge", "init_knowledge_schema"),
    ("care", "dev_care", "init_care_schema"),
    ("intelligence", "dev_intelligence", "init_intelligence_schema"),
    ("account", "dev_account", "init_account_schema"),
    ("patient_ai", "dev_patient_ai", "init_patient_ai_schema"),
)


def migrate(engine):
    """Reconcile additive component schemas, stamping only successful steps.

    Components own their transactional upgrades. Re-running after interruption
    is safe; no records are dropped by this coordinator. Refuse unknown future
    versions instead of silently attempting a code rollback across schemas.
    """
    from dev_portal import _require_synthetic
    _require_synthetic(engine)
    with engine.begin() as c:
        execute_sqlite_script(c, """
CREATE TABLE IF NOT EXISTS dev_schema_versions (
 component TEXT PRIMARY KEY,
 version INTEGER NOT NULL,
 applied_at TEXT NOT NULL
);
""")
        for row in c.execute(text("SELECT component,version FROM dev_schema_versions")).mappings():
            if row["component"] not in {name for name, _, _ in COMPONENTS} or row["version"] != 1:
                raise RuntimeError("Unsupported development schema version; use matching code and a new backup copy.")
    for component, module, initializer in COMPONENTS:
        getattr(importlib.import_module(module), initializer)(engine)
        with engine.begin() as c:
            c.execute(text("INSERT OR IGNORE INTO dev_schema_versions(component,version,applied_at) VALUES (:c,1,:at)"),
                      {"c": component, "at": datetime.now(timezone.utc).isoformat()})


def schema_status(engine):
    with engine.connect() as c:
        exists = c.exec_driver_sql("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='dev_schema_versions')").scalar()
        if not exists:
            return {"ready": False, "components": {}}
        versions = dict(c.execute(text("SELECT component,version FROM dev_schema_versions")).all())
        ready = versions == {name: 1 for name, _, _ in COMPONENTS}
        return {"ready": ready, "components": versions}


def backup(engine, destination):
    """Make a consistent backup copy; refuse to replace any destination file."""
    from dev_portal import _require_synthetic
    _require_synthetic(engine)
    target = Path(destination)
    # Exclusive creation protects an existing database/backup from replacement.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        with engine.connect() as c, sqlite3.connect(target) as output:
            c.connection.driver_connection.backup(output)
            if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Backup integrity check failed.")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("migrate", "status", "backup"))
    parser.add_argument("--database-url", required=True, help="Dedicated synthetic SQLite URL")
    parser.add_argument("--output", help="New backup filename; never an existing database")
    args = parser.parse_args()
    engine = create_engine(args.database_url)
    try:
        from dev_portal import _require_synthetic
        _require_synthetic(engine)
        if args.command == "migrate":
            migrate(engine)
            print("Synthetic development schemas initialized; existing records preserved.")
        elif args.command == "status":
            state = schema_status(engine)
            print("ready" if state["ready"] else "migration required")
            return 0 if state["ready"] else 1
        else:
            if not args.output:
                parser.error("backup requires --output pointing to a new file")
            backup(engine, args.output)
            print("Synthetic backup copy created and integrity checked. Source unchanged.")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
