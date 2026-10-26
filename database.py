"""SQLite schema scripts executed without splitting trigger bodies or committing early."""

import sqlite3


def execute_sqlite_script(connection, script: str) -> None:
    """Execute complete statements on the caller's SQLAlchemy connection.

    sqlite3.complete_statement understands quoted semicolons and BEGIN/END
    trigger bodies. Unlike DBAPI executescript(), this does not implicitly
    commit the caller's transaction. Validate the entire script before use.
    """
    statements = []
    pending = []
    for character in script:
        pending.append(character)
        if character == ";":
            candidate = "".join(pending)
            if sqlite3.complete_statement(candidate):
                statements.append(candidate)
                pending = []
    if "".join(pending).strip():
        raise ValueError("Incomplete SQLite schema statement (missing terminator).")
    for statement in statements:
        connection.exec_driver_sql(statement)
