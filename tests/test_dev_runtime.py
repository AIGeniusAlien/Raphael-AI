import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from dev_portal import create_dev_user
from dev_runtime import backup, migrate, schema_status


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="raphael-runtime-")
        self.addCleanup(self.directory.cleanup)
        self.engine = create_engine("sqlite:///" + self.directory.name + "/synthetic.db")
        self.addCleanup(self.engine.dispose)
        flag = patch.dict(os.environ, {"RAPHAEL_DEV_SYNTHETIC_ONLY": "1"})
        flag.start()
        self.addCleanup(flag.stop)

    def test_repeat_migration_and_backup_copy_preserve_users(self):
        self.assertFalse(schema_status(self.engine)["ready"])
        migrate(self.engine)
        create_dev_user(self.engine, "synthetic", "user", "Synthetic-only-password", "patient")
        migrate(self.engine)
        self.assertTrue(schema_status(self.engine)["ready"])
        output = Path(self.directory.name) / "backup.db"
        backup(self.engine, output)
        with sqlite3.connect(output) as c:
            self.assertEqual(c.execute("SELECT username FROM dev_users").fetchall(), [("user",)])
            self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            backup(self.engine, output)
        self.assertEqual(before, output.read_bytes())

    def test_unknown_schema_version_refused_without_erasing_rows(self):
        migrate(self.engine)
        with self.engine.begin() as c:
            c.exec_driver_sql("UPDATE dev_schema_versions SET version=99 WHERE component='portal'")
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            migrate(self.engine)
        self.assertFalse(schema_status(self.engine)["ready"])
