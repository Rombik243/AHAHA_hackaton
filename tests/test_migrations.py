"""Exercise the documented schema lifecycle and PostgreSQL SQL compilation."""
import os
import subprocess
import sys

from app.config import ROOT


def migration(env, *arguments):
    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments], cwd=ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        timeout=30,
    )


def test_sqlite_migrations_roundtrip_and_no_model_drift(tmp_path):
    env = dict(os.environ, DATABASE_URL=f"sqlite+aiosqlite:///{(tmp_path / 'migration.db').as_posix()}", AUTO_CREATE_TABLES="false")
    migration(env, "upgrade", "head")
    check = migration(env, "check")
    assert "No new upgrade operations detected" in check.stdout
    migration(env, "downgrade", "base")
    migration(env, "upgrade", "head")


def test_postgresql_migration_compiles_without_live_server():
    env = dict(os.environ, DATABASE_URL="postgresql+asyncpg://example:example@localhost/example")
    result = migration(env, "upgrade", "head", "--sql")
    assert "CREATE TABLE company_lot_results" in result.stdout
    assert "history_bonus" in result.stdout
    assert "okpd2_source" in result.stdout
