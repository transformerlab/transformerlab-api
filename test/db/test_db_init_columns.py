import pytest
import transformerlab.db as db


# ---------- tiny async fakes ---------- #
class _FakeConn:
    """aiosqlite connection stub (await-able)."""

    def __init__(self, raise_on_alter=False):
        self._raise = raise_on_alter

    async def execute(self, sql, *args):
        if "ALTER TABLE workflows" in sql and self._raise:
            raise RuntimeError("some unexpected error")
        return None

    async def commit(self):
        return None

    async def close(self):
        return None


class _FakeEngineCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run_sync(self, _):
        return None


class _FakeAsyncEngine:
    # NOTE: this is a **sync** function returning an async-context-manager,
    # matching SQLAlchemy's real behaviour.  This fixes the TypeError.
    def begin(self):
        return _FakeEngineCtx()


class _FakeAsyncSession:
    async def execute(self, query):
        return _FakeResult()
    
    async def commit(self):
        pass
    
    def add(self, obj):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeResult:
    def scalar_one_or_none(self):
        return None


# ---------- test ---------- #
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raise_on_alter, expected_snippet",
    [
        (False, "✅ Added trigger_configs column to workflows table"),
        (True,  "⚠️  Note about trigger_configs column"),
    ],
)
async def test_db_init_alter_table_prints(monkeypatch, raise_on_alter, expected_snippet):
    """Run db.init() under two scenarios and ensure the correct print appears."""
    # aiosqlite.connect is *awaited* inside db.init(), so we supply an async stub
    async def _fake_connect(*_a, **_kw):
        return _FakeConn(raise_on_alter)

    monkeypatch.setattr("transformerlab.db.aiosqlite.connect", _fake_connect, raising=True)
    monkeypatch.setattr("transformerlab.db.async_engine", _FakeAsyncEngine(), raising=True)
    
    # Mock the async_session to avoid SQLAlchemy complexity
    def _fake_async_session():
        return _FakeAsyncSession()
    
    monkeypatch.setattr("transformerlab.db.async_session", _fake_async_session, raising=True)
    
    # Mock job_cancel_in_progress_jobs to avoid more database operations
    async def _fake_job_cancel():
        pass
    
    monkeypatch.setattr("transformerlab.db.job_cancel_in_progress_jobs", _fake_job_cancel, raising=True)

    captured = []

    def _capture_print(*args, **_kw):
        captured.append(" ".join(map(str, args)))

    monkeypatch.setattr("builtins.print", _capture_print, raising=True)

    # run
    await db.init()

    assert any(expected_snippet in line for line in captured), captured
