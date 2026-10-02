"""End-to-end contract tests run the actual background pipeline and database."""
import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import ROOT, Settings
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'api.db').as_posix()}",
        llm_enabled=False, web_search_provider="demo",
    )
    engine = create_async_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    with TestClient(create_app(settings, sessions)) as instance:
        yield instance


def submit(client, body: bytes, filename="procurement.json"):
    response = client.post("/runs", files={"file": (filename, body, "application/json")})
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "pending"
    return response.json()["run_id"]


def test_demo_full_pipeline_and_csv(client):
    run_id = submit(client, (ROOT / "data/sample_procurement.json").read_bytes())
    # ASGI TestClient waits for BackgroundTasks; a real HTTP client can poll processing states.
    status = client.get(f"/runs/{run_id}").json()
    assert status["status"] == "completed", status
    assert status["progress"] == 100
    assert status["mode"] == "demo"
    result = client.get(f"/runs/{run_id}/results")
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["lots"][0]["okpd2_source"] == "input"
    companies = payload["lots"][0]["companies"]
    assert len(companies) >= 3
    assert [c["rank"] for c in companies] == list(range(1, len(companies) + 1))
    assert [c["score"] for c in companies] == sorted((c["score"] for c in companies), reverse=True)
    assert any("history" in c["origins"] for c in companies)
    assert any(c["history_wins"] > 0 for c in companies)
    assert any(any(f["is_demo"] for f in c["facts"]) for c in companies)
    statuses = {check["status"] for c in companies for check in c["checks"]}
    assert statuses == {"confirmed", "contradicted", "unknown"}
    for company in companies:
        assert company["confirmed_count"] + company["contradicted_count"] + company["unknown_count"] == len(company["checks"])
        assert all(f["source_url"] and f["source_quote"] for f in company["facts"])
    export = client.get(f"/runs/{run_id}/export")
    assert export.status_code == 200
    assert export.content.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig"))))
    assert len(rows) == len(companies)
    assert rows[0]["mode"] == "demo"


def test_invalid_json_fails_run_without_breaking_api(client):
    run_id = submit(client, b"{broken")
    status = client.get(f"/runs/{run_id}").json()
    assert status["status"] == "failed"
    assert status["error"]
    assert client.get(f"/runs/{run_id}/export").status_code == 409
    assert client.get("/health").status_code == 200


def test_test_mode_hard_run_deadline_marks_timed_out_and_preserves_partial_api(client, monkeypatch):
    import asyncio
    client.app.state.settings.test_mode = True
    client.app.state.settings.max_run_seconds = 0.05

    async def slow_run(run_id):
        await asyncio.sleep(1)

    monkeypatch.setattr(client.app.state.pipeline, "_run_with_lock", slow_run)
    run_id = submit(client, b'{"lots": []}')
    status = client.get(f"/runs/{run_id}").json()
    assert status["status"] == "completed"
    assert status["stage"] == "partial_timeout"
    assert status["progress"] < 100
    assert status["error"] is None
    assert any("MAX_RUN_SECONDS=0.05" in warning for warning in status["warnings"])
    assert client.get("/health").json()["test_mode"] is True
    assert client.get("/health").json()["max_run_seconds"] == 0.05


def test_upload_validation_and_missing_runs(client):
    assert client.post("/runs", files={"file": ("x.exe", b"abc")}).status_code == 415
    assert client.post("/runs", files={"file": ("x.json", b"")}).status_code == 422
    assert client.get("/runs/999999").status_code == 404
    assert client.get("/runs/999999/results").status_code == 404
    assert client.get("/runs/999999/export").status_code == 404


def test_multiple_lots_and_repeat_runs_are_isolated(client):
    sample = json.loads((ROOT / "data/sample_procurement.json").read_text(encoding="utf-8-sig"))
    source = sample["lots"][0]
    other = dict(source, name="Второй лот", external_id="second")
    first = submit(client, json.dumps({"lots": [source, other]}, ensure_ascii=False).encode())
    second = submit(client, json.dumps({"lots": [source]}, ensure_ascii=False).encode())
    one = client.get(f"/runs/{first}/results").json()
    two = client.get(f"/runs/{second}/results").json()
    assert one["run"]["status"] == two["run"]["status"] == "completed"
    assert len(one["lots"]) == 2
    assert len(two["lots"]) == 1
    assert {lot["id"] for lot in one["lots"]}.isdisjoint({lot["id"] for lot in two["lots"]})
    assert len(one["lots"][0]["companies"]) == len(two["lots"][0]["companies"])


def test_unavailable_search_preserves_history(client, monkeypatch):
    class BrokenProvider:
        async def search(self, query):
            raise TimeoutError("simulated unavailable search")

    monkeypatch.setattr("app.agent.pipeline.create_web_search_provider", lambda settings: BrokenProvider())
    run_id = submit(client, (ROOT / "data/sample_procurement.json").read_bytes())
    result = client.get(f"/runs/{run_id}/results").json()
    assert result["run"]["status"] == "completed"
    assert result["lots"][0]["companies"]
    assert any("search request failed" in warning for warning in result["run"]["warnings"])


def test_csv_upload_and_openapi(client):
    csv_body = "name,description,okpd2\nПлиты,Промышленные электрические плиты,27.51.28.130\n"
    run_id = submit(client, csv_body.encode("utf-8-sig"), "lots.csv")
    result = client.get(f"/runs/{run_id}/results").json()
    assert result["run"]["status"] == "completed", result
    assert result["lots"][0]["requirements"] == []
    assert any("search-only" in warning for warning in result["run"]["warnings"])
    schema = client.get("/openapi.json").json()
    assert "multipart/form-data" in schema["paths"]["/runs"]["post"]["requestBody"]["content"]


def test_missing_okpd2_classification_is_visible_in_api(client):
    body = {"lots": [{"name": "Поставка промышленных электрических плит", "description": "Электрические плиты для школьного пищеблока"}]}
    run_id = submit(client, json.dumps(body, ensure_ascii=False).encode())
    result = client.get(f"/runs/{run_id}/results").json()
    assert result["run"]["status"] == "completed", result
    lot = result["lots"][0]
    assert lot["okpd2_source"] == "history"
    assert lot["okpd2_source_url"].startswith("history://")
    assert lot["okpd2_similarity"] > 0.9
