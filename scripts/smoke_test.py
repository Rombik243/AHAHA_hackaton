"""Run against a started demo server: python scripts/smoke_test.py [base-url]."""
import csv
import io
import sys
import time
from pathlib import Path

import httpx


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    source = Path(__file__).resolve().parents[1] / "data/sample_procurement.json"
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
        assert client.get("/health").json()["web_search_provider"] == "demo", "Smoke test expects demo mode"
        response = client.post("/runs", files={"file": (source.name, source.read_bytes(), "application/json")})
        response.raise_for_status()
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 60
        observed = [response.json()["status"]]
        while True:
            run = client.get(f"/runs/{run_id}").json()
            if run["status"] != observed[-1]:
                observed.append(run["status"])
            if run["status"] in {"completed", "failed"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Run {run_id} did not complete")
            time.sleep(0.1)
        assert run["status"] == "completed", run
        results = client.get(f"/runs/{run_id}/results")
        results.raise_for_status()
        payload = results.json()
        companies = payload["lots"][0]["companies"]
        assert companies and all(f["source_url"] and f["source_quote"] for c in companies for f in c["facts"])
        export = client.get(f"/runs/{run_id}/export")
        export.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig"))))
        assert len(rows) == len(companies)
        assert client.get("/docs").status_code == 200
        print(f"Run {run_id}: {' -> '.join(observed)}")
        print(f"Lots: {len(payload['lots'])}; companies: {len(companies)}; CSV rows: {len(rows)}")
        print(f"Top score: {companies[0]['score']}; coverage: {companies[0]['coverage']}")
        print(f"Historical winners: {sum(c['history_wins'] > 0 for c in companies)}")
        print(f"Results: {base_url}/runs/{run_id}/results")


if __name__ == "__main__":
    main()
