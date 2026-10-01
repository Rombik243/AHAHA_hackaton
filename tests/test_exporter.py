import csv
import io
import json

from app.schemas.result import RunResultsOut
from app.services.exporter import _safe_cell, export_results


def test_spreadsheet_formula_injection_and_plain_values():
    assert _safe_cell("=HYPERLINK(\"https://example.com\")").startswith("'")
    assert _safe_cell("  +123").startswith("'")
    assert _safe_cell("ООО Компания") == "ООО Компания"
    assert _safe_cell("0012345678") == "0012345678"


def test_supplier_metadata_survives_csv_export_with_exact_quote_and_null_compatibility():
    supplier_data = {
        "name": "Поставщик", "website": "https://supplier.example",
        "facts": [{
            "type": "product", "value": "насосы", "source_url": "https://supplier.example/catalog",
            "source_quote": 'Поставляем насосы, в том числе модель "А".\nЗаказы принимаются ежедневно.',
        }],
    }
    company = {
        "company_id": 1, "name": "Поставщик", "score": 0, "rank": 1,
        "confirmed_count": 0, "contradicted_count": 0, "unknown_count": 0, "coverage": 0,
        "explanation": "", "checks": [], "facts": [], "sources": [],
    }
    results = RunResultsOut.model_validate({
        "run": {
            "id": 1, "status": "completed", "stage": "completed", "progress": 100,
            "mode": "demo", "created_at": "2026-10-01T00:00:00Z",
        },
        "lots": [{
            "id": 1, "name": "Оборудование", "description": "", "requirements": [],
            "companies": [dict(company, supplier_data=supplier_data), dict(company, company_id=2, rank=2)],
        }],
    })
    rows = list(csv.DictReader(io.StringIO(export_results(results).decode("utf-8-sig"))))
    parsed = json.loads(rows[0]["supplier_data"])
    assert parsed["facts"] == supplier_data["facts"]
    assert parsed["products"] == ["насосы"]
    assert rows[1]["supplier_data"] == ""
