"""Explicit offline corpus + REAL Ollama; never a live search or ML ground truth.

python scripts/evaluate_offer_relevance.py --cases tests/fixtures/offer_relevance_cases.json --output tmp/offer_eval.json
"""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent.ollama_client import OllamaClient
from app.agent.offer_prompts import DECIDE_SYSTEM, EXTRACT_SYSTEM
from app.config import Settings
from app.schemas.offers import ModelDecision, OfferExtraction, TargetItem
from app.services.fact_extractor import FetchedPage
from app.services.offer_pages import archive_page
from app.services.offer_validation import evidence_valid, validate_decision, validate_offer


def metrics(rows):
    errors = sum(bool(r.get("error")) for r in rows)
    false_inclusions = sum(r["predicted"] == "relevant" and r["expected"] != "relevant" for r in rows)
    missed = sum(r["predicted"] != "relevant" and r["expected"] == "relevant" for r in rows)
    fields = {}
    for key in ("supplier_inn", "product_name", "observed_okpd2_codes", "geography"):
        available = [r for r in rows if key in r.get("predicted_fields", {})]
        filled = sum(bool(r["predicted_fields"][key]) for r in available)
        correct = sum(r["predicted_fields"][key] == r["expected_fields"][key] for r in available)
        fields[key] = {"evaluated": len(available), "filled": filled, "exactly_correct": correct,
                       "fill_rate": filled / len(rows) if rows else None,
                       "exact_accuracy": correct / len(rows) if rows else None}
    return {"cases": len(rows), "errors": errors, "correct_status": sum(r["predicted"] == r["expected"] for r in rows),
            "false_inclusions": false_inclusions, "missed_relevant": missed,
            "confusion": dict(Counter(f'{r["expected"]}->{r["predicted"]}' for r in rows)), "fields": fields}


async def evaluate(args):
    source = json.loads(args.cases.read_text(encoding="utf-8"))
    selected = source["cases"][:args.limit] if args.limit else source["cases"]
    settings = Settings(llm_enabled=True, ollama_retries=0, ollama_timeout=args.timeout,
                        page_archive_path=args.output.parent / "eval_pages")
    client = OllamaClient(settings)
    digest = await client.model_digest()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in selected:
        started = time.monotonic()
        url = f'https://offline.example/{case["id"]}'
        page, fragments = archive_page(FetchedPage(url, "Fictional evaluation fixture", case["text"]), url, settings)
        target = TargetItem(target_lot_id=case["id"], target_item_id=case["id"] + "-item",
            target_product_name=case["product"], target_subject=case["product"], target_offer_type=case["offer_type"],
            target_okpd2_code=case["okpd2"], target_delivery_region=case["region"], region_origin="request",
            target_attributes=case["attributes"])
        row = {"id": case["id"], "expected": case["expected"], "expected_fields": case["expected_fields"],
               "predicted": "processing_error", "decisions": []}
        try:
            response = await client.structured(json.dumps({"focus_chunk_id": next(iter(fragments)),
                "fragments": list(fragments.values()), "inns": [i.model_dump() for i in page.inns]}, ensure_ascii=False),
                OfferExtraction, system=EXTRACT_SYSTEM, context_tokens=settings.relevance_context_tokens)
            row["raw_extraction"] = response.model_dump(mode="json")
            raw_proofs = [e for o in response.offers for e in [*o.evidence, *o.company_link_evidence,
                          *(e for f in [*o.facts, *o.attributes, *o.observed_okpd2_codes] for e in f.evidence)]]
            row["quote_exact_count"] = sum(evidence_valid(e, fragments) for e in raw_proofs)
            row["quote_total"] = len(raw_proofs)
            offers = [validate_offer(o, page, fragments, page.inns) for o in response.offers]
            row["offers"] = [o.model_dump(mode="json") for o in offers]
            for offer in offers:
                raw = await client.structured(json.dumps({"target": target.model_dump(), "offer": offer.model_dump(mode="json"),
                    "fragments": list(fragments.values())}, ensure_ascii=False), ModelDecision, system=DECIDE_SYSTEM,
                    context_tokens=settings.relevance_context_tokens)
                row["decisions"].append(validate_decision(raw, target, offer, fragments, settings, digest).model_dump(mode="json"))
            statuses = {d["relevance_status"] for d in row["decisions"]}
            row["predicted"] = "relevant" if "relevant" in statuses else ("irrelevant" if statuses == {"irrelevant"} else "needs_review")
            row["predicted_fields"] = {"supplier_inn": offers[0].supplier_inn if offers else None,
                "product_name": offers[0].product_name if offers else None,
                "observed_okpd2_codes": list(dict.fromkeys(f.value for o in offers for f in o.observed_okpd2_codes)),
                "geography": list(dict.fromkeys(f.value for o in offers for f in o.facts if f.type in {"delivery_region", "service_region"}))}
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        row["seconds"] = round(time.monotonic() - started, 2)
        rows.append(row)
        report = {"mode": "offline_corpus_real_ollama", "is_live_search": False, "is_ml_ground_truth": False,
                  "corpus_description": source["description"], "model_name": settings.ollama_model,
                  "model_digest": digest, "metrics": metrics(rows), "rows": rows}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: row[k] for k in ("id", "expected", "predicted", "seconds")}, ensure_ascii=False), flush=True)
    print(json.dumps(metrics(rows), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    asyncio.run(evaluate(parser.parse_args()))
