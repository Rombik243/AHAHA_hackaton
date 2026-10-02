from app.schemas.candidates import SupplierCandidate
from app.services.supplier_observations import add_observations, refresh_fields


def merge_candidate_groups(*groups: list[SupplierCandidate]) -> list[SupplierCandidate]:
    """Union by INN, preserving every distinct source/target link and source order."""
    by_inn: dict[str, SupplierCandidate] = {}
    seen_matches: dict[str, set[str]] = {}
    for group in groups:
        for candidate in group:
            if candidate.supplier_inn not in by_inn:
                by_inn[candidate.supplier_inn] = candidate.model_copy(update={"matches": []}, deep=True)
                seen_matches[candidate.supplier_inn] = set()
            combined = by_inn[candidate.supplier_inn]
            combined.supplier_name = combined.supplier_name or candidate.supplier_name
            combined.website = combined.website or candidate.website
            combined.origins = list(dict.fromkeys([*combined.origins, *candidate.origins]))
            input_keys = {record.model_dump_json() for record in combined.input_records}
            for record in candidate.input_records:
                if record.model_dump_json() not in input_keys:
                    combined.input_records.append(record)
                    input_keys.add(record.model_dump_json())
            add_observations(combined, candidate.observations)
            if candidate.enrichment and combined.enrichment is None:
                combined.enrichment = candidate.enrichment.model_copy(deep=True)
            if combined.enrichment:
                refresh_fields(combined)
                if (combined.enrichment.status in {"pending", "unavailable"}
                        and any(value.status == "source_supported" for value in combined.observations)):
                    combined.enrichment.status = "partial" if combined.enrichment.warnings else "completed"
            for match in candidate.matches:
                key = match.model_dump_json()
                if key not in seen_matches[candidate.supplier_inn]:
                    combined.matches.append(match)
                    seen_matches[candidate.supplier_inn].add(key)
    return list(by_inn.values())
