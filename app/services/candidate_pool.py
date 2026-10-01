from app.schemas.candidates import SupplierCandidate


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
            for match in candidate.matches:
                key = match.model_dump_json()
                if key not in seen_matches[candidate.supplier_inn]:
                    combined.matches.append(match)
                    seen_matches[candidate.supplier_inn].add(key)
    return list(by_inn.values())
