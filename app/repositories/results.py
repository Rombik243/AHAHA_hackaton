"""Persist and read complete lot results without cross-run evidence leakage."""
from collections import defaultdict

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Company, CompanyFact, CompanyLotResult, Lot, Requirement, RequirementCheck
from app.repositories.companies import upsert_company
from app.repositories.runs import get_run
from app.schemas.domain import FactData, RankedCandidate

FACT_FIELDS = tuple(FactData.model_fields)
REQUIREMENT_FIELDS = ("id", "name", "description", "fact_type", "type", "expected_value", "operator")
RESULT_FIELDS = (
    "score", "rank", "confirmed_count", "contradicted_count", "unknown_count", "coverage",
    "relevance", "origins", "explanation", "base_score", "history_bonus", "history_wins",
)


async def save_lot_results(
    session: AsyncSession, run_id: int, lot: Lot, ranked: list[RankedCandidate],
) -> None:
    """Replace this lot's results, checks, and evidence in one committed transaction."""
    if lot.run_id != run_id:
        raise ValueError("Lot does not belong to this run")
    # Read requirements explicitly: callers may pass a detached lot after a checkpoint.
    requirements = list((await session.scalars(
        select(Requirement).where(Requirement.lot_id == lot.id).order_by(Requirement.id)
    )).all())
    await session.execute(delete(RequirementCheck).where(RequirementCheck.lot_id == lot.id))
    await session.execute(delete(CompanyLotResult).where(CompanyLotResult.lot_id == lot.id))
    await session.execute(delete(CompanyFact).where(CompanyFact.run_id == run_id, CompanyFact.lot_id == lot.id))
    company_ids: set[int] = set()
    for result in ranked:
        company = await upsert_company(session, result.candidate, commit=False)
        if company.id in company_ids:
            raise ValueError("Ranked candidates must contain distinct companies")
        company_ids.add(company.id)
        session.add(CompanyLotResult(
            company_id=company.id, lot_id=lot.id,
            **result.model_dump(exclude={"candidate", "checks"}),
            relevance=result.candidate.relevance, origins=result.candidate.origins,
            company_snapshot=result.candidate.model_dump(
                mode="json", include={"name", "inn", "website", "supplier_data"},
            ),
        ))
        facts = {fact.model_dump_json(): fact for fact in result.candidate.facts}
        checked_indices: set[int] = set()
        for check in result.checks:
            if not 0 <= check.requirement_index < len(requirements):
                raise ValueError("Requirement check references an invalid requirement index")
            if check.requirement_index in checked_indices:
                raise ValueError("Duplicate requirement check")
            checked_indices.add(check.requirement_index)
            requirement = requirements[check.requirement_index]
            evidence = [fact.model_dump(mode="json") for fact in check.evidence]
            session.add(RequirementCheck(
                company_id=company.id, lot_id=lot.id, requirement_id=requirement.id,
                status=check.status, actual_value=check.actual_value, reason=check.reason, evidence=evidence,
            ))
            for fact in check.evidence:
                facts[fact.model_dump_json()] = fact
        for fact in facts.values():
            session.add(CompanyFact(
                run_id=run_id, lot_id=lot.id, company_id=company.id, **fact.model_dump(mode="json"),
            ))
    await session.commit()


async def get_run_results(session: AsyncSession, run_id: int) -> dict | None:
    run = await get_run(session, run_id)
    if run is None:
        return None
    lots = list((await session.scalars(select(Lot).where(Lot.run_id == run_id).order_by(Lot.id))).all())
    lot_ids = [lot.id for lot in lots]
    response = {
        "run": {key: getattr(run, key) for key in (
            "id", "status", "stage", "progress", "error", "warnings", "mode", "created_at", "finished_at",
        )},
        "lots": [],
    }
    if not lot_ids:
        return response
    result_rows = (await session.execute(
        select(CompanyLotResult, Company).join(Company, Company.id == CompanyLotResult.company_id)
        .where(CompanyLotResult.lot_id.in_(lot_ids))
        .order_by(CompanyLotResult.lot_id, CompanyLotResult.rank, Company.id)
    )).all()
    checks = (await session.execute(
        select(RequirementCheck, Requirement.name)
        .join(Requirement, Requirement.id == RequirementCheck.requirement_id)
        .where(RequirementCheck.lot_id.in_(lot_ids)).order_by(RequirementCheck.requirement_id)
    )).all()
    facts = (await session.scalars(
        select(CompanyFact).where(CompanyFact.run_id == run_id, CompanyFact.lot_id.in_(lot_ids))
        .order_by(CompanyFact.id)
    )).all()
    checks_by_company = defaultdict(list)
    facts_by_company = defaultdict(list)
    results_by_lot = defaultdict(list)
    for check, name in checks:
        checks_by_company[(check.lot_id, check.company_id)].append({
            "requirement_id": check.requirement_id, "name": name, "status": check.status,
            "actual_value": check.actual_value, "reason": check.reason, "evidence": check.evidence,
        })
    for fact in facts:
        facts_by_company[(fact.lot_id, fact.company_id)].append({
            key: getattr(fact, key) for key in FACT_FIELDS
        })
    for result, company in result_rows:
        key = (result.lot_id, company.id)
        company_facts = facts_by_company[key]
        sources = {}
        for fact in company_facts:
            sources[(fact["source_url"], fact["is_demo"])] = {
                "url": fact["source_url"], "title": fact["source_title"], "is_demo": fact["is_demo"],
            }
        results_by_lot[result.lot_id].append({
            "company_id": company.id, "name": company.name, "inn": company.inn, "website": company.website,
            **result.company_snapshot,
            **{field: getattr(result, field) for field in RESULT_FIELDS},
            "checks": checks_by_company[key], "facts": company_facts, "sources": list(sources.values()),
        })
    for lot in lots:
        response["lots"].append({
            **{key: getattr(lot, key) for key in (
                "id", "external_id", "name", "description", "okpd2", "quantity", "budget",
                "okpd2_source", "okpd2_similarity", "okpd2_source_url", "okpd2_source_quote",
            )},
            "requirements": [
                {key: getattr(requirement, key) for key in REQUIREMENT_FIELDS} for requirement in lot.requirements
            ],
            "companies": results_by_lot[lot.id],
        })
    return response
