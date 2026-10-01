"""Integration checks for identity, committed checkpoints, and evidence isolation."""
import pytest
from sqlalchemy import delete, event, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Company, CompanyFact, CompanyLotResult, Lot, RequirementCheck, Run
from app.repositories import companies, results, runs
from app.schemas.domain import (
    CompanyCandidate, FactData, ParsedLot, ParsedProcurement, ParsedRequirement,
    RankedCandidate, RequirementEvaluation,
)
from app.schemas.result import RunResultsOut
from app.schemas.supplier import FactType, LLMFact, SupplierParsedData


@pytest.fixture
async def sessions():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def procurement() -> ParsedProcurement:
    return ParsedProcurement(lots=[ParsedLot(
        name="Поставка оборудования", requirements=[ParsedRequirement(
            name="Опыт от пяти лет", fact_type="experience_years", type="number", operator="gte", expected_value=5,
        )],
    )])


def ranked(value: int, suffix: str) -> list[RankedCandidate]:
    fact = FactData(
        fact_type="experience_years", value=value, source_url=f"https://supplier.example/{suffix}",
        source_title="О компании", source_quote=f"Опыт работы {value} лет", is_demo=True, source_kind="demo",
    )
    confirmed = value >= 5
    return [RankedCandidate(
        candidate=CompanyCandidate(
            name="ООО Поставщик", inn="1234567890", website="https://supplier.example",
            facts=[fact], relevance=0.8, origins=["demo"],
        ),
        checks=[RequirementEvaluation(
            requirement_index=0, name="Опыт от пяти лет", status="confirmed" if confirmed else "contradicted",
            actual_value=value, reason="Числовое сравнение", evidence=[fact],
        )],
        score=100 if confirmed else 0, confirmed_count=int(confirmed), contradicted_count=int(not confirmed),
        unknown_count=0, coverage=1, rank=1, explanation="Проверено по указанному источнику",
    )]


async def test_results_keep_evidence_separate_per_run_and_replace_idempotently(sessions):
    async with sessions() as session:
        first = await runs.create_run(session, "one.json", "payload", "demo")
        first_lot = (await runs.create_lots(session, first.id, procurement()))[0]
        assert first_lot.requirements[0].id is not None
        await results.save_lot_results(session, first.id, first_lot, ranked(8, "first"))
        await results.save_lot_results(session, first.id, first_lot, ranked(8, "first"))
        second = await runs.create_run(session, "two.json", "payload", "demo")
        second_lot = (await runs.create_lots(session, second.id, procurement()))[0]
        await results.save_lot_results(session, second.id, second_lot, ranked(3, "second"))

    # Fresh session proves the repository methods committed all checkpoints.
    async with sessions() as session:
        first_data = await results.get_run_results(session, first.id)
        second_data = await results.get_run_results(session, second.id)
        first_company = first_data["lots"][0]["companies"][0]
        second_company = second_data["lots"][0]["companies"][0]
        assert first_company["company_id"] == second_company["company_id"]
        assert [fact["value"] for fact in first_company["facts"]] == [8]
        assert [fact["value"] for fact in second_company["facts"]] == [3]
        assert first_company["checks"][0]["status"] == "confirmed"
        assert second_company["checks"][0]["status"] == "contradicted"
        assert first_company["sources"] == [{
            "url": "https://supplier.example/first", "title": "О компании", "is_demo": True,
        }]
        assert await session.scalar(select(func.count()).select_from(CompanyFact)) == 2
        assert await session.scalar(select(func.count()).select_from(RequirementCheck)) == 2
        assert await session.scalar(select(func.count()).select_from(CompanyLotResult)) == 2


async def test_known_conflicting_inns_are_never_combined(sessions):
    async with sessions() as session:
        a = await companies.upsert_company(session, CompanyCandidate(
            name="ООО Поставщик", inn="1234567890", website="https://shared.example",
        ))
        b = await companies.upsert_company(session, CompanyCandidate(
            name="Поставщик ООО", inn="0987654321", website="https://shared.example",
        ))
        a_again = await companies.upsert_company(session, CompanyCandidate(name="Новое имя", inn="1234567890"))
        assert a.id != b.id
        assert a_again.id == a.id
        assert await session.scalar(select(func.count()).select_from(Company)) == 2


async def test_restart_only_fails_unfinished_jobs(sessions):
    async with sessions() as session:
        pending = await runs.create_run(session, "pending.json", "", "demo")
        processing = await runs.create_run(session, "processing.json", "", "demo")
        completed = await runs.create_run(session, "completed.json", "", "demo")
        await runs.update_run(session, processing.id, status="processing", stage="web_search", progress=40)
        await runs.update_run(session, completed.id, status="completed", stage="completed", progress=100)
        await runs.mark_interrupted(session)
        assert (await runs.get_run(session, pending.id)).status == "failed"
        interrupted = await runs.get_run(session, processing.id)
        assert interrupted.status == "failed"
        assert interrupted.finished_at is not None
        assert interrupted.progress == 40
        assert (await runs.get_run(session, completed.id)).status == "completed"


async def test_database_cascade_and_scope_validation(sessions):
    async with sessions() as session:
        run = await runs.create_run(session, "one.json", "payload", "demo")
        lot = (await runs.create_lots(session, run.id, procurement()))[0]
        with pytest.raises(ValueError, match="does not belong"):
            await results.save_lot_results(session, run.id + 1, lot, ranked(8, "first"))
        await results.save_lot_results(session, run.id, lot, ranked(8, "first"))
        await session.execute(delete(Run).where(Run.id == run.id))
        await session.commit()
        for model in (Lot, CompanyFact, CompanyLotResult, RequirementCheck):
            assert await session.scalar(select(func.count()).select_from(model)) == 0
        assert await session.scalar(select(func.count()).select_from(Company)) == 1


async def test_result_company_identity_is_a_run_snapshot(sessions):
    async with sessions() as session:
        run = await runs.create_run(session, "snapshot.json", "", "demo")
        lot = (await runs.create_lots(session, run.id, procurement()))[0]
        data = ranked(8, "first")
        data[0].candidate.inn = None
        await results.save_lot_results(session, run.id, lot, data)
        await companies.upsert_company(session, data[0].candidate.model_copy(update={"inn": "1234567890"}))
        saved = await results.get_run_results(session, run.id)
        assert saved["lots"][0]["companies"][0]["inn"] is None


async def test_supplier_data_is_saved_as_an_isolated_result_snapshot(sessions):
    payload = SupplierParsedData(
        name="ООО Поставщик", inn="1234567890", website="https://supplier.example",
        phone="+74951234567", email="info@supplier.example", address="Москва, ул. Примерная, 1",
        facts=[LLMFact(
            type=FactType.PRODUCT, value="насосы", source_url="https://supplier.example/catalog",
            source_quote="Поставляем насосы.",
        )],
    )
    async with sessions() as session:
        first = await runs.create_run(session, "snapshot.json", "", "demo")
        first_lot = (await runs.create_lots(session, first.id, procurement()))[0]
        data = ranked(8, "first")
        data[0].candidate.supplier_data = payload
        await results.save_lot_results(session, first.id, first_lot, data)
        second = await runs.create_run(session, "demo.json", "", "demo")
        second_lot = (await runs.create_lots(session, second.id, procurement()))[0]
        await results.save_lot_results(session, second.id, second_lot, ranked(3, "second"))

    async with sessions() as session:
        saved = RunResultsOut.model_validate(await results.get_run_results(session, first.id))
        second_saved = RunResultsOut.model_validate(await results.get_run_results(session, second.id))
        company = saved.lots[0].companies[0]
        assert company.supplier_data == payload
        assert company.supplier_data.products == ["насосы"]
        assert second_saved.lots[0].companies[0].supplier_data is None
        snapshot = await session.scalar(
            select(CompanyLotResult.company_snapshot).where(CompanyLotResult.lot_id == first_lot.id)
        )
        assert snapshot["supplier_data"] == payload.model_dump(mode="json")

        # Previously persisted snapshots lack the optional field and still deserialize.
        result_row = await session.scalar(
            select(CompanyLotResult).where(CompanyLotResult.lot_id == first_lot.id)
        )
        result_row.company_snapshot = {key: value for key, value in snapshot.items() if key != "supplier_data"}
        await session.commit()
        legacy = RunResultsOut.model_validate(await results.get_run_results(session, first.id))
        assert legacy.lots[0].companies[0].supplier_data is None
