"""Run persistence. Each mutation is an explicit committed pipeline checkpoint."""
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Lot, Requirement, Run
from app.models.run import utc_now
from app.schemas.domain import ParsedProcurement


async def create_run(session: AsyncSession, filename: str, content: str, mode: str) -> Run:
    run = Run(filename=filename, input_content=content, mode=mode)
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run


async def get_run(session: AsyncSession, run_id: int) -> Run | None:
    return await session.get(Run, run_id, populate_existing=True)


async def update_run(session: AsyncSession, run_id: int, **fields) -> None:
    allowed = {"status", "stage", "progress", "error", "warnings", "finished_at"}
    if invalid := fields.keys() - allowed:
        raise ValueError(f"Unsupported run fields: {sorted(invalid)}")
    if fields:
        await session.execute(update(Run).where(Run.id == run_id).values(**fields))
        await session.commit()


async def mark_interrupted(session: AsyncSession) -> None:
    """An in-process MVP cannot resume jobs lost during a server restart."""
    await session.execute(
        update(Run).where(Run.status.in_(["pending", "processing"])).values(
            status="failed", stage="interrupted", error="Processing interrupted by server restart; submit again.",
            finished_at=utc_now(),
        )
    )
    await session.commit()


async def create_lots(session: AsyncSession, run_id: int, procurement: ParsedProcurement) -> list[Lot]:
    existing = await session.scalar(select(Lot.id).where(Lot.run_id == run_id).limit(1))
    if existing is not None:
        raise ValueError("Lots already exist for this run")
    lots = [
        Lot(
            run_id=run_id,
            **item.model_dump(exclude={"requirements"}),
            requirements=[Requirement(**requirement.model_dump()) for requirement in item.requirements],
        )
        for item in procurement.lots
    ]
    session.add_all(lots)
    await session.commit()
    # expire_on_commit=False keeps the initialized requirement collections usable.
    return lots
