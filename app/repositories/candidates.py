from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candidate import CandidateSearchResult
from app.schemas.candidates import CandidateResults, CandidateRunResults
from app.schemas.run import RunStatus


async def save_candidates(session: AsyncSession, run_id: int, results: CandidateResults) -> None:
    row = await session.get(CandidateSearchResult, run_id)
    if row is None:
        session.add(CandidateSearchResult(run_id=run_id, payload=results.model_dump(mode="json")))
    else:
        row.payload = results.model_dump(mode="json")
    await session.commit()


async def get_candidates(session: AsyncSession, run) -> CandidateRunResults:
    row = await session.get(CandidateSearchResult, run.id)
    return CandidateRunResults(run=RunStatus.model_validate(run), **(row.payload if row else {}))
