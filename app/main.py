"""Run with: python -m uvicorn app.main:app --reload"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app import database
from app.agent.pipeline import ProcurementPipeline
from app.api.runs import router
from app.api.suppliers import router as suppliers_router
from app.config import Settings, get_settings
from app.repositories.runs import mark_interrupted


def create_app(
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    sessions = session_factory or database.SessionLocal
    bound_engine = sessions.kw["bind"]

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        if settings.auto_create_tables:
            async with bound_engine.begin() as connection:
                await connection.run_sync(database.Base.metadata.create_all)
        async with sessions() as session:
            await mark_interrupted(session)
        application.state.pipeline = ProcurementPipeline(settings, sessions)
        yield
        await bound_engine.dispose()

    application = FastAPI(
        title="Procurement Scout", version="0.1.0", lifespan=lifespan,
        description="Supplier discovery with evidence. Default demo mode uses fictional local data. Run one API worker.",
    )
    application.state.settings = settings
    application.state.session_factory = sessions
    application.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"], allow_headers=["*"],
    )
    application.include_router(router)
    application.include_router(suppliers_router)

    @application.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "web_search_provider": settings.web_search_provider,
                "llm_enabled": settings.llm_enabled, "test_mode": settings.test_mode,
                "max_run_seconds": settings.max_run_seconds if settings.test_mode else None}

    return application


app = create_app()
