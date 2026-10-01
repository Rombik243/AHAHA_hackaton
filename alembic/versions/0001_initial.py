"""Create the initial procurement analysis schema.

Revision ID: 0001_initial
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("stage", sa.String(50), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("input_content", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name="ck_run_status"),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_run_progress"),
    )
    op.create_index("ix_runs_status", "runs", ["status"])
    op.create_table(
        "lots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_id", sa.String(200), nullable=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("okpd2", sa.String(50), nullable=True),
        sa.Column("okpd2_source", sa.String(20), nullable=False),
        sa.Column("okpd2_similarity", sa.Float(), nullable=True),
        sa.Column("okpd2_source_url", sa.String(2000), nullable=True),
        sa.Column("okpd2_source_quote", sa.Text(), nullable=True),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("budget", sa.Float(), nullable=True),
    )
    op.create_index("ix_lots_run_id", "lots", ["run_id"])
    op.create_table(
        "requirements",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("lot_id", sa.Integer(), sa.ForeignKey("lots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("fact_type", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("expected_value", sa.JSON(), nullable=False),
        sa.Column("operator", sa.String(20), nullable=False),
    )
    op.create_index("ix_requirements_lot_id", "requirements", ["lot_id"])
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("inn", sa.String(12), nullable=True, unique=True),
        sa.Column("website", sa.String(2000), nullable=True),
        sa.Column("normalized_name", sa.String(500), nullable=False),
        sa.Column("domain", sa.String(500), nullable=True),
    )
    op.create_index("ix_companies_normalized_name", "companies", ["normalized_name"])
    op.create_index("ix_companies_domain", "companies", ["domain"])
    op.create_table(
        "company_facts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("company_id", sa.Integer(), sa.ForeignKey("companies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("lot_id", sa.Integer(), sa.ForeignKey("lots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("fact_type", sa.String(100), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("source_url", sa.String(2000), nullable=False),
        sa.Column("source_title", sa.String(1000), nullable=False),
        sa.Column("source_quote", sa.Text(), nullable=False),
        sa.Column("source_kind", sa.String(20), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_fact_scope", "company_facts", ["run_id", "lot_id", "company_id"])
    op.create_table(
        "requirement_checks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("company_id", sa.Integer(), sa.ForeignKey("companies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("lot_id", sa.Integer(), sa.ForeignKey("lots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("requirement_id", sa.Integer(), sa.ForeignKey("requirements.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("actual_value", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.UniqueConstraint("company_id", "requirement_id", name="uq_check_company_requirement"),
        sa.CheckConstraint("status IN ('confirmed', 'contradicted', 'unknown')", name="ck_requirement_check_status"),
    )
    op.create_index("ix_requirement_checks_lot_id", "requirement_checks", ["lot_id"])
    op.create_table(
        "company_lot_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("company_id", sa.Integer(), sa.ForeignKey("companies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("lot_id", sa.Integer(), sa.ForeignKey("lots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("base_score", sa.Float(), nullable=False),
        sa.Column("history_bonus", sa.Float(), nullable=False),
        sa.Column("history_wins", sa.Integer(), nullable=False),
        sa.Column("company_snapshot", sa.JSON(), nullable=False),
        sa.Column("confirmed_count", sa.Integer(), nullable=False),
        sa.Column("contradicted_count", sa.Integer(), nullable=False),
        sa.Column("unknown_count", sa.Integer(), nullable=False),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("origins", sa.JSON(), nullable=False),
        sa.UniqueConstraint("company_id", "lot_id", name="uq_result_company_lot"),
        sa.CheckConstraint("score >= 0 AND score <= 100", name="ck_result_score"),
        sa.CheckConstraint("coverage >= 0 AND coverage <= 1", name="ck_result_coverage"),
    )
    op.create_index("ix_company_lot_results_lot_id", "company_lot_results", ["lot_id"])


def downgrade() -> None:
    op.drop_table("company_lot_results")
    op.drop_table("requirement_checks")
    op.drop_table("company_facts")
    op.drop_table("companies")
    op.drop_table("requirements")
    op.drop_table("lots")
    op.drop_table("runs")
