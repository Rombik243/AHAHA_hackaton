"""Persist candidate-only results without changing the ranking schema."""
from alembic import op
import sqlalchemy as sa

revision = "0002_candidate_search"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candidate_search_results",
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("candidate_search_results")
