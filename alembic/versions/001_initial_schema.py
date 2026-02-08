"""Initial schema - all tables for multi-user platform.

Revision ID: 001
Revises:
Create Date: 2026-02-07
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("first_name", sa.String(100), nullable=False),
        sa.Column("last_name", sa.String(100), nullable=False),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("linkedin_public_id", sa.String(200), nullable=True),
        sa.Column("linkedin_profile_url", sa.String(500), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
    )

    # Invitation codes
    op.create_table(
        "codes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("code", sa.String(50), unique=True, nullable=False, index=True),
        sa.Column("name", sa.String(200), nullable=True),
        sa.Column("max_uses", sa.Integer(), default=1),
        sa.Column("current_uses", sa.Integer(), default=0),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # Payments
    op.create_table(
        "payments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("amount_usd", sa.Float(), default=5.0),
        sa.Column("payment_method", sa.String(50), nullable=False),
        sa.Column("invitation_code_id", UUID(as_uuid=True), sa.ForeignKey("codes.id"), nullable=True),
        sa.Column("status", sa.String(20), default="completed"),
        sa.Column("max_days_back", sa.Integer(), default=90),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # Scrape sessions
    op.create_table(
        "scrape_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("payment_id", UUID(as_uuid=True), sa.ForeignKey("payments.id"), nullable=False),
        sa.Column("from_date", sa.DateTime(), nullable=False),
        sa.Column("to_date", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(20), default="pending"),
        sa.Column("posts_scraped", sa.Integer(), default=0),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # Posts
    op.create_table(
        "posts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("scrape_session_id", UUID(as_uuid=True), sa.ForeignKey("scrape_sessions.id"), nullable=True),
        sa.Column("linkedin_activity_id", sa.String(200), nullable=False, index=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("date", sa.DateTime(), nullable=False, index=True),
        sa.Column("content_type", sa.String(50), default="text"),
        sa.Column("reactions", sa.Integer(), default=0),
        sa.Column("comments", sa.Integer(), default=0),
        sa.Column("shares", sa.Integer(), default=0),
        sa.Column("engagement", sa.Integer(), default=0),
        sa.Column("impressions", sa.Integer(), default=0),
        sa.Column("has_image", sa.Boolean(), default=False),
        sa.Column("image_urls", JSON, nullable=True),
        sa.Column("text_length", sa.Integer(), default=0),
        sa.Column("url", sa.String(500), nullable=True),
        sa.Column("is_repost", sa.Boolean(), default=False),
        sa.Column("original_author", sa.String(200), nullable=True),
        sa.Column("reshare_comment", sa.Text(), nullable=True),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("sentiment", sa.String(50), nullable=True),
        sa.Column("topics", JSON, nullable=True),
        sa.Column("image_type", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("user_id", "linkedin_activity_id", name="uq_user_activity"),
    )

    # Analysis results
    op.create_table(
        "analysis_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("analysis_type", sa.String(50), nullable=False),
        sa.Column("from_date", sa.DateTime(), nullable=True),
        sa.Column("to_date", sa.DateTime(), nullable=True),
        sa.Column("post_count", sa.Integer(), default=0),
        sa.Column("result_data", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("analysis_results")
    op.drop_table("posts")
    op.drop_table("scrape_sessions")
    op.drop_table("payments")
    op.drop_table("codes")
    op.drop_table("users")
