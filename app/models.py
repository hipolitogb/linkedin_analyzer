"""SQLAlchemy ORM models for the LinkedIn Analyzer multi-user platform."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone = Column(String(50), nullable=True)
    linkedin_public_id = Column(String(200), nullable=True)
    linkedin_profile_url = Column(String(500), nullable=True)
    encrypted_linkedin_cookies = Column(Text, nullable=True)
    linkedin_cookies_updated_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    payments = relationship("Payment", back_populates="user")
    scrape_sessions = relationship("ScrapeSession", back_populates="user")
    posts = relationship("Post", back_populates="user")


class InvitationCode(Base):
    __tablename__ = "codes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=True)
    max_uses = Column(Integer, default=1)
    current_uses = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Payment(Base):
    __tablename__ = "payments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    amount_usd = Column(Float, default=5.0)
    payment_method = Column(String(50), nullable=False)  # "invitation_code" | "mock_payment"
    invitation_code_id = Column(UUID(as_uuid=True), ForeignKey("codes.id"), nullable=True)
    status = Column(String(20), default="completed")  # "completed" | "pending" | "failed"
    max_days_back = Column(Integer, default=90)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="payments")
    scrape_session = relationship("ScrapeSession", back_populates="payment", uselist=False)


class ScrapeSession(Base):
    __tablename__ = "scrape_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    payment_id = Column(UUID(as_uuid=True), ForeignKey("payments.id"), nullable=False)
    from_date = Column(DateTime, nullable=False)
    to_date = Column(DateTime, nullable=False)
    status = Column(String(20), default="pending")  # "pending" | "scraping" | "completed" | "failed"
    posts_scraped = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="scrape_sessions")
    payment = relationship("Payment", back_populates="scrape_session")


class Post(Base):
    __tablename__ = "posts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    scrape_session_id = Column(UUID(as_uuid=True), ForeignKey("scrape_sessions.id"), nullable=True)
    linkedin_activity_id = Column(String(200), nullable=False, index=True)
    text = Column(Text, nullable=True)
    date = Column(DateTime, nullable=False, index=True)
    content_type = Column(String(50), default="text")
    reactions = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    engagement = Column(Integer, default=0)
    impressions = Column(Integer, default=0)
    has_image = Column(Boolean, default=False)
    image_urls = Column(JSON, default=list)
    text_length = Column(Integer, default=0)
    url = Column(String(500), nullable=True)
    is_repost = Column(Boolean, default=False)
    original_author = Column(String(200), nullable=True)
    reshare_comment = Column(Text, nullable=True)
    # AI classification fields
    category = Column(String(50), nullable=True)
    sentiment = Column(String(50), nullable=True)
    topics = Column(JSON, nullable=True)
    image_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "linkedin_activity_id", name="uq_user_activity"),
    )

    user = relationship("User", back_populates="posts")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)  # "pattern_analysis" | "dashboard_cache"
    from_date = Column(DateTime, nullable=True)
    to_date = Column(DateTime, nullable=True)
    post_count = Column(Integer, default=0)
    result_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
