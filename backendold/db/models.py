import uuid
from sqlalchemy import (
    Column, String, Integer, Boolean, Date, DateTime, ForeignKey,
    DECIMAL, Text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from .connection import Base


class BillingPlan(Base):
    __tablename__ = "billing_plans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    price_monthly = Column(DECIMAL(10, 2))
    description = Column(Text)
    query_limit = Column(Integer)
    token_limit = Column(Integer)
    stt_seconds = Column(Integer)
    tts_seconds = Column(Integer)
    docs_limit = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    billing_plan_id = Column(UUID(as_uuid=True), ForeignKey("billing_plans.id"))
    configs = Column(JSONB)
    join_code_hash = Column(String, nullable=False)  # ADD THIS LINE


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    fname = Column(String)
    lname = Column(String)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)  # ADD THIS LINE
    dob = Column(Date)
    gender = Column(String)
    display_name = Column(String)
    is_admin = Column(Boolean, default=False)


class RateLimit(Base):
    __tablename__ = "rate_limits"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    billing_plan_id = Column(UUID(as_uuid=True), ForeignKey("billing_plans.id"), nullable=False)
    name = Column(String, nullable=False)
    value = Column(Integer, nullable=False)
    unit = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Usage(Base):
    __tablename__ = "usage"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    queries = Column(Integer, default=0)
    tokens = Column(Integer, default=0)
    stt_seconds = Column(Integer, default=0)
    tts_seconds = Column(Integer, default=0)
    docs_uploaded = Column(Integer, default=0)
    updated = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    is_public = Column(Boolean, default=False)
    path = Column(String)
    filetype = Column(String)
    size = Column(Integer)
    status = Column(String, default='processing')
    upload_time = Column(DateTime, default=datetime.utcnow)


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    language = Column(String)


class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(UUID(as_uuid=True))
    query = Column(Text, nullable=False)
    response = Column(Text)
    latency = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    queries = Column(Integer, default=0)
    tokens = Column(Integer, default=0)
    seconds = Column(Integer, default=0)
    channel = Column(String)
    status = Column(String, default='active')


class Link(Base):
    __tablename__ = "links"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    url = Column(String, nullable=False)
    name = Column(String)
    description = Column(Text)
    depth = Column(Integer, default=1)
    excluded_sublinks = Column(String)  # Pipe-separated patterns
    status = Column(String, default='active')  # active, disabled, scraping, completed, error
    last_scraped = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
