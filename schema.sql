
-- this is postgresql schema for a multi-tenant application with billing, rate limits, and usage tracking
-- ======================
-- Billing Plans
-- ======================
drop table if exists billing_plans cascade;
CREATE TABLE billing_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    price_monthly DECIMAL(10, 2),
    description TEXT,
    query_limit INT,
    token_limit INT,
    stt_seconds INT,
    tts_seconds INT,
    docs_limit INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ======================
-- Tenants
-- ======================
drop table if exists tenants cascade;
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    billing_plan_id UUID REFERENCES billing_plans(id),
    configs JSONB
);

-- ======================
-- Users
-- ======================
drop table if exists users cascade;
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    fname TEXT,
    lname TEXT,
    email TEXT UNIQUE NOT NULL,
    dob DATE,
    gender TEXT,
    display_name TEXT,
    is_admin BOOLEAN DEFAULT FALSE
);

-- ======================
-- Rate Limits
-- ======================
drop table if exists rate_limits cascade;
CREATE TABLE rate_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    billing_plan_id UUID NOT NULL REFERENCES billing_plans(id),
    name TEXT NOT NULL,              -- e.g. 'qps', 'concurrent_requests'
    value INT NOT NULL,
    unit TEXT NOT NULL,             -- e.g. 'second', 'minute'
    scope TEXT NOT NULL,            -- 'per_user', 'per_tenant'
    created_at TIMESTAMP DEFAULT NOW()
);

-- ======================
-- Usage
-- ======================
drop table if exists usage cascade;
CREATE TABLE usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    queries INT DEFAULT 0,
    tokens INT DEFAULT 0,
    stt_seconds INT DEFAULT 0,
    tts_seconds INT DEFAULT 0,
    docs_uploaded INT DEFAULT 0,
    updated TIMESTAMP DEFAULT NOW()
);

-- ======================
-- Documents
-- ======================
drop table if exists documents cascade;
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID NOT NULL REFERENCES users(id),
    filename TEXT NOT NULL,
    is_public BOOLEAN DEFAULT FALSE,
    path TEXT,                      -- optional: local/server path if needed
    filetype TEXT,
    size INT,
    status TEXT DEFAULT 'processing',
    upload_time TIMESTAMP DEFAULT NOW()
);

-- ======================
-- Chunks
-- ======================
drop table if exists chunks cascade;
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(id),
    text TEXT NOT NULL,
    language TEXT      -- or use JSONB if you're not using pgvector
);

-- ======================
-- Query Logs
-- ======================
drop table if exists query_logs cascade;
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    session_id UUID,
    query TEXT NOT NULL,
    response TEXT,
    latency INT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- ======================
-- Sessions
-- ======================
drop table if exists sessions cascade;
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    queries INT DEFAULT 0,
    tokens INT DEFAULT 0,
    seconds INT DEFAULT 0,
    channel TEXT,              -- e.g. 'voice', 'web'
    status TEXT DEFAULT 'active'
);
