-- PolicyLens Database Schema
-- Compatible with PostgreSQL (including existing Prisma migrations and CUID/UUID keys)

-- 1. Ensure User attributes exist
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    email VARCHAR(255) UNIQUE NOT NULL,
    email_normalized VARCHAR(255),
    display_name VARCHAR(100),
    name VARCHAR(100),
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'CITIZEN',
    is_email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    suspension_reason TEXT,
    stakeholder_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS email_normalized VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS name VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'ACTIVE';
ALTER TABLE users ADD COLUMN IF NOT EXISTS suspension_reason TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS stakeholder_type TEXT;

UPDATE users SET email_normalized = LOWER(email) WHERE email_normalized IS NULL;
UPDATE users SET display_name = COALESCE(name, display_name, 'Citizen User') WHERE display_name IS NULL;
UPDATE users SET name = COALESCE(display_name, name, 'Citizen User') WHERE name IS NULL;

CREATE INDEX IF NOT EXISTS idx_users_email_norm ON users(email_normalized);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- 2. Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    ip_hash VARCHAR(64),
    ip_address TEXT,
    user_agent VARCHAR(255),
    expires_at TIMESTAMPTZ NOT NULL,
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS ip_hash VARCHAR(64);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS ip_address TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS is_revoked BOOLEAN DEFAULT FALSE;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

-- 3. Email verifications table
CREATE TABLE IF NOT EXISTS email_verifications (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_email_verif_token ON email_verifications(token_hash);
CREATE INDEX IF NOT EXISTS idx_email_verif_user ON email_verifications(user_id);

-- 4. Admin Invitations
CREATE TABLE IF NOT EXISTS admin_invitations (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    email VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    invited_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    accepted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_admin_invitations_token ON admin_invitations(token_hash);
CREATE INDEX IF NOT EXISTS idx_admin_invitations_email ON admin_invitations(email);

-- 5. Durable per-user-policy comment quota (enforces hard max 5 comments)
CREATE TABLE IF NOT EXISTS policy_comment_quotas (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    policy_slug VARCHAR(120) NOT NULL,
    submission_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, policy_slug),
    CONSTRAINT quota_non_negative CHECK (submission_count >= 0)
);

-- 6. Policy comments table
CREATE TABLE IF NOT EXISTS policy_comments (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    policy_slug VARCHAR(120) NOT NULL,
    stakeholder VARCHAR(100) NOT NULL,
    original_text TEXT NOT NULL,
    normalized_text_hash VARCHAR(64) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    moderation_reason TEXT,
    moderated_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    moderated_at TIMESTAMPTZ,
    spam_score FLOAT DEFAULT 0.0,
    spam_flags JSONB DEFAULT '[]'::jsonb,
    ip_hash VARCHAR(64),
    user_agent VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comments_policy_status ON policy_comments(policy_slug, status);
CREATE INDEX IF NOT EXISTS idx_comments_user ON policy_comments(user_id);
CREATE INDEX IF NOT EXISTS idx_comments_norm_hash ON policy_comments(policy_slug, normalized_text_hash);

-- 7. Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    event_type VARCHAR(50) NOT NULL,
    actor_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(120),
    action_result VARCHAR(20) NOT NULL,
    ip_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_event ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);

-- 8. Rate Limits
CREATE TABLE IF NOT EXISTS rate_limits (
    key VARCHAR(255) PRIMARY KEY,
    count INT NOT NULL DEFAULT 1,
    expires_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE rate_limits ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_rate_limits_expires ON rate_limits(expires_at);
