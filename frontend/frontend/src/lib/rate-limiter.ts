import { query } from "./db";

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetSeconds: number;
}

export async function checkRateLimit(
  key: string,
  maxAttempts: number,
  windowSeconds: number
): Promise<RateLimitResult> {
  const now = new Date();
  const expiresAt = new Date(now.getTime() + windowSeconds * 1000);

  // Atomic upsert with expiration
  const sql = `
    INSERT INTO rate_limits (key, count, expires_at)
    VALUES ($1, 1, $2)
    ON CONFLICT (key) DO UPDATE
    SET
      count = CASE
        WHEN rate_limits.expires_at < NOW() THEN 1
        ELSE rate_limits.count + 1
      END,
      expires_at = CASE
        WHEN rate_limits.expires_at < NOW() THEN $2
        ELSE rate_limits.expires_at
      END
    RETURNING count, EXTRACT(EPOCH FROM (expires_at - NOW()))::int AS ttl;
  `;

  try {
    const res = await query<{ count: number; ttl: number }>(sql, [key, expiresAt]);
    const row = res.rows[0];
    const currentCount = row ? row.count : 1;
    const ttl = row ? Math.max(0, row.ttl) : windowSeconds;

    if (currentCount > maxAttempts) {
      return {
        allowed: false,
        remaining: 0,
        resetSeconds: ttl,
      };
    }

    return {
      allowed: true,
      remaining: Math.max(0, maxAttempts - currentCount),
      resetSeconds: ttl,
    };
  } catch (err) {
    console.error("[RATE_LIMITER] Error checking rate limit:", err);
    // On unexpected DB rate limit error, fail closed or open safely?
    // For rate limiting, fail-open with logging so users are not blocked on DB transient errors
    return { allowed: true, remaining: 1, resetSeconds: 0 };
  }
}

export async function resetRateLimit(key: string): Promise<void> {
  try {
    await query(`DELETE FROM rate_limits WHERE key = $1`, [key]);
  } catch (err) {
    console.error("[RATE_LIMITER] Error resetting rate limit:", err);
  }
}

/**
 * Checks whether ANY of the provided rate limit keys have exceeded maxAttempts in a single DB query.
 * Does NOT increment any counters (read-only pre-flight check).
 */
export async function checkRateLimitStatusBulk(
  keys: string[],
  maxAttempts: number
): Promise<{ allowed: boolean; waitSeconds: number }> {
  if (!keys || keys.length === 0) return { allowed: true, waitSeconds: 0 };
  const sql = `
    SELECT key, count, EXTRACT(EPOCH FROM (expires_at - NOW()))::int AS ttl
    FROM rate_limits
    WHERE key = ANY($1) AND expires_at > NOW();
  `;
  try {
    const res = await query<{ key: string; count: number; ttl: number }>(sql, [keys]);
    for (const row of res.rows) {
      if (row.count >= maxAttempts) {
        return { allowed: false, waitSeconds: Math.max(0, row.ttl) };
      }
    }
    return { allowed: true, waitSeconds: 0 };
  } catch (err) {
    console.error("[RATE_LIMITER] Error checking bulk rate limits:", err);
    return { allowed: true, waitSeconds: 0 };
  }
}

/**
 * Increments failure count for multiple keys in a single atomic database statement.
 * Called only when an actual authentication failure occurs.
 */
export async function recordFailedAttemptsBulk(
  keys: string[],
  windowSeconds: number
): Promise<void> {
  if (!keys || keys.length === 0) return;
  const now = new Date();
  const expiresAt = new Date(now.getTime() + windowSeconds * 1000);

  const sql = `
    INSERT INTO rate_limits (key, count, expires_at)
    SELECT unnest($1::text[]), 1, $2
    ON CONFLICT (key) DO UPDATE
    SET
      count = CASE
        WHEN rate_limits.expires_at < NOW() THEN 1
        ELSE rate_limits.count + 1
      END,
      expires_at = CASE
        WHEN rate_limits.expires_at < NOW() THEN $2
        ELSE rate_limits.expires_at
      END;
  `;

  try {
    await query(sql, [keys, expiresAt]);
  } catch (err) {
    console.error("[RATE_LIMITER] Error recording failed attempts:", err);
  }
}

