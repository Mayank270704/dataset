import { cookies } from "next/headers";
import { query } from "./db";
import { generateSecureToken, hashToken, hashIp, extractClientIp } from "./auth-crypto";

export const SESSION_COOKIE_NAME = "policylens_session";
export const SESSION_DURATION_SECONDS = 7 * 24 * 60 * 60; // 7 days

export type UserRole = "CITIZEN" | "MODERATOR" | "ADMIN";
export type UserStatus = "ACTIVE" | "RESTRICTED" | "SUSPENDED";

export interface SessionUser {
  id: string;
  email: string;
  displayName: string;
  role: UserRole;
  isEmailVerified: boolean;
  status: UserStatus;
  suspensionReason?: string | null;
}

export interface UserSession {
  sessionId: string;
  userId: string;
  tokenHash: string;
  expiresAt: Date;
}

export async function createSession(
  userId: string,
  req?: Request
): Promise<{ token: string; expiresAt: Date }> {
  const token = generateSecureToken(32);
  const tokenHash = hashToken(token);
  const expiresAt = new Date(Date.now() + SESSION_DURATION_SECONDS * 1000);

  const ipHash = req ? hashIp(extractClientIp(req)) : null;
  const userAgent = req ? req.headers.get("user-agent")?.slice(0, 255) || null : null;

  const sql = `
    INSERT INTO sessions (user_id, token_hash, ip_hash, user_agent, expires_at, created_at)
    VALUES ($1, $2, $3, $4, $5, NOW())
    RETURNING id;
  `;

  await query(sql, [userId, tokenHash, ipHash, userAgent, expiresAt]);

  return { token, expiresAt };
}

export async function validateSessionToken(token: string): Promise<SessionUser | null> {
  if (!token || typeof token !== "string" || token.length < 20) {
    return null;
  }

  const tokenHash = hashToken(token);

  const sql = `
    SELECT
      u.id,
      u.email,
      COALESCE(u.display_name, u.name, 'Citizen User') AS "displayName",
      u.role,
      u.is_email_verified AS "isEmailVerified",
      COALESCE(u.status, 'ACTIVE') AS "status",
      u.suspension_reason AS "suspensionReason",
      s.id AS "sessionId",
      s.expires_at AS "expiresAt"
    FROM sessions s
    JOIN users u ON s.user_id = u.id
    WHERE s.token_hash = $1
      AND s.revoked_at IS NULL
      AND (s.is_revoked IS FALSE OR s.is_revoked IS NULL)
      AND s.expires_at > NOW();
  `;

  const res = await query<SessionUser & { sessionId: string; expiresAt: Date }>(sql, [tokenHash]);
  if (!res.rows[0]) {
    return null;
  }

  const row = res.rows[0];
  if (row.status === "SUSPENDED" || !row.isEmailVerified) {
    return null;
  }

  return {
    id: row.id,
    email: row.email,
    displayName: row.displayName,
    role: row.role,
    isEmailVerified: row.isEmailVerified,
    status: row.status,
    suspensionReason: row.suspensionReason,
  };
}

export async function getSessionFromRequest(req: Request): Promise<SessionUser | null> {
  // Extract token from Cookie header
  const cookieHeader = req.headers.get("cookie") || "";
  const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${SESSION_COOKIE_NAME}=([^;]+)`));
  if (!match || !match[1]) {
    return null;
  }
  const token = decodeURIComponent(match[1]);
  return validateSessionToken(token);
}

export async function getCurrentUser(): Promise<SessionUser | null> {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get(SESSION_COOKIE_NAME)?.value;
    if (!token) return null;
    return validateSessionToken(token);
  } catch {
    return null;
  }
}

export async function revokeSession(token: string): Promise<void> {
  if (!token) return;
  const tokenHash = hashToken(token);
  await query(
    `UPDATE sessions SET is_revoked = TRUE, revoked_at = NOW() WHERE token_hash = $1`,
    [tokenHash]
  );
}

export async function revokeAllUserSessions(userId: string): Promise<void> {
  await query(
    `UPDATE sessions SET is_revoked = TRUE, revoked_at = NOW() WHERE user_id = $1`,
    [userId]
  );
}

export function getSessionCookieHeader(token: string, expiresAt: Date): string {
  const isProd = process.env.NODE_ENV === "production";
  return `${SESSION_COOKIE_NAME}=${encodeURIComponent(token)}; Path=/; HttpOnly; SameSite=Lax; Expires=${expiresAt.toUTCString()}; Max-Age=${SESSION_DURATION_SECONDS}${isProd ? "; Secure" : ""}`;
}

export function getClearSessionCookieHeader(): string {
  const isProd = process.env.NODE_ENV === "production";
  return `${SESSION_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Max-Age=0${isProd ? "; Secure" : ""}`;
}

