import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import {
  isValidEmail,
  normalizeEmail,
  verifyPassword,
  extractClientIp,
  hashIp,
} from "@/lib/auth-crypto";
import {
  checkRateLimitStatusBulk,
  recordFailedAttemptsBulk,
  resetRateLimit,
} from "@/lib/rate-limiter";
import { logAudit } from "@/lib/audit-logger";
import { createSession, getSessionCookieHeader } from "@/lib/session-manager";
import { validateOrigin } from "@/lib/csrf";

export async function POST(req: NextRequest) {
  const startTime = performance.now();
  try {
    if (!validateOrigin(req)) {
      return NextResponse.json(
        { success: false, error: "FORBIDDEN", message: "Cross-origin request rejected." },
        { status: 403 }
      );
    }

    const ip = extractClientIp(req);
    const ipHash = hashIp(ip);

    const body = await req.json().catch(() => null);
    if (!body || typeof body !== "object") {
      return NextResponse.json(
        { success: false, error: "INVALID_REQUEST", message: "Invalid JSON request body." },
        { status: 400 }
      );
    }

    const { email, password } = body;

    if (!isValidEmail(email) || typeof password !== "string") {
      return NextResponse.json(
        { success: false, error: "INVALID_CREDENTIALS", message: "Invalid email or password." },
        { status: 401 }
      );
    }

    const emailNorm = normalizeEmail(email);
    const ipKey = `login:fail:ip:${ipHash}`;
    const emailKey = `login:fail:email:${emailNorm}`;

    // Fast atomic rate-limit pre-check (single query, non-mutating)
    const rateLimitCheck = await checkRateLimitStatusBulk([ipKey, emailKey], 5);
    if (!rateLimitCheck.allowed) {
      const waitMinutes = Math.ceil(rateLimitCheck.waitSeconds / 60) || 15;
      logAudit({
        eventType: "AUTH_LOGIN_FAILURE",
        actionResult: "REJECTED",
        ipHash,
        metadata: { reason: "RATE_LIMIT_EXCEEDED", emailNorm },
      }).catch((e) => console.error("[AUDIT_LOG_ERROR]", e));

      return NextResponse.json(
        {
          success: false,
          error: "TOO_MANY_ATTEMPTS",
          message: `Too many failed login attempts. Please try again in ${waitMinutes} minutes.`,
        },
        { status: 429 }
      );
    }

    // Lookup user by normalized email
    const sql = `
      SELECT
        id,
        email,
        COALESCE(display_name, name, 'Citizen User') AS display_name,
        password_hash,
        role,
        is_email_verified,
        COALESCE(status, 'ACTIVE') AS status,
        suspension_reason
      FROM users
      WHERE email_normalized = $1 OR LOWER(email) = $1;
    `;
    const userRes = await query(sql, [emailNorm]);
    const user = userRes.rows[0];

    // Constant-time failure handling: if user not found
    if (!user) {
      await recordFailedAttemptsBulk([ipKey, emailKey], 15 * 60);
      logAudit({
        eventType: "AUTH_LOGIN_FAILURE",
        actionResult: "FAILURE",
        ipHash,
        metadata: { reason: "USER_NOT_FOUND", emailNorm },
      }).catch((e) => console.error("[AUDIT_LOG_ERROR]", e));

      return NextResponse.json(
        { success: false, error: "INVALID_CREDENTIALS", message: "Invalid email or password." },
        { status: 401 }
      );
    }

    // Check account status
    if (user.status === "SUSPENDED") {
      logAudit({
        eventType: "AUTH_LOGIN_FAILURE",
        actorId: user.id,
        actionResult: "REJECTED",
        ipHash,
        metadata: { reason: "ACCOUNT_SUSPENDED" },
      }).catch((e) => console.error("[AUDIT_LOG_ERROR]", e));

      return NextResponse.json(
        {
          success: false,
          error: "ACCOUNT_SUSPENDED",
          message: `Your account is suspended: ${user.suspension_reason || "Violation of policy guidelines."}`,
        },
        { status: 403 }
      );
    }

    // Verify Argon2id password hash
    const isValid = await verifyPassword(user.password_hash, password);
    if (!isValid) {
      await recordFailedAttemptsBulk([ipKey, emailKey], 15 * 60);
      logAudit({
        eventType: "AUTH_LOGIN_FAILURE",
        actorId: user.id,
        actionResult: "FAILURE",
        ipHash,
        metadata: { reason: "PASSWORD_MISMATCH" },
      }).catch((e) => console.error("[AUDIT_LOG_ERROR]", e));

      return NextResponse.json(
        { success: false, error: "INVALID_CREDENTIALS", message: "Invalid email or password." },
        { status: 401 }
      );
    }

    // STRICT AUTHENTICATION: Block login for unverified accounts
    // Never create a session or issue an auth cookie for unverified users
    if (!user.is_email_verified) {
      logAudit({
        eventType: "AUTH_LOGIN_FAILURE",
        actorId: user.id,
        actionResult: "REJECTED",
        ipHash,
        metadata: { reason: "EMAIL_NOT_VERIFIED", emailNorm },
      }).catch((e) => console.error("[AUDIT_LOG_ERROR]", e));

      return NextResponse.json(
        {
          success: false,
          error: "EMAIL_NOT_VERIFIED",
          message: "Please verify your email address before signing in. Check your inbox or request a new verification link.",
          email: user.email,
        },
        { status: 403 }
      );
    }

    // Clean up any stale failure counters on successful login in background
    resetRateLimit(emailKey).catch(() => {});

    // Create session (32-byte opaque token, SHA-256 hash stored in PostgreSQL)
    const { token, expiresAt } = await createSession(user.id, req);

    logAudit({
      eventType: "AUTH_LOGIN_SUCCESS",
      actorId: user.id,
      actionResult: "SUCCESS",
      ipHash,
    }).catch((e) => console.error("[AUDIT_LOG_ERROR]", e));

    const totalDuration = Math.round(performance.now() - startTime);
    console.log(`[AUTH_TIMING] Login succeeded in ${totalDuration}ms for user ${user.id}`);

    // Build response with secure HttpOnly SameSite=Lax cookie
    const response = NextResponse.json({
      success: true,
      message: "Login successful.",
      user: {
        id: user.id,
        email: user.email,
        displayName: user.display_name,
        role: user.role,
        isEmailVerified: user.is_email_verified,
        status: user.status,
      },
    });

    response.headers.set("Set-Cookie", getSessionCookieHeader(token, expiresAt));
    return response;
  } catch (err) {
    console.error("[LOGIN_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Login failed. Please try again later." },
      { status: 500 }
    );
  }
}

