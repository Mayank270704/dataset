import { NextRequest, NextResponse } from "next/server";
import { query, withTransaction } from "@/lib/db";
import { hashToken, extractClientIp, hashIp } from "@/lib/auth-crypto";
import { logAudit } from "@/lib/audit-logger";

export async function POST(req: NextRequest) {
  try {
    const ipHash = hashIp(extractClientIp(req));
    const body = await req.json().catch(() => null);
    const token = body?.token;

    if (!token || typeof token !== "string" || token.length < 10) {
      return NextResponse.json(
        { success: false, error: "INVALID_TOKEN", message: "A valid verification token is required." },
        { status: 400 }
      );
    }

    const tokenHash = hashToken(token);

    // Look up verification record
    const sql = `
      SELECT id, user_id, expires_at, used_at
      FROM email_verifications
      WHERE token_hash = $1;
    `;
    const res = await query(sql, [tokenHash]);
    const record = res.rows[0];

    if (!record) {
      await logAudit({
        eventType: "AUTH_VERIFY_EMAIL",
        actionResult: "FAILURE",
        ipHash,
        metadata: { reason: "TOKEN_NOT_FOUND" },
      });
      return NextResponse.json(
        { success: false, error: "INVALID_TOKEN", message: "Verification link is invalid or has expired." },
        { status: 400 }
      );
    }

    if (record.used_at) {
      await logAudit({
        eventType: "AUTH_VERIFY_EMAIL",
        actorId: record.user_id,
        actionResult: "REJECTED",
        ipHash,
        metadata: { reason: "TOKEN_ALREADY_USED" },
      });
      return NextResponse.json(
        { success: false, error: "TOKEN_USED", message: "This verification link has already been used." },
        { status: 400 }
      );
    }

    if (new Date(record.expires_at) < new Date()) {
      await logAudit({
        eventType: "AUTH_VERIFY_EMAIL",
        actorId: record.user_id,
        actionResult: "REJECTED",
        ipHash,
        metadata: { reason: "TOKEN_EXPIRED" },
      });
      return NextResponse.json(
        { success: false, error: "TOKEN_EXPIRED", message: "This verification link has expired. Please request a new one." },
        { status: 400 }
      );
    }

    // Atomic transaction: mark token as used and set user.is_email_verified = TRUE
    await withTransaction(async (client) => {
      await client.query(
        `UPDATE email_verifications SET used_at = NOW() WHERE id = $1`,
        [record.id]
      );
      await client.query(
        `UPDATE users SET is_email_verified = TRUE, updated_at = NOW() WHERE id = $1`,
        [record.user_id]
      );
    });

    await logAudit({
      eventType: "AUTH_VERIFY_EMAIL",
      actorId: record.user_id,
      actionResult: "SUCCESS",
      ipHash,
    });

    return NextResponse.json({
      success: true,
      message: "Your email address has been verified successfully. You can now participate and share comments.",
    });
  } catch (err) {
    console.error("[VERIFY_EMAIL_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Email verification failed. Please try again." },
      { status: 500 }
    );
  }
}

