import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import {
  isValidEmail,
  normalizeEmail,
  generateSecureToken,
  hashToken,
  extractClientIp,
  hashIp,
} from "@/lib/auth-crypto";
import { checkRateLimit } from "@/lib/rate-limiter";
import { emailService } from "@/lib/email-service";
import { getSessionFromRequest } from "@/lib/session-manager";

export async function POST(req: NextRequest) {
  try {
    const ipHash = hashIp(extractClientIp(req));
    const sessionUser = await getSessionFromRequest(req);
    const body = await req.json().catch(() => null);

    const targetEmail = sessionUser?.email || body?.email;
    if (!targetEmail || !isValidEmail(targetEmail)) {
      return NextResponse.json(
        { success: false, error: "INVALID_EMAIL", message: "Please provide a valid email address." },
        { status: 400 }
      );
    }

    const emailNorm = normalizeEmail(targetEmail);

    // Rate limit: 3 resend attempts per hour per email
    const limit = await checkRateLimit(`resend:email:${emailNorm}`, 3, 3600);
    if (!limit.allowed) {
      return NextResponse.json(
        {
          success: false,
          error: "RATE_LIMIT_EXCEEDED",
          message: `Too many verification requests. Please wait ${Math.ceil(limit.resetSeconds / 60)} minutes before requesting again.`,
        },
        { status: 429 }
      );
    }

    const userRes = await query<{ id: string; email: string; is_email_verified: boolean }>(
      `SELECT id, email, is_email_verified FROM users WHERE email_normalized = $1 OR LOWER(email) = $1`,
      [emailNorm]
    );
    const user = userRes.rows[0];

    // Generic safe response to prevent enumeration if user not found
    if (!user) {
      return NextResponse.json({
        success: true,
        message: "If an account exists for this email, a verification link has been sent.",
      });
    }

    // Invalidate prior unused tokens
    await query(
      `UPDATE email_verifications SET used_at = NOW() WHERE user_id = $1 AND used_at IS NULL`,
      [user.id]
    );

    // Create fresh token
    const token = generateSecureToken(32);
    const tokenHash = hashToken(token);
    const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);

    await query(
      `INSERT INTO email_verifications (user_id, token_hash, expires_at) VALUES ($1, $2, $3)`,
      [user.id, tokenHash, expiresAt]
    );
    console.log(`[AUTH_TRACE: VERIFICATION_TOKEN_CREATED] Fresh token stored in DB for user ${user.id}`);

    const baseUrl = req.nextUrl.origin;
    console.log(`[AUTH_TRACE: EMAIL_DISPATCH_STARTED] Calling sendVerificationEmail for user ${user.id} via Resend...`);
    const emailResult = await emailService.sendVerificationEmail(user.email, token, baseUrl);

    if (emailResult.success) {
      console.log(`[AUTH_TRACE: RESEND_API_RESPONSE] Resend accepted email for user ${user.id} (Status: ${emailResult.deliveryStatus}, ID: ${emailResult.messageId || "N/A"})`);
    } else {
      console.error(`[AUTH_TRACE: EMAIL_DISPATCH_FAILED] Resend rejected dispatch for user ${user.id}: ${emailResult.error}`);
    }

    if (!emailResult.success) {
      return NextResponse.json(
        {
          success: false,
          error: "EMAIL_DELIVERY_FAILED",
          message: `Could not deliver verification email: ${emailResult.error || "Email provider error"}.`,
        },
        { status: 502 }
      );
    }

    return NextResponse.json({
      success: true,
      message: "A verification email has been accepted by the provider. Please check your inbox and spam folder.",
      delivery: {
        accepted: emailResult.accepted,
        delivered: emailResult.delivered,
        status: emailResult.deliveryStatus,
        messageId: emailResult.messageId,
        details: emailResult.details,
      },
    });
  } catch (err) {
    console.error("[RESEND_VERIFICATION_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Could not send verification email." },
      { status: 500 }
    );
  }
}

