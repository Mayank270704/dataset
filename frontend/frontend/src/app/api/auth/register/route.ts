import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import {
  isValidEmail,
  normalizeEmail,
  validatePassword,
  hashPassword,
  generateSecureToken,
  hashToken,
  extractClientIp,
  hashIp,
} from "@/lib/auth-crypto";
import { checkRateLimit } from "@/lib/rate-limiter";
import { logAudit } from "@/lib/audit-logger";
import { emailService } from "@/lib/email-service";

export async function POST(req: NextRequest) {
  try {
    const ip = extractClientIp(req);
    const ipHash = hashIp(ip);

    console.log(`[AUTH_TRACE: REGISTER_STARTED] Registration attempt received from IP hash ${ipHash.slice(0, 10)}...`);

    // Rate limit: 5 registration requests per hour per IP
    const rateLimit = await checkRateLimit(`reg:ip:${ipHash}`, 5, 3600);
    if (!rateLimit.allowed) {
      console.log(`[AUTH_TRACE: REGISTER_RATE_LIMITED] IP hash ${ipHash.slice(0, 10)}... exceeded hourly registration limit.`);
      await logAudit({
        eventType: "AUTH_REGISTER",
        actionResult: "REJECTED",
        ipHash,
        metadata: { reason: "RATE_LIMIT_EXCEEDED" },
      });
      return NextResponse.json(
        {
          success: false,
          error: "RATE_LIMIT_EXCEEDED",
          message: `Too many registration attempts. Please try again in ${Math.ceil(rateLimit.resetSeconds / 60)} minutes.`,
        },
        { status: 429 }
      );
    }

    const body = await req.json().catch(() => null);
    if (!body || typeof body !== "object") {
      return NextResponse.json(
        { success: false, error: "INVALID_REQUEST", message: "Invalid JSON request body." },
        { status: 400 }
      );
    }

    const { email, password, displayName } = body;

    // Validate email
    if (!isValidEmail(email)) {
      return NextResponse.json(
        { success: false, error: "INVALID_EMAIL", message: "Please enter a valid email address." },
        { status: 400 }
      );
    }

    // Validate password
    const passCheck = validatePassword(password);
    if (!passCheck.valid) {
      return NextResponse.json(
        {
          success: false,
          error: "WEAK_PASSWORD",
          message: passCheck.errors[0] || "Password does not meet complexity requirements.",
          details: passCheck.errors,
        },
        { status: 400 }
      );
    }

    // Validate display name
    const cleanName = typeof displayName === "string" ? displayName.trim() : "";
    if (cleanName.length < 2 || cleanName.length > 80) {
      return NextResponse.json(
        { success: false, error: "INVALID_NAME", message: "Display name must be between 2 and 80 characters." },
        { status: 400 }
      );
    }

    const emailNorm = normalizeEmail(email);
    const domain = emailNorm.split("@")[1] || "unknown";
    console.log(`[AUTH_TRACE: VALIDATION_PASSED] Credentials validated for domain @${domain}`);

    const baseUrl = req.nextUrl.origin;

    // Check if email already registered
    const existing = await query<{ id: string; email: string; is_email_verified: boolean }>(
      `SELECT id, email, is_email_verified FROM users WHERE email_normalized = $1 OR LOWER(email) = $1`,
      [emailNorm]
    );

    if (existing.rows.length > 0) {
      const existingUser = existing.rows[0];

      if (existingUser.is_email_verified) {
        console.log(`[AUTH_TRACE: REGISTER_EXISTING_VERIFIED] User ${existingUser.id} is already registered and verified.`);
        await logAudit({
          eventType: "AUTH_REGISTER",
          actorId: existingUser.id,
          actionResult: "REJECTED",
          ipHash,
          metadata: { reason: "DUPLICATE_EMAIL_ALREADY_VERIFIED", emailNorm },
        });

        return NextResponse.json(
          {
            success: false,
            error: "EMAIL_ALREADY_EXISTS",
            message: "An account with this email address already exists and is verified. Please sign in with your password.",
            alreadyVerified: true,
          },
          { status: 409 }
        );
      }

      // Existing UNVERIFIED user: allow re-triggering verification email without creating duplicate user
      console.log(`[AUTH_TRACE: REGISTER_EXISTING_UNVERIFIED_RETRY] User ${existingUser.id} exists but is unverified. Generating fresh token...`);
      
      // Invalidate prior unused tokens
      await query(
        `UPDATE email_verifications SET used_at = NOW() WHERE user_id = $1 AND used_at IS NULL`,
        [existingUser.id]
      );

      const verificationToken = generateSecureToken(32);
      const verificationTokenHash = hashToken(verificationToken);
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);

      await query(
        `INSERT INTO email_verifications (user_id, token_hash, expires_at)
         VALUES ($1, $2, $3)`,
        [existingUser.id, verificationTokenHash, expiresAt]
      );
      console.log(`[AUTH_TRACE: VERIFICATION_TOKEN_CREATED] Fresh token stored in DB for existing unverified user ${existingUser.id}`);

      console.log(`[AUTH_TRACE: EMAIL_DISPATCH_STARTED] Calling sendVerificationEmail for user ${existingUser.id} via Resend...`);
      const emailResult = await emailService.sendVerificationEmail(existingUser.email, verificationToken, baseUrl);

      if (emailResult.success) {
        console.log(`[AUTH_TRACE: RESEND_API_RESPONSE] Resend accepted email for user ${existingUser.id} (Status: ${emailResult.deliveryStatus}, ID: ${emailResult.messageId || "N/A"})`);
      } else {
        console.error(`[AUTH_TRACE: EMAIL_DISPATCH_FAILED] Resend rejected dispatch for user ${existingUser.id}: ${emailResult.error}`);
      }

      await logAudit({
        eventType: "AUTH_REGISTER",
        actorId: existingUser.id,
        actionResult: emailResult.success ? "SUCCESS" : "REJECTED",
        ipHash,
        metadata: {
          emailNorm,
          isRetryForExistingUnverified: true,
          emailDeliverySuccess: emailResult.success,
          emailDeliveryAccepted: emailResult.accepted,
          emailDeliveryStatus: emailResult.deliveryStatus,
          emailDeliveryError: emailResult.error || null,
        },
      });

      if (!emailResult.success) {
        return NextResponse.json(
          {
            success: false,
            error: "EMAIL_DELIVERY_FAILED",
            message: `Account exists, but verification email could not be delivered: ${emailResult.error || "Provider error"}. Please check your email configuration in Resend.`,
            userId: existingUser.id,
            delivery: {
              delivered: false,
              accepted: false,
              status: emailResult.deliveryStatus,
              error: emailResult.error,
            },
          },
          { status: 502 }
        );
      }

      return NextResponse.json(
        {
          success: true,
          message: "An account with this email address was previously created. A fresh verification email has been dispatched. Please check your inbox (and spam folder).",
          userId: existingUser.id,
          delivery: {
            accepted: emailResult.accepted,
            delivered: emailResult.delivered,
            status: emailResult.deliveryStatus,
            messageId: emailResult.messageId,
            details: emailResult.details,
          },
        },
        { status: 200 }
      );
    }

    // New user registration
    // Hash password with Argon2id
    const passwordHash = await hashPassword(password);

    // Public registration always assigns role 'CITIZEN' and is_email_verified = false
    const userRes = await query<{ id: string }>(
      `INSERT INTO users (email, email_normalized, display_name, name, password_hash, role, is_email_verified, status)
       VALUES ($1, $2, $3::text, $3::text, $4, 'CITIZEN', FALSE, 'ACTIVE')
       RETURNING id`,
      [email.trim(), emailNorm, cleanName, passwordHash]
    );

    const newUser = userRes.rows[0];
    console.log(`[AUTH_TRACE: USER_CREATED] New CITIZEN user record persisted with ID ${newUser.id} (is_email_verified: false)`);

    // Generate cryptographically secure email verification token
    const verificationToken = generateSecureToken(32);
    const verificationTokenHash = hashToken(verificationToken);
    const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours

    await query(
      `INSERT INTO email_verifications (user_id, token_hash, expires_at)
       VALUES ($1, $2, $3)`,
      [newUser.id, verificationTokenHash, expiresAt]
    );
    console.log(`[AUTH_TRACE: VERIFICATION_TOKEN_CREATED] Verification token hash stored in PostgreSQL for user ${newUser.id}`);

    // Dispatch verification email via Resend
    console.log(`[AUTH_TRACE: EMAIL_DISPATCH_STARTED] Calling sendVerificationEmail for new user ${newUser.id} to domain @${domain}...`);
    const emailResult = await emailService.sendVerificationEmail(email.trim(), verificationToken, baseUrl);

    if (emailResult.success) {
      console.log(`[AUTH_TRACE: RESEND_API_RESPONSE] Resend accepted email for new user ${newUser.id} (Status: ${emailResult.deliveryStatus}, ID: ${emailResult.messageId || "N/A"})`);
    } else {
      console.error(`[AUTH_TRACE: EMAIL_DISPATCH_FAILED] Resend rejected dispatch for new user ${newUser.id}: ${emailResult.error}`);
    }

    await logAudit({
      eventType: "AUTH_REGISTER",
      actorId: newUser.id,
      actionResult: emailResult.success ? "SUCCESS" : "REJECTED",
      ipHash,
      metadata: {
        emailNorm,
        emailDeliverySuccess: emailResult.success,
        emailDeliveryAccepted: emailResult.accepted,
        emailDeliveryStatus: emailResult.deliveryStatus,
        emailDeliveryError: emailResult.error || null,
      },
    });

    if (!emailResult.success) {
      return NextResponse.json(
        {
          success: false,
          error: "EMAIL_DELIVERY_FAILED",
          message: `Account created, but verification email could not be delivered: ${emailResult.error || "Provider error"}. Please check your email configuration or domain verification in Resend.`,
          userId: newUser.id,
          delivery: {
            delivered: false,
            accepted: false,
            status: emailResult.deliveryStatus,
            error: emailResult.error,
          },
        },
        { status: 502 }
      );
    }

    return NextResponse.json(
      {
        success: true,
        message: "Registration successful. A verification email has been sent. Please check your inbox (and spam folder).",
        userId: newUser.id,
        delivery: {
          accepted: emailResult.accepted,
          delivered: emailResult.delivered,
          status: emailResult.deliveryStatus,
          messageId: emailResult.messageId,
          details: emailResult.details,
        },
      },
      { status: 201 }
    );
  } catch (err) {
    console.error("[REGISTER_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Registration failed. Please try again later." },
      { status: 500 }
    );
  }
}

