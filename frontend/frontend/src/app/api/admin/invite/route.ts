import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { getSessionFromRequest } from "@/lib/session-manager";
import {
  isValidEmail,
  normalizeEmail,
  generateSecureToken,
  hashToken,
  extractClientIp,
  hashIp,
} from "@/lib/auth-crypto";
import { logAudit } from "@/lib/audit-logger";

export async function POST(req: NextRequest) {
  try {
    const sessionUser = await getSessionFromRequest(req);
    if (!sessionUser) {
      return NextResponse.json({ success: false, error: "UNAUTHENTICATED" }, { status: 401 });
    }

    if (sessionUser.role !== "ADMIN") {
      return NextResponse.json(
        { success: false, error: "FORBIDDEN", message: "Admin role required." },
        { status: 403 }
      );
    }

    const body = await req.json().catch(() => null);
    const { email, role } = body || {};

    if (!isValidEmail(email)) {
      return NextResponse.json(
        { success: false, error: "INVALID_EMAIL", message: "Valid email is required." },
        { status: 400 }
      );
    }

    if (role !== "MODERATOR" && role !== "ADMIN") {
      return NextResponse.json(
        { success: false, error: "INVALID_ROLE", message: "Role must be MODERATOR or ADMIN." },
        { status: 400 }
      );
    }

    const emailNorm = normalizeEmail(email);

    // If an account already exists with this email, we can promote it directly
    const existingUser = await query<{ id: string; role: string }>(
      `SELECT id, role FROM users WHERE email_normalized = $1`,
      [emailNorm]
    );

    if (existingUser.rows.length > 0) {
      const targetUser = existingUser.rows[0];
      await query(`UPDATE users SET role = $1, updated_at = NOW() WHERE id = $2`, [role, targetUser.id]);

      await logAudit({
        eventType: "USER_ROLE_CHANGE",
        actorId: sessionUser.id,
        resourceType: "user",
        resourceId: targetUser.id,
        actionResult: "SUCCESS",
        ipHash: hashIp(extractClientIp(req)),
        metadata: { oldRole: targetUser.role, newRole: role },
      });

      return NextResponse.json({
        success: true,
        message: `Existing account ${emailNorm} promoted to ${role}.`,
      });
    }

    // Otherwise create invitation
    const inviteToken = generateSecureToken(32);
    const tokenHash = hashToken(inviteToken);
    const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000); // 7 days

    await query(
      `INSERT INTO admin_invitations (email, role, token_hash, invited_by_user_id, expires_at)
       VALUES ($1, $2, $3, $4, $5)`,
      [emailNorm, role, tokenHash, sessionUser.id, expiresAt]
    );

    await logAudit({
      eventType: "ADMIN_INVITE",
      actorId: sessionUser.id,
      actionResult: "SUCCESS",
      ipHash: hashIp(extractClientIp(req)),
      metadata: { invitedEmail: emailNorm, role },
    });

    return NextResponse.json({
      success: true,
      message: `Invitation generated for ${emailNorm} as ${role}.`,
      inviteToken, // Returned in admin session response
    });
  } catch (err) {
    console.error("[ADMIN_INVITE_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to create invitation." },
      { status: 500 }
    );
  }
}

