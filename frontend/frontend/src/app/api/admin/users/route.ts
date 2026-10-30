import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { getSessionFromRequest } from "@/lib/session-manager";
import { logAudit } from "@/lib/audit-logger";
import { extractClientIp, hashIp } from "@/lib/auth-crypto";

export async function GET(req: NextRequest) {
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

    const sql = `
      SELECT
        u.id,
        u.email,
        u.display_name AS "displayName",
        u.role,
        u.is_email_verified AS "isEmailVerified",
        u.status,
        u.suspension_reason AS "suspensionReason",
        u.created_at AS "createdAt",
        (SELECT COUNT(*)::int FROM policy_comments WHERE user_id = u.id) AS "commentsCount"
      FROM users u
      ORDER BY u.created_at DESC
      LIMIT 100;
    `;

    const res = await query(sql);

    return NextResponse.json({
      success: true,
      users: res.rows,
    });
  } catch (err) {
    console.error("[ADMIN_USERS_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to load users." },
      { status: 500 }
    );
  }
}

export async function PATCH(req: NextRequest) {
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
    const { userId, status, reason } = body || {};

    if (!userId || !["ACTIVE", "RESTRICTED", "SUSPENDED"].includes(status)) {
      return NextResponse.json(
        { success: false, error: "INVALID_INPUT", message: "Valid userId and status are required." },
        { status: 400 }
      );
    }

    // Prevent modifying self
    if (userId === sessionUser.id) {
      return NextResponse.json(
        { success: false, error: "INVALID_ACTION", message: "Admins cannot modify their own account status." },
        { status: 400 }
      );
    }

    const sql = `
      UPDATE users
      SET status = $1, suspension_reason = $2, updated_at = NOW()
      WHERE id = $3
      RETURNING id, email, role, status;
    `;

    const res = await query(sql, [status, reason || null, userId]);
    if (res.rows.length === 0) {
      return NextResponse.json(
        { success: false, error: "NOT_FOUND", message: "User not found." },
        { status: 404 }
      );
    }

    await logAudit({
      eventType: "USER_RESTRICT",
      actorId: sessionUser.id,
      resourceType: "user",
      resourceId: userId,
      actionResult: "SUCCESS",
      ipHash: hashIp(extractClientIp(req)),
      metadata: { status, reason },
    });

    return NextResponse.json({
      success: true,
      user: res.rows[0],
    });
  } catch (err) {
    console.error("[ADMIN_USER_STATUS_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to update user." },
      { status: 500 }
    );
  }
}

