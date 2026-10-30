import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { getSessionFromRequest } from "@/lib/session-manager";
import { logAudit } from "@/lib/audit-logger";
import { extractClientIp, hashIp } from "@/lib/auth-crypto";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const sessionUser = await getSessionFromRequest(req);
    if (!sessionUser) {
      return NextResponse.json({ success: false, error: "UNAUTHENTICATED" }, { status: 401 });
    }

    if (sessionUser.role !== "MODERATOR" && sessionUser.role !== "ADMIN") {
      return NextResponse.json(
        { success: false, error: "FORBIDDEN", message: "Moderator or Admin role required." },
        { status: 403 }
      );
    }

    const body = await req.json().catch(() => null);
    const action = body?.action?.toUpperCase();
    const reason = typeof body?.reason === "string" ? body.reason.trim() : "";

    if (action !== "APPROVE" && action !== "REJECT") {
      return NextResponse.json(
        { success: false, error: "INVALID_ACTION", message: "Action must be APPROVE or REJECT." },
        { status: 400 }
      );
    }

    const newStatus = action === "APPROVE" ? "APPROVED" : "REJECTED";

    const updateSql = `
      UPDATE policy_comments
      SET
        status = $1,
        moderated_by = $2,
        moderated_at = NOW(),
        moderation_reason = $3
      WHERE id = $4
      RETURNING id, policy_slug, status;
    `;

    const res = await query(updateSql, [newStatus, sessionUser.id, reason || null, id]);
    if (res.rows.length === 0) {
      return NextResponse.json(
        { success: false, error: "NOT_FOUND", message: "Comment not found." },
        { status: 404 }
      );
    }

    await logAudit({
      eventType: "COMMENT_MODERATE",
      actorId: sessionUser.id,
      resourceType: "policy_comment",
      resourceId: id,
      actionResult: "SUCCESS",
      ipHash: hashIp(extractClientIp(req)),
      metadata: { action, reason, newStatus },
    });

    return NextResponse.json({
      success: true,
      comment: res.rows[0],
    });
  } catch (err) {
    console.error("[MODERATION_ACTION_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to update comment." },
      { status: 500 }
    );
  }
}

