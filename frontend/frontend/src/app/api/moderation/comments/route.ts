import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { getSessionFromRequest } from "@/lib/session-manager";

export async function GET(req: NextRequest) {
  try {
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

    const { searchParams } = new URL(req.url);
    const statusFilter = searchParams.get("status") || "PENDING,FLAGGED";
    const statuses = statusFilter.split(",").map((s) => s.trim().toUpperCase());

    const sql = `
      SELECT
        c.id,
        c.policy_slug AS "policySlug",
        c.stakeholder,
        c.original_text AS "originalText",
        c.status,
        c.moderation_reason AS "moderationReason",
        c.spam_score AS "spamScore",
        c.spam_flags AS "spamFlags",
        c.created_at AS "createdAt",
        u.id AS "authorId",
        u.email AS "authorEmail",
        u.display_name AS "authorName",
        u.status AS "authorStatus"
      FROM policy_comments c
      JOIN users u ON c.user_id = u.id
      WHERE c.status = ANY($1)
      ORDER BY c.created_at DESC
      LIMIT 100;
    `;

    const res = await query(sql, [statuses]);

    return NextResponse.json({
      success: true,
      comments: res.rows,
    });
  } catch (err) {
    console.error("[MODERATION_GET_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to load moderation queue." },
      { status: 500 }
    );
  }
}

