import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { getSessionFromRequest } from "@/lib/session-manager";

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

    const { searchParams } = new URL(req.url);
    const limit = Math.min(Math.max(1, parseInt(searchParams.get("limit") || "50", 10)), 100);
    const offset = Math.max(0, parseInt(searchParams.get("offset") || "0", 10));

    const countRes = await query<{ count: string }>(`SELECT COUNT(*) FROM audit_logs`);
    const total = parseInt(countRes.rows[0]?.count || "0", 10);

    const sql = `
      SELECT
        a.id,
        a.event_type AS "eventType",
        a.actor_id AS "actorId",
        u.email AS "actorEmail",
        a.resource_type AS "resourceType",
        a.resource_id AS "resourceId",
        a.action_result AS "actionResult",
        a.ip_hash AS "ipHash",
        a.metadata,
        a.created_at AS "createdAt"
      FROM audit_logs a
      LEFT JOIN users u ON a.actor_id = u.id
      ORDER BY a.created_at DESC
      LIMIT $1 OFFSET $2;
    `;

    const res = await query(sql, [limit, offset]);

    return NextResponse.json({
      success: true,
      total,
      limit,
      offset,
      logs: res.rows,
    });
  } catch (err) {
    console.error("[ADMIN_AUDIT_LOGS_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to load audit logs." },
      { status: 500 }
    );
  }
}

