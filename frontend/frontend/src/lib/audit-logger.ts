import { query } from "./db";

export type AuditEventType =
  | "AUTH_REGISTER"
  | "AUTH_VERIFY_EMAIL"
  | "AUTH_LOGIN_SUCCESS"
  | "AUTH_LOGIN_FAILURE"
  | "AUTH_LOGOUT"
  | "AUTH_SESSION_REVOKE"
  | "USER_ROLE_CHANGE"
  | "USER_RESTRICT"
  | "COMMENT_SUBMIT"
  | "COMMENT_MODERATE"
  | "ADMIN_INVITE";

export type ActionResult = "SUCCESS" | "FAILURE" | "REJECTED";

export interface LogAuditParams {
  eventType: AuditEventType;
  actorId?: string | null;
  resourceType?: string | null;
  resourceId?: string | null;
  actionResult: ActionResult;
  ipHash?: string | null;
  metadata?: Record<string, unknown>;
}

// Redact sensitive patterns if passed accidentally
function sanitizeMetadata(meta?: Record<string, unknown>): Record<string, unknown> {
  if (!meta) return {};
  const cleaned: Record<string, unknown> = {};
  const forbidden = ["password", "token", "hash", "secret", "cookie", "authorization"];
  for (const [key, value] of Object.entries(meta)) {
    const lowerKey = key.toLowerCase();
    if (forbidden.some((f) => lowerKey.includes(f))) {
      cleaned[key] = "[REDACTED]";
    } else if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      cleaned[key] = sanitizeMetadata(value as Record<string, unknown>);
    } else {
      cleaned[key] = value;
    }
  }
  return cleaned;
}

export async function logAudit(params: LogAuditParams): Promise<void> {
  const sql = `
    INSERT INTO audit_logs (
      event_type, actor_id, resource_type, resource_id,
      action_result, ip_hash, metadata, created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW());
  `;

  const safeMeta = sanitizeMetadata(params.metadata);

  try {
    await query(sql, [
      params.eventType,
      params.actorId || null,
      params.resourceType || null,
      params.resourceId || null,
      params.actionResult,
      params.ipHash || null,
      JSON.stringify(safeMeta),
    ]);
  } catch (err) {
    console.error("[AUDIT_LOG] Failed to write audit log:", err);
  }
}

