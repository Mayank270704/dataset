import { NextRequest, NextResponse } from "next/server";
import {
  SESSION_COOKIE_NAME,
  revokeSession,
  getClearSessionCookieHeader,
} from "@/lib/session-manager";
import { logAudit } from "@/lib/audit-logger";
import { extractClientIp, hashIp } from "@/lib/auth-crypto";
import { validateOrigin } from "@/lib/csrf";

export async function POST(req: NextRequest) {
  try {
    if (!validateOrigin(req)) {
      return NextResponse.json(
        { success: false, error: "FORBIDDEN", message: "Cross-origin request rejected." },
        { status: 403 }
      );
    }

    const cookieHeader = req.headers.get("cookie") || "";
    const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${SESSION_COOKIE_NAME}=([^;]+)`));
    const token = match && match[1] ? decodeURIComponent(match[1]) : null;

    if (token) {
      await revokeSession(token);
      await logAudit({
        eventType: "AUTH_LOGOUT",
        actionResult: "SUCCESS",
        ipHash: hashIp(extractClientIp(req)),
      });
    }

    const response = NextResponse.json({
      success: true,
      message: "Logged out successfully.",
    });

    response.headers.set("Set-Cookie", getClearSessionCookieHeader());
    return response;
  } catch (err) {
    console.error("[LOGOUT_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Logout failed." },
      { status: 500 }
    );
  }
}

