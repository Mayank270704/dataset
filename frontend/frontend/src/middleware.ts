import { NextRequest, NextResponse } from "next/server";

/**
 * Lightweight route-protection middleware.
 *
 * Design constraints:
 *  - No DB calls here (middleware runs on every matched request on the Edge).
 *  - Only checks for the presence of the session cookie; the actual session
 *    validity is enforced by each API route / page that calls getSessionFromRequest().
 *  - Admin UI pages redirect unauthenticated visitors to /login.
 *  - Admin API routes return 401 JSON for unauthenticated callers.
 */

const SESSION_COOKIE_NAME = "pl_session";

function hasSessionCookie(req: NextRequest): boolean {
  return Boolean(req.cookies.get(SESSION_COOKIE_NAME)?.value);
}

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // ── Protect /api/admin/* ───────────────────────────────────────────────────
  if (pathname.startsWith("/api/admin/") || pathname.startsWith("/api/moderation/")) {
    if (!hasSessionCookie(req)) {
      return NextResponse.json(
        {
          success: false,
          error: "UNAUTHENTICATED",
          message: "Authentication required.",
        },
        { status: 401 }
      );
    }
    return NextResponse.next();
  }

  // ── Protect /admin (UI page) ───────────────────────────────────────────────
  if (pathname.startsWith("/admin")) {
    if (!hasSessionCookie(req)) {
      const loginUrl = req.nextUrl.clone();
      loginUrl.pathname = "/login";
      loginUrl.searchParams.set("redirect", pathname);
      return NextResponse.redirect(loginUrl);
    }
    return NextResponse.next();
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/admin/:path*",
    "/api/admin/:path*",
    "/api/moderation/:path*",
  ],
};
