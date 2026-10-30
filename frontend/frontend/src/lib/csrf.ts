/**
 * Origin-based CSRF protection for cookie-authenticated state-changing requests.
 *
 * Same-site cookies (SameSite=Lax) already provide substantial protection, but
 * an explicit Origin check adds defence-in-depth for cross-origin POST requests.
 *
 * Rules:
 *  - Requests without an Origin header are assumed to be same-origin (e.g.
 *    server-to-server, curl, or browsers that never send Origin on GET). They
 *    pass without restriction.
 *  - Requests whose Origin host matches the Host header are allowed.
 *  - All other Origins are rejected.
 */
export function validateOrigin(req: Request): boolean {
  const origin = req.headers.get("origin");
  const host = req.headers.get("host");

  // No Origin header → allow (same-origin requests, curl, server calls)
  if (!origin) return true;
  // No Host header → can't validate; be conservative and allow (reverse proxy
  // may strip Host in some edge configurations)
  if (!host) return true;

  try {
    const originHost = new URL(origin).host;
    return originHost === host;
  } catch {
    // Malformed Origin URL → reject
    return false;
  }
}
