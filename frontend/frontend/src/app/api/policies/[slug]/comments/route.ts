import { NextResponse } from "next/server";

/**
 * Public comment endpoints — DISABLED (P0 containment).
 *
 * The previous handlers accepted unauthenticated comment submissions, wrote them to a JSON file on
 * disk without rate limiting, moderation or locking, and returned every stored submission to anyone.
 * Both methods stay disabled until authentication, abuse protection, moderation, privacy rules and a
 * dedicated storage location are in place. The comment form on policy pages is a local preview and
 * does not call this endpoint.
 */
const DISABLED_RESPONSE = {
  success: false,
  error: "COMMENTS_DISABLED",
  message: "Public comment submission and retrieval are disabled in this prototype. No comments are stored or published."
};

export async function GET() {
  return NextResponse.json(DISABLED_RESPONSE, { status: 503 });
}

export async function POST() {
  return NextResponse.json(DISABLED_RESPONSE, { status: 503 });
}
