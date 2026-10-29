import { NextResponse } from "next/server";

/**
 * Policy analysis endpoint — DISABLED (P0 containment).
 *
 * This route previously returned a hard-coded "pilot benchmark" record that described Canadian
 * consultation comments as human-validated analysis of an Indian policy. That record has no verified
 * provenance and has been withdrawn. No analysis is served until verified research outputs are
 * connected through a reviewed, read-only integration.
 */
export async function GET() {
  return NextResponse.json(
    {
      success: false,
      error: "ANALYSIS_UNAVAILABLE",
      message:
        "Policy analysis is not available. PolicyLens has no validated analysis to publish for this policy."
    },
    { status: 503 }
  );
}
