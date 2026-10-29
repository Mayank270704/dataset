import { NextRequest, NextResponse } from "next/server";
import { isValidSlugFormat, isKnownPolicySlug } from "@/lib/security";

/**
 * Trigger Analysis Route Handler (Disabled / Safely Contained)
 *
 * Security Containment:
 * Direct external execution of Python processes via unauthenticated HTTP POST
 * has been decommissioned. In accordance with application security architecture,
 * background analysis jobs will only execute via authenticated administrative
 * triggers or an isolated background task worker queue.
 */
export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params;

  // 1. Strict slug format validation (reject path traversal, special characters)
  if (!isValidSlugFormat(slug)) {
    return NextResponse.json(
      {
        success: false,
        error: "INVALID_SLUG",
        message: "Policy identifier contains invalid characters or format."
      },
      { status: 400 }
    );
  }

  // 2. Verified policy existence
  if (!isKnownPolicySlug(slug)) {
    return NextResponse.json(
      {
        success: false,
        error: "POLICY_NOT_FOUND",
        message: "Policy not found."
      },
      { status: 404 }
    );
  }

  // 3. Safely reject unauthenticated external execution
  return NextResponse.json(
    {
      success: false,
      error: "EXTERNAL_TRIGGER_DISABLED",
      message:
        "Direct unauthenticated execution of background analysis jobs is disabled. Analysis workflows are scheduled internally after the consultation window closes."
    },
    { status: 403 }
  );
}
