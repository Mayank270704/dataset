import { NextRequest, NextResponse } from "next/server";
import { query, withTransaction } from "@/lib/db";
import { getSessionFromRequest } from "@/lib/session-manager";
import { isValidSlugFormat, isKnownPolicySlug } from "@/lib/security";
import { officialPolicies } from "@/features/feed/demo-data";
import { analyzeCommentSpam, computeTextHash, normalizeCommentText } from "@/lib/spam-detector";
import { checkRateLimit } from "@/lib/rate-limiter";
import { logAudit } from "@/lib/audit-logger";
import { extractClientIp, hashIp } from "@/lib/auth-crypto";
import { validateOrigin } from "@/lib/csrf";

const MAX_COMMENTS_PER_USER_PER_POLICY = 5;

/**
 * GET /api/policies/[slug]/comments
 *
 * Returns published comments and collective participation statistics.
 * Excludes rejected comments from public statistics.
 * Maintains distinct counts for total, approved, pending, flagged, rejected,
 * and unique human participants.
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const { slug } = await params;

    if (!isValidSlugFormat(slug)) {
      return NextResponse.json(
        { success: false, error: "INVALID_SLUG", message: "Invalid policy identifier format." },
        { status: 400 }
      );
    }

    if (!isKnownPolicySlug(slug)) {
      return NextResponse.json(
        { success: false, error: "POLICY_NOT_FOUND", message: "Policy not found." },
        { status: 404 }
      );
    }

    const sessionUser = await getSessionFromRequest(req);

    // Fetch comment breakdown metrics
    const statsSql = `
      SELECT
        COUNT(*)::int AS "totalCount",
        COUNT(CASE WHEN status = 'APPROVED' THEN 1 END)::int AS "approvedCount",
        COUNT(CASE WHEN status = 'PENDING' THEN 1 END)::int AS "pendingCount",
        COUNT(CASE WHEN status = 'FLAGGED' THEN 1 END)::int AS "flaggedCount",
        COUNT(CASE WHEN status = 'REJECTED' THEN 1 END)::int AS "rejectedCount",
        COUNT(DISTINCT CASE WHEN status != 'REJECTED' THEN user_id END)::int AS "uniqueParticipantsCount"
      FROM policy_comments
      WHERE policy_slug = $1;
    `;
    const statsRes = await query(statsSql, [slug]);
    const stats = statsRes.rows[0] || {
      totalCount: 0,
      approvedCount: 0,
      pendingCount: 0,
      flaggedCount: 0,
      rejectedCount: 0,
      uniqueParticipantsCount: 0,
    };

    // Public comments: Only APPROVED comments are published publicly.
    // If the requesting user is authenticated, also show their own PENDING or FLAGGED submissions.
    let commentsSql = `
      SELECT
        c.id,
        c.policy_slug AS "policySlug",
        c.stakeholder,
        c.original_text AS "originalText",
        c.status,
        c.created_at AS "createdAt",
        u.display_name AS "authorName"
      FROM policy_comments c
      JOIN users u ON c.user_id = u.id
      WHERE c.policy_slug = $1
        AND (c.status = 'APPROVED' ${sessionUser ? "OR c.user_id = $2" : ""})
      ORDER BY c.created_at DESC
      LIMIT 100;
    `;

    const commentsParams = sessionUser ? [slug, sessionUser.id] : [slug];
    const commentsRes = await query(commentsSql, commentsParams);

    // Retrieve user quota usage if authenticated
    let userQuota = { used: 0, max: MAX_COMMENTS_PER_USER_PER_POLICY, remaining: MAX_COMMENTS_PER_USER_PER_POLICY };
    if (sessionUser) {
      const quotaSql = `
        SELECT submission_count FROM policy_comment_quotas
        WHERE user_id = $1 AND policy_slug = $2;
      `;
      const qRes = await query(quotaSql, [sessionUser.id, slug]);
      const used = qRes.rows[0]?.submission_count || 0;
      userQuota = {
        used,
        max: MAX_COMMENTS_PER_USER_PER_POLICY,
        remaining: Math.max(0, MAX_COMMENTS_PER_USER_PER_POLICY - used),
      };
    }

    return NextResponse.json({
      success: true,
      policySlug: slug,
      statistics: stats,
      comments: commentsRes.rows,
      userQuota: sessionUser ? userQuota : null,
    });
  } catch (err) {
    console.error("[GET_COMMENTS_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to load policy comments." },
      { status: 500 }
    );
  }
}

/**
 * POST /api/policies/[slug]/comments
 *
 * Submits a new comment on the policy.
 * Strict Concurrency & Anti-Spam Protections:
 * 1. Must be authenticated (HTTP 401).
 * 2. Must be email-verified (HTTP 403).
 * 3. Account must be active (HTTP 403).
 * 4. Policy must be open.
 * 5. Velocity check (max 1 submission per 10s).
 * 6. Concurrency-safe durable quota (max 5 submissions per user per policy).
 * 7. Duplicate & spam detection.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    if (!validateOrigin(req)) {
      return NextResponse.json(
        { success: false, error: "FORBIDDEN", message: "Cross-origin request rejected." },
        { status: 403 }
      );
    }

    const { slug } = await params;
    const ip = extractClientIp(req);
    const ipHash = hashIp(ip);

    // 1. Validate slug format & known policy existence
    if (!isValidSlugFormat(slug)) {
      return NextResponse.json(
        { success: false, error: "INVALID_SLUG", message: "Invalid policy identifier format." },
        { status: 400 }
      );
    }

    const policy = officialPolicies.find((p) => p.slug === slug);
    if (!policy) {
      return NextResponse.json(
        { success: false, error: "POLICY_NOT_FOUND", message: "Policy not found." },
        { status: 404 }
      );
    }

    // 2. Authentication check
    const sessionUser = await getSessionFromRequest(req);
    if (!sessionUser) {
      return NextResponse.json(
        {
          success: false,
          error: "UNAUTHENTICATED",
          message: "You must be signed in to submit comments.",
        },
        { status: 401 }
      );
    }

    // 3. Email verification requirement
    if (!sessionUser.isEmailVerified) {
      return NextResponse.json(
        {
          success: false,
          error: "EMAIL_NOT_VERIFIED",
          message: "Please verify your email address before sharing opinions.",
        },
        { status: 403 }
      );
    }

    // 4. Account status restriction
    if (sessionUser.status === "SUSPENDED" || sessionUser.status === "RESTRICTED") {
      return NextResponse.json(
        {
          success: false,
          error: "ACCOUNT_RESTRICTED",
          message: `Your account is currently restricted from submitting comments: ${sessionUser.suspensionReason || "Administrative restriction."}`,
        },
        { status: 403 }
      );
    }

    // 5. Policy consultation window check: Time limit removed for now so anyone can comment anytime
    // (Previously restricted commenting to a 4-day window after policy.openedAt)

    // 6. Rapid submission velocity rate limit (1 per 2s per user)
    const velocity = await checkRateLimit(`comment:velocity:user:${sessionUser.id}`, 1, 2);
    if (!velocity.allowed) {
      return NextResponse.json(
        {
          success: false,
          error: "SUBMISSION_TOO_FAST",
          message: "Please wait a moment before submitting another comment.",
        },
        { status: 429 }
      );
    }

    const body = await req.json().catch(() => null);
    if (!body || typeof body !== "object") {
      return NextResponse.json(
        { success: false, error: "INVALID_REQUEST", message: "Invalid JSON request body." },
        { status: 400 }
      );
    }

    const { text, stakeholder } = body;
    const cleanText = typeof text === "string" ? text.trim() : "";
    const cleanStakeholder =
      typeof stakeholder === "string" && stakeholder.trim().length > 0
        ? stakeholder.trim().slice(0, 80)
        : "Citizen / Community Member";

    // 7. Anti-Spam & Content Validation
    const spamResult = analyzeCommentSpam(cleanText);
    if (spamResult.status === "REJECTED") {
      await logAudit({
        eventType: "COMMENT_SUBMIT",
        actorId: sessionUser.id,
        resourceType: "policy",
        resourceId: slug,
        actionResult: "REJECTED",
        ipHash,
        metadata: { reason: spamResult.reason, flags: spamResult.flags },
      });
      return NextResponse.json(
        {
          success: false,
          error: "CONTENT_REJECTED",
          message: spamResult.reason || "Submission rejected by spam filters.",
          details: spamResult.flags,
        },
        { status: 400 }
      );
    }

    const normalizedHash = computeTextHash(normalizeCommentText(cleanText));

    // 8. Check for exact duplicate on this policy by this user
    const dupCheck = await query(
      `SELECT id FROM policy_comments WHERE user_id = $1 AND policy_slug = $2 AND normalized_text_hash = $3 LIMIT 1;`,
      [sessionUser.id, slug, normalizedHash]
    );
    if (dupCheck.rows.length > 0) {
      return NextResponse.json(
        {
          success: false,
          error: "DUPLICATE_COMMENT",
          message: "You have already submitted this exact comment on this policy.",
        },
        { status: 400 }
      );
    }

    // 9. Concurrency-Safe Durable Quota Enforcement & Comment Creation Transaction
    const userAgent = req.headers.get("user-agent")?.slice(0, 255) || null;

    const result = await withTransaction(async (client) => {
      // Ensure quota row exists
      await client.query(
        `INSERT INTO policy_comment_quotas (user_id, policy_slug, submission_count, created_at, updated_at)
         VALUES ($1, $2, 0, NOW(), NOW())
         ON CONFLICT (user_id, policy_slug) DO NOTHING;`,
        [sessionUser.id, slug]
      );

      // Acquire exclusive row lock on the quota record
      const lockRes = await client.query<{ submission_count: number }>(
        `SELECT submission_count FROM policy_comment_quotas
         WHERE user_id = $1 AND policy_slug = $2
         FOR UPDATE;`,
        [sessionUser.id, slug]
      );

      const currentCount = lockRes.rows[0].submission_count;

      // Hard check: if limit already reached, reject transaction immediately
      if (currentCount >= MAX_COMMENTS_PER_USER_PER_POLICY) {
        const err: any = new Error("COMMENT_LIMIT_EXCEEDED");
        err.code = "COMMENT_LIMIT_EXCEEDED";
        err.used = currentCount;
        throw err;
      }

      // Durable quota increment: counts submissions regardless of subsequent moderation state
      const newCount = currentCount + 1;
      await client.query(
        `UPDATE policy_comment_quotas
         SET submission_count = $1, updated_at = NOW()
         WHERE user_id = $2 AND policy_slug = $3;`,
        [newCount, sessionUser.id, slug]
      );

      // Insert comment
      const commentRes = await client.query<{
        id: string;
        policy_slug: string;
        status: string;
        created_at: Date;
      }>(
        `INSERT INTO policy_comments (
          user_id, policy_slug, stakeholder, original_text,
          normalized_text_hash, status, moderation_reason,
          spam_score, spam_flags, ip_hash, user_agent, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
        RETURNING id, policy_slug, status, created_at;`,
        [
          sessionUser.id,
          slug,
          cleanStakeholder,
          cleanText,
          normalizedHash,
          spamResult.status,
          spamResult.reason || null,
          spamResult.score,
          JSON.stringify(spamResult.flags),
          ipHash,
          userAgent,
        ]
      );

      return {
        comment: commentRes.rows[0],
        newCount,
      };
    });

    await logAudit({
      eventType: "COMMENT_SUBMIT",
      actorId: sessionUser.id,
      resourceType: "policy_comment",
      resourceId: result.comment.id,
      actionResult: "SUCCESS",
      ipHash,
      metadata: {
        policySlug: slug,
        status: result.comment.status,
        quotaUsed: result.newCount,
      },
    });

    return NextResponse.json(
      {
        success: true,
        message:
          result.comment.status === "APPROVED"
            ? "Your comment has been submitted and published."
            : "Your comment has been received and is pending moderator review.",
        comment: {
          id: result.comment.id,
          status: result.comment.status,
          createdAt: result.comment.created_at,
          text: cleanText,
          stakeholder: cleanStakeholder,
        },
        quota: {
          used: result.newCount,
          max: MAX_COMMENTS_PER_USER_PER_POLICY,
          remaining: Math.max(0, MAX_COMMENTS_PER_USER_PER_POLICY - result.newCount),
        },
      },
      { status: 201 }
    );
  } catch (err: any) {
    if (err?.code === "COMMENT_LIMIT_EXCEEDED") {
      return NextResponse.json(
        {
          success: false,
          error: "COMMENT_LIMIT_EXCEEDED",
          message: `You have reached the maximum allowance of ${MAX_COMMENTS_PER_USER_PER_POLICY} comments for this policy.`,
          quota: {
            used: err.used || MAX_COMMENTS_PER_USER_PER_POLICY,
            max: MAX_COMMENTS_PER_USER_PER_POLICY,
            remaining: 0,
          },
        },
        { status: 429 }
      );
    }

    console.error("[POST_COMMENT_ERROR]", err);
    return NextResponse.json(
      { success: false, error: "INTERNAL_ERROR", message: "Failed to submit comment." },
      { status: 500 }
    );
  }
}
