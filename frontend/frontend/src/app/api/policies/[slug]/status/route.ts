import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import { officialPolicies } from "@/features/feed/demo-data";
import {
  isValidSlugFormat,
  resolveSafeDataFilePath,
  SecurityError,
  BENCHMARK_PILOT_SLUG
} from "@/lib/security";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params;

  // 1. Validate slug format (reject path traversal, special characters)
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

  // 2. Validate known policy existence
  const policy = officialPolicies.find((p) => p.slug === slug);
  if (!policy) {
    return NextResponse.json(
      {
        success: false,
        error: "POLICY_NOT_FOUND",
        message: "Policy not found."
      },
      { status: 404 }
    );
  }

  // 3. Resolve safe paths inside submissions, jobs, and analysis directories
  let submissionsFile: string;
  let jobFile: string;
  let analysisFile: string;

  try {
    submissionsFile = resolveSafeDataFilePath("submissions", slug);
    jobFile = resolveSafeDataFilePath("jobs", slug);
    analysisFile = resolveSafeDataFilePath("analysis", slug);
  } catch (secErr) {
    const msg = secErr instanceof SecurityError ? secErr.message : "Access denied";
    return NextResponse.json(
      { success: false, error: "SECURITY_VIOLATION", message: msg },
      { status: 400 }
    );
  }

  let totalSubmissions = 0;
  if (fs.existsSync(submissionsFile)) {
    try {
      const subs = JSON.parse(fs.readFileSync(submissionsFile, "utf-8"));
      totalSubmissions = Array.isArray(subs) ? subs.length : 0;
    } catch {
      totalSubmissions = 0;
    }
  }

  let jobStatus: { status: string; job_id?: string; completed_at?: string; error_message?: string } = {
    status: policy.consultationStatus || "open"
  };

  if (fs.existsSync(jobFile)) {
    try {
      jobStatus = JSON.parse(fs.readFileSync(jobFile, "utf-8"));
    } catch {
      // fallback to default
    }
  }

  // Time limit removed for now: anyone can comment at anytime
  const isCommentingOpen = true;

  const hasAnalysis = fs.existsSync(analysisFile) || slug === BENCHMARK_PILOT_SLUG;

  return NextResponse.json({
    slug,
    title: policy.title,
    commentingDeadline: policy.commentingDeadline,
    isCommentingOpen,
    totalSubmissions,
    jobStatus: jobStatus.status,
    jobId: jobStatus.job_id,
    hasAnalysis,
    lastCompletedAt: jobStatus.completed_at
  });
}
