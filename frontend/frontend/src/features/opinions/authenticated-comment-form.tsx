"use client";

import React, { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Send,
  AlertTriangle,
  CheckCircle,
  Loader2,
  MessageSquare,
  MailCheck,
  ShieldOff,
  LogIn,
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";

interface AuthenticatedCommentFormProps {
  policySlug: string;
  policyTitle: string;
}

interface UserQuota {
  used: number;
  max: number;
  remaining: number;
}

const STAKEHOLDER_OPTIONS = [
  "Citizen / Community Member",
  "Small Business / MSME Representative",
  "Farmer / Agricultural Producer",
  "Academic / Policy Researcher",
  "Public Sector / Healthcare Worker",
  "Non-Profit / Civil Society Organization",
];

export function AuthenticatedCommentForm({
  policySlug,
  policyTitle,
}: AuthenticatedCommentFormProps) {
  const { user, loading } = useAuth();

  const [stakeholder, setStakeholder] = useState(STAKEHOLDER_OPTIONS[0]);
  const [commentText, setCommentText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [quota, setQuota] = useState<UserQuota | null>(null);
  const [quotaLoading, setQuotaLoading] = useState(false);
  const [result, setResult] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);
  const [resendingVerification, setResendingVerification] = useState(false);
  const [resendResult, setResendResult] = useState<string | null>(null);

  // Fetch quota when user is logged in
  const fetchQuota = useCallback(async () => {
    if (!user) return;
    setQuotaLoading(true);
    try {
      const res = await fetch(`/api/policies/${encodeURIComponent(policySlug)}/comments`, {
        cache: "no-store",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.userQuota) {
          setQuota(data.userQuota);
        }
      }
    } catch {
      // Non-critical: quota display degrades gracefully
    } finally {
      setQuotaLoading(false);
    }
  }, [user, policySlug]);

  useEffect(() => {
    fetchQuota();
  }, [fetchQuota]);

  const handleResendVerification = async () => {
    setResendingVerification(true);
    setResendResult(null);
    try {
      const res = await fetch("/api/auth/resend-verification", { method: "POST" });
      const data = await res.json();
      setResendResult(
        res.ok
          ? "Verification email sent. Please check your inbox."
          : data.message || "Failed to send verification email."
      );
    } catch {
      setResendResult("Failed to send verification email. Please try again.");
    } finally {
      setResendingVerification(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = commentText.trim();
    if (trimmed.length < 10) {
      setResult({ type: "error", message: "Please write at least 10 characters." });
      return;
    }

    setSubmitting(true);
    setResult(null);

    try {
      const res = await fetch(`/api/policies/${encodeURIComponent(policySlug)}/comments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed, stakeholder }),
      });

      const data = await res.json();

      if (res.ok && data.success) {
        setResult({
          type: "success",
          message: data.message || "Your comment has been submitted.",
        });
        setCommentText("");
        if (data.quota) setQuota(data.quota);
      } else {
        // Map specific error codes to friendly messages
        const errorMessages: Record<string, string> = {
          COMMENT_LIMIT_EXCEEDED: `You have used all ${data.quota?.max ?? 5} comments for this policy.`,
          SUBMISSION_TOO_FAST: "Please wait a few seconds before submitting another comment.",
          DUPLICATE_COMMENT: "You have already submitted this exact comment on this policy.",
          CONTENT_REJECTED: data.message || "Your comment was rejected by the content filter.",
          UNAUTHENTICATED: "You must be signed in to submit comments.",
          EMAIL_NOT_VERIFIED: "Please verify your email address before submitting.",
          ACCOUNT_RESTRICTED: data.message || "Your account is currently restricted.",
          CONSULTATION_CLOSED: "The commenting window for this policy has closed.",
        };
        setResult({
          type: "error",
          message:
            errorMessages[data.error] ||
            data.message ||
            "Failed to submit comment. Please try again.",
        });
        // Refresh quota on limit-exceeded
        if (data.error === "COMMENT_LIMIT_EXCEEDED" && data.quota) {
          setQuota(data.quota);
        }
      }
    } catch {
      setResult({ type: "error", message: "Network error. Please check your connection and try again." });
    } finally {
      setSubmitting(false);
    }
  };

  // ── Loading state ──────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-2xs flex items-center gap-3 text-[14px] text-[#4b5563]">
        <Loader2 size={16} className="animate-spin text-[#143454]" />
        <span>Loading…</span>
      </div>
    );
  }

  // ── Not logged in ──────────────────────────────────────────────────────────
  if (!user) {
    const currentPath = typeof window !== "undefined" ? window.location.pathname : `/policy/${policySlug}`;
    return (
      <div className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-2xs space-y-3">
        <div className="flex items-center gap-2">
          <MessageSquare size={18} className="text-[#143454]" />
          <h2 className="font-display text-[18px] font-bold text-[#171717]">Share Your Opinion</h2>
        </div>
        <p className="text-[13px] text-[#4b5563]">
          Sign in to submit your official opinion on &ldquo;{policyTitle}&rdquo; and have your voice
          counted.
        </p>
        <Link
          href={`/login?redirect=${encodeURIComponent(currentPath)}`}
          className="inline-flex items-center gap-1.5 rounded-lg bg-[#143454] px-4 py-2.5 text-[13px] font-bold text-white shadow-xs hover:bg-[#0e2a47] transition-colors"
        >
          <LogIn size={14} />
          Login to Comment
        </Link>
        <p className="text-[11px] text-[#6b7280]">
          Don&apos;t have an account?{" "}
          <Link href="/register" className="text-[#143454] font-semibold underline">
            Register here
          </Link>
        </p>
      </div>
    );
  }

  // ── Email not verified ─────────────────────────────────────────────────────
  if (!user.isEmailVerified) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 sm:p-6 space-y-3">
        <div className="flex items-center gap-2">
          <MailCheck size={18} className="text-amber-600" />
          <h2 className="font-display text-[17px] font-bold text-amber-900">
            Email Verification Required
          </h2>
        </div>
        <p className="text-[13px] text-amber-800">
          You must verify your email address before submitting comments. Please check your inbox
          for the verification link sent to <strong>{user.email}</strong>.
        </p>
        {resendResult && (
          <p className="text-[12px] text-amber-900 bg-amber-100 rounded-md px-3 py-2">
            {resendResult}
          </p>
        )}
        <button
          onClick={handleResendVerification}
          disabled={resendingVerification}
          className="inline-flex items-center gap-1.5 rounded-lg border border-amber-400 bg-white px-4 py-2 text-[13px] font-semibold text-amber-800 hover:bg-amber-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {resendingVerification ? <Loader2 size={13} className="animate-spin" /> : <MailCheck size={13} />}
          {resendingVerification ? "Sending…" : "Resend Verification Email"}
        </button>
      </div>
    );
  }

  // ── Account restricted / suspended ────────────────────────────────────────
  if (user.status === "SUSPENDED" || user.status === "RESTRICTED") {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-5 sm:p-6 space-y-2">
        <div className="flex items-center gap-2">
          <ShieldOff size={18} className="text-red-600" />
          <h2 className="font-display text-[17px] font-bold text-red-900">
            {user.status === "SUSPENDED" ? "Account Suspended" : "Account Restricted"}
          </h2>
        </div>
        <p className="text-[13px] text-red-800">
          {user.status === "SUSPENDED"
            ? "Your account has been suspended and you cannot submit comments. Please contact support."
            : "Your account is currently restricted from submitting comments. Please contact support."}
        </p>
      </div>
    );
  }

  // ── Verified, active user — show comment form ──────────────────────────────
  const isQuotaFull = quota !== null && quota.remaining <= 0;
  const canSubmit = !submitting && !isQuotaFull && commentText.trim().length >= 10;

  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-2xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-3 border-b border-[#f1f3f5]">
        <div>
          <h2 className="font-display text-[19px] sm:text-[21px] font-bold text-[#171717] flex items-center gap-2">
            <MessageSquare size={19} className="text-[#143454]" />
            <span>Submit Your Opinion</span>
          </h2>
          <p className="text-[13px] sm:text-[14px] text-[#4b5563] mt-0.5">
            Officially comment on &ldquo;{policyTitle}&rdquo;. Comments are stored and moderated.
          </p>
        </div>

        {/* Quota indicator */}
        <div className="shrink-0 flex flex-col items-end gap-0.5">
          {quotaLoading ? (
            <span className="text-[12px] text-[#6b7280]">Loading quota…</span>
          ) : quota ? (
            <>
              <span
                className={`text-[12px] px-2.5 py-1 rounded-md border font-semibold ${
                  isQuotaFull
                    ? "text-red-700 bg-red-50 border-red-200"
                    : "text-[#143454] bg-[#f0f4f8] border-[#d6e0ea]"
                }`}
              >
                {quota.used} of {quota.max} comments used
              </span>
              {isQuotaFull && (
                <span className="text-[11px] text-red-600">Limit reached</span>
              )}
            </>
          ) : null}
        </div>
      </div>

      {/* Quota exhausted message */}
      {isQuotaFull && (
        <div className="rounded-lg border border-[#fca5a5] bg-[#fef2f2] p-3.5 text-[13px] text-[#991b1b] flex items-start gap-2">
          <AlertTriangle size={15} className="shrink-0 mt-0.5" />
          <span>
            You have used all {quota?.max ?? 5} comments allowed for this policy. Thank you for
            your participation.
          </span>
        </div>
      )}

      {/* Success / error feedback */}
      {result && (
        <div
          className={`rounded-lg border p-3.5 text-[13px] flex items-start gap-2 ${
            result.type === "success"
              ? "border-green-200 bg-green-50 text-green-800"
              : "border-[#fca5a5] bg-[#fef2f2] text-[#991b1b]"
          }`}
        >
          {result.type === "success" ? (
            <CheckCircle size={15} className="shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle size={15} className="shrink-0 mt-0.5" />
          )}
          <span>{result.message}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Stakeholder select */}
        <div className="space-y-1.5">
          <label
            htmlFor="auth-stakeholder-select"
            className="block text-[13px] sm:text-[14px] font-semibold text-[#171717]"
          >
            Stakeholder / Community Perspective
          </label>
          <select
            id="auth-stakeholder-select"
            value={stakeholder}
            onChange={(e) => setStakeholder(e.target.value)}
            disabled={isQuotaFull || submitting}
            className="w-full rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-2.5 text-[13px] text-[#171717] outline-none focus:border-[#143454] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {STAKEHOLDER_OPTIONS.map((opt) => (
              <option key={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* Comment textarea */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[13px]">
            <label htmlFor="auth-comment-text" className="font-semibold text-[#171717]">
              Your Comment
            </label>
            <span className="text-[12px] text-[#6b7280]">
              {commentText.length} / 2,000 characters (min. 10)
            </span>
          </div>
          <textarea
            id="auth-comment-text"
            rows={4}
            maxLength={2000}
            value={commentText}
            onChange={(e) => setCommentText(e.target.value)}
            disabled={isQuotaFull || submitting}
            placeholder={
              isQuotaFull
                ? "You have reached the comment limit for this policy."
                : "Write your opinion, concern, or suggested improvement…"
            }
            className="w-full rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-3 text-[13px] sm:text-[14px] text-[#171717] outline-none focus:border-[#E87518] focus:bg-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed leading-relaxed"
          />
        </div>

        <div className="flex items-center justify-end pt-1">
          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#143454] px-4 py-2.5 text-[13px] font-bold text-white shadow-xs hover:bg-[#0e2a47] disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            {submitting ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Send size={14} />
            )}
            <span>{submitting ? "Submitting…" : "Submit Comment"}</span>
          </button>
        </div>
      </form>
    </section>
  );
}
