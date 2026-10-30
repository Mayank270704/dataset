"use client";

import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { CheckCircle2, AlertCircle, Loader2, Mail } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

function VerifyEmailContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const emailQueryParam = searchParams.get("email") || "";
  const { user, refresh } = useAuth();

  const [loading, setLoading] = useState(Boolean(token));
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const [emailInput, setEmailInput] = useState(emailQueryParam || user?.email || "");
  const [resending, setResending] = useState(false);
  const [resendMessage, setResendMessage] = useState("");
  const [resendError, setResendError] = useState("");

  useEffect(() => {
    if (user?.email && !emailInput) {
      setEmailInput(user.email);
    }
  }, [user?.email, emailInput]);

  useEffect(() => {
    if (!token) return;

    let mounted = true;
    async function verify() {
      try {
        const res = await fetch("/api/auth/verify-email", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token }),
        });
        const data = await res.json().catch(() => ({}));

        if (!mounted) return;

        if (res.ok && data.success) {
          setSuccess(true);
          await refresh();
          setTimeout(() => {
            router.push("/login?verified=true");
          }, 3500);
        } else {
          setError(data.message || "Email verification failed or token expired.");
        }
      } catch {
        if (mounted) setError("A network error occurred while verifying your email.");
      } finally {
        if (mounted) setLoading(false);
      }
    }

    verify();
    return () => {
      mounted = false;
    };
  }, [token, refresh, router]);

  const handleResend = async (e: React.FormEvent) => {
    e.preventDefault();
    const targetEmail = (emailInput || user?.email || "").trim();
    if (!targetEmail) {
      setResendError("Please enter your registered email address.");
      return;
    }

    setResending(true);
    setResendMessage("");
    setResendError("");

    try {
      const res = await fetch("/api/auth/resend-verification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: targetEmail }),
      });
      const data = await res.json().catch(() => ({}));

      if (res.ok && data.success) {
        setResendMessage(
          data.message || "A new verification link has been sent. Please check your inbox."
        );
      } else {
        setResendError(
          data.message || "Failed to request verification email. Please try again."
        );
      }
    } catch {
      setResendError("Failed to request verification email. Please try again.");
    } finally {
      setResending(false);
    }
  };

  return (
    <main className="mx-auto max-w-md px-4 py-12 pb-24">
      <section className="rounded-xl border border-slate-200 bg-white p-6 sm:p-8 shadow-xs text-center space-y-4">
        {loading ? (
          <div className="py-8 space-y-3">
            <Loader2 size={36} className="animate-spin text-[#143454] mx-auto" />
            <h1 className="font-display text-xl font-bold text-[#171717]">Verifying your email…</h1>
            <p className="text-xs text-slate-500">Validating your single-use cryptographic token</p>
          </div>
        ) : success ? (
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-full bg-emerald-50 border border-emerald-200 flex items-center justify-center mx-auto text-emerald-600">
              <CheckCircle2 size={26} />
            </div>
            <h1 className="font-display text-2xl font-bold text-[#171717]">Email Verified!</h1>
            <p className="text-sm text-[#475569] leading-relaxed">
              Your email address has been successfully verified. You now have full access to share comments and participate in policy consultations.
            </p>
            <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row justify-center gap-3">
              <Link
                href="/login?verified=true"
                className="inline-flex items-center justify-center rounded-lg bg-[#143454] px-4 py-2 text-xs font-semibold text-white hover:bg-[#0e2a47]"
              >
                Sign In to Your Account
              </Link>
              <Link
                href="/"
                className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                Return to Home Feed
              </Link>
            </div>
            <p className="text-[11px] text-slate-400">Redirecting to login in a few seconds…</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-full bg-amber-50 border border-amber-200 flex items-center justify-center mx-auto text-amber-600">
              <Mail size={24} />
            </div>
            <h1 className="font-display text-xl font-bold text-[#171717]">Email Verification</h1>
            {error && (
              <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-800 flex items-start gap-2 text-left">
                <AlertCircle size={15} className="shrink-0 mt-0.5 text-red-600" />
                <span>{error}</span>
              </div>
            )}
            <p className="text-sm text-[#475569] leading-relaxed">
              Enter your email address below to request or resend your secure verification link.
            </p>
            {resendMessage && (
              <div className="rounded-lg bg-emerald-50 border border-emerald-200 p-3 text-xs text-emerald-800 flex items-start gap-2 text-left">
                <CheckCircle2 size={15} className="shrink-0 mt-0.5 text-emerald-600" />
                <span>{resendMessage}</span>
              </div>
            )}
            {resendError && (
              <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-800 flex items-start gap-2 text-left">
                <AlertCircle size={15} className="shrink-0 mt-0.5 text-red-600" />
                <span>{resendError}</span>
              </div>
            )}
            <form onSubmit={handleResend} className="pt-2 flex flex-col gap-3 text-left">
              <div>
                <label className="block text-xs font-semibold text-[#334155] mb-1">
                  Email Address
                </label>
                <input
                  type="email"
                  required
                  value={emailInput}
                  onChange={(e) => setEmailInput(e.target.value)}
                  placeholder="name@example.com"
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-[#171717] outline-none focus:border-[#143454] focus:ring-1 focus:ring-[#143454]"
                />
              </div>
              <button
                type="submit"
                disabled={resending}
                className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-[#143454] px-4 py-2.5 text-xs font-semibold text-white hover:bg-[#0e2a47] disabled:opacity-50 transition-colors"
              >
                {resending ? "Sending Verification Link…" : "Send Verification Email"}
              </button>
              <div className="pt-2 text-center flex flex-col gap-1.5 text-xs text-slate-500">
                <Link href="/login" className="font-semibold text-[#143454] hover:underline">
                  Sign in to your account
                </Link>
                <Link href="/" className="hover:underline">
                  Return to Home
                </Link>
              </div>
            </form>
          </div>
        )}
      </section>
    </main>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-sm text-slate-500">Loading verification...</div>}>
      <VerifyEmailContent />
    </Suspense>
  );
}

