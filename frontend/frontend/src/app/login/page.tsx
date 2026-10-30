"use client";

import { useState, useRef, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { LogIn, AlertCircle, ArrowRight, Mail, CheckCircle2 } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirectUrl = searchParams.get("redirect") || "/";
  const { refresh } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [unverifiedEmail, setUnverifiedEmail] = useState("");
  const [resendStatus, setResendStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [resendFeedback, setResendFeedback] = useState("");
  const isSubmittingRef = useRef(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmittingRef.current || loading) return;
    isSubmittingRef.current = true;
    setError("");
    setUnverifiedEmail("");
    setResendStatus("idle");
    setResendFeedback("");
    setLoading(true);

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim(), password }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok || !data.success) {
        if (data.error === "EMAIL_NOT_VERIFIED") {
          setUnverifiedEmail(email.trim());
          setError(data.message || "Your email address is not verified.");
        } else {
          setError(data.message || "Invalid email or password.");
        }
        return;
      }

      await refresh();
      router.push(redirectUrl);
    } catch {
      setError("An unexpected network error occurred. Please try again.");
    } finally {
      isSubmittingRef.current = false;
      setLoading(false);
    }
  };

  const handleQuickResend = async () => {
    if (!unverifiedEmail || resendStatus === "sending") return;
    setResendStatus("sending");
    try {
      const res = await fetch("/api/auth/resend-verification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: unverifiedEmail }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        setResendStatus("sent");
        setResendFeedback("Verification email dispatched! Please check your inbox.");
      } else {
        setResendStatus("error");
        setResendFeedback(data.message || "Could not send verification email.");
      }
    } catch {
      setResendStatus("error");
      setResendFeedback("Network error. Please try again.");
    }
  };

  return (
    <main className="mx-auto max-w-md px-4 py-12 pb-24">
      <section className="rounded-xl border border-slate-200 bg-white p-6 sm:p-8 shadow-xs">
        <div className="flex items-center gap-2.5 text-[#143454] pb-2 border-b border-slate-100">
          <div className="w-8 h-8 rounded-lg bg-[#e8eef5] flex items-center justify-center">
            <LogIn size={18} className="text-[#143454]" />
          </div>
          <div>
            <h1 className="font-display text-xl font-bold text-[#171717]">Sign in to PolicyLens</h1>
            <p className="text-xs text-[#666666]">Access verified civic feedback &amp; policy discussions</p>
          </div>
        </div>

        {unverifiedEmail ? (
          <div className="mt-4 rounded-lg bg-amber-50 border border-amber-200 p-3.5 text-xs text-amber-900 space-y-2">
            <div className="flex items-start gap-2">
              <Mail size={16} className="shrink-0 mt-0.5 text-amber-600" />
              <div>
                <strong className="block font-semibold">Email Verification Required</strong>
                <p className="text-amber-800 mt-0.5">{error}</p>
              </div>
            </div>
            {resendFeedback && (
              <div
                className={`p-2 rounded text-xs flex items-center gap-1.5 ${
                  resendStatus === "sent"
                    ? "bg-emerald-100/70 text-emerald-900 border border-emerald-200"
                    : "bg-red-100/70 text-red-900 border border-red-200"
                }`}
              >
                {resendStatus === "sent" ? (
                  <CheckCircle2 size={13} className="text-emerald-700 shrink-0" />
                ) : (
                  <AlertCircle size={13} className="text-red-700 shrink-0" />
                )}
                <span>{resendFeedback}</span>
              </div>
            )}
            <div className="pt-1 flex items-center gap-2">
              <button
                type="button"
                onClick={handleQuickResend}
                disabled={resendStatus === "sending"}
                className="inline-flex items-center gap-1.5 rounded-md bg-[#143454] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[#0e2a47] disabled:opacity-50"
              >
                {resendStatus === "sending" ? "Sending…" : "Resend Verification Email"}
              </button>
              <Link
                href={`/verify-email?email=${encodeURIComponent(unverifiedEmail)}`}
                className="text-xs text-[#143454] font-medium underline"
              >
                Enter token manually
              </Link>
            </div>
          </div>
        ) : error ? (
          <div className="mt-4 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-800 flex items-start gap-2">
            <AlertCircle size={15} className="shrink-0 mt-0.5 text-red-600" />
            <span>{error}</span>
          </div>
        ) : null}

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-[#334155] mb-1">
              Email Address
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@example.com"
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-[#171717] outline-none focus:border-[#143454] focus:ring-1 focus:ring-[#143454]"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-[#334155] mb-1">
              Password
            </label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-[#171717] outline-none focus:border-[#143454] focus:ring-1 focus:ring-[#143454]"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 inline-flex items-center justify-center gap-2 rounded-lg bg-[#143454] px-4 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-[#0e2a47] disabled:opacity-60 transition-colors"
          >
            <span>{loading ? "Signing in…" : "Sign In"}</span>
            {!loading && <ArrowRight size={15} />}
          </button>
        </form>

        <div className="mt-6 pt-4 border-t border-slate-100 flex flex-col gap-2 text-xs text-slate-500">
          <div className="flex items-center justify-between">
            <span>Don&apos;t have an account?</span>
            <Link
              href={`/register${redirectUrl !== "/" ? `?redirect=${encodeURIComponent(redirectUrl)}` : ""}`}
              className="font-semibold text-[#E87518] hover:underline"
            >
              Register here
            </Link>
          </div>
          <div className="flex items-center justify-between pt-1">
            <span>Need to verify your email?</span>
            <Link
              href={`/verify-email${email ? `?email=${encodeURIComponent(email)}` : ""}`}
              className="font-semibold text-[#143454] hover:underline"
            >
              Request verification email
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-sm text-slate-500">Loading sign in...</div>}>
      <LoginForm />
    </Suspense>
  );
}
