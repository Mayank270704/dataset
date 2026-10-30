"use client";

import { useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { UserPlus, AlertCircle, CheckCircle2, Mail } from "lucide-react";

function RegisterForm() {
  const searchParams = useSearchParams();
  const redirectUrl = searchParams.get("redirect") || "/";

  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [registeredSuccess, setRegisteredSuccess] = useState(false);

  const hasLength = password.length >= 8;
  const hasLower = /[a-z]/.test(password);
  const hasUpper = /[A-Z]/.test(password);
  const hasNumber = /[0-9]/.test(password);
  const hasSymbol = /[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?`~]/.test(password);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ displayName, email, password }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok || !data.success) {
        setError(data.message || "Registration failed. Please check your details.");
        return;
      }

      setRegisteredSuccess(true);
    } catch {
      setError("An unexpected network error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (registeredSuccess) {
    return (
      <main className="mx-auto max-w-md px-4 py-12 pb-24">
        <section className="rounded-xl border border-emerald-200 bg-white p-6 sm:p-8 shadow-xs text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-emerald-50 border border-emerald-200 flex items-center justify-center mx-auto text-emerald-600">
            <Mail size={24} />
          </div>
          <h1 className="font-display text-2xl font-bold text-[#171717]">Verify your email</h1>
          <p className="text-sm text-[#475569] leading-relaxed">
            We have sent a verification link to <strong className="text-[#171717]">{email}</strong>.
            Please click the link in your email to activate your account and enable commenting.
          </p>
          <div className="pt-4 border-t border-slate-100 flex justify-center gap-3">
            <Link
              href={`/login${redirectUrl !== "/" ? `?redirect=${encodeURIComponent(redirectUrl)}` : ""}`}
              className="inline-flex rounded-lg bg-[#143454] px-4 py-2 text-xs font-semibold text-white hover:bg-[#0e2a47]"
            >
              Go to Sign In
            </Link>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-md px-4 py-12 pb-24">
      <section className="rounded-xl border border-slate-200 bg-white p-6 sm:p-8 shadow-xs">
        <div className="flex items-center gap-2.5 text-[#143454] pb-2 border-b border-slate-100">
          <div className="w-8 h-8 rounded-lg bg-[#e8eef5] flex items-center justify-center">
            <UserPlus size={18} className="text-[#143454]" />
          </div>
          <div>
            <h1 className="font-display text-xl font-bold text-[#171717]">Create Citizen Account</h1>
            <p className="text-xs text-[#666666]">Participate in verified civic policy consultations</p>
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-800 flex items-start gap-2">
            <AlertCircle size={15} className="shrink-0 mt-0.5 text-red-600" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-[#334155] mb-1">
              Full Name / Display Name
            </label>
            <input
              type="text"
              required
              minLength={2}
              maxLength={80}
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="e.g. Ramesh Sharma"
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-[#171717] outline-none focus:border-[#143454] focus:ring-1 focus:ring-[#143454]"
            />
          </div>

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

            {/* Password strength checklist */}
            <div className="mt-2.5 rounded-lg bg-slate-50 border border-slate-200 p-2.5 space-y-1 text-[11px] text-[#64748b]">
              <span className="font-semibold block text-[#334155] mb-1">Password Requirements:</span>
              <div className="grid grid-cols-2 gap-1">
                <span className={`flex items-center gap-1 ${hasLength ? "text-emerald-700 font-medium" : ""}`}>
                  <CheckCircle2 size={12} className={hasLength ? "text-emerald-600" : "text-slate-300"} />
                  8+ characters
                </span>
                <span className={`flex items-center gap-1 ${hasLower ? "text-emerald-700 font-medium" : ""}`}>
                  <CheckCircle2 size={12} className={hasLower ? "text-emerald-600" : "text-slate-300"} />
                  Lowercase letter
                </span>
                <span className={`flex items-center gap-1 ${hasUpper ? "text-emerald-700 font-medium" : ""}`}>
                  <CheckCircle2 size={12} className={hasUpper ? "text-emerald-600" : "text-slate-300"} />
                  Uppercase letter
                </span>
                <span className={`flex items-center gap-1 ${hasNumber ? "text-emerald-700 font-medium" : ""}`}>
                  <CheckCircle2 size={12} className={hasNumber ? "text-emerald-600" : "text-slate-300"} />
                  Number (0–9)
                </span>
                <span className={`flex items-center gap-1 col-span-2 ${hasSymbol ? "text-emerald-700 font-medium" : ""}`}>
                  <CheckCircle2 size={12} className={hasSymbol ? "text-emerald-600" : "text-slate-300"} />
                  Special symbol (!@#$%^&*)
                </span>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 inline-flex items-center justify-center gap-2 rounded-lg bg-[#E87518] px-4 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-[#d4660f] disabled:opacity-60 transition-colors"
          >
            <span>{loading ? "Creating Account…" : "Create Citizen Account"}</span>
          </button>
        </form>

        <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
          <span>Already registered?</span>
          <Link
            href={`/login${redirectUrl !== "/" ? `?redirect=${encodeURIComponent(redirectUrl)}` : ""}`}
            className="font-semibold text-[#143454] hover:underline"
          >
            Sign in here
          </Link>
        </div>
      </section>
    </main>
  );
}

export default function RegisterPage() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-sm text-slate-500">Loading registration...</div>}>
      <RegisterForm />
    </Suspense>
  );
}
