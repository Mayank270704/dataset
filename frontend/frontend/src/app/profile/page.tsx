"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import {
  User,
  ShieldCheck,
  Mail,
  AlertTriangle,
  LogOut,
  Bookmark,
  History,
  Palette,
  CheckCircle2,
  LogIn,
  UserPlus,
} from "lucide-react";
import { useTheme } from "@/components/theme/theme-provider";
import { readHistory, readSaved, usePolicyLibrary } from "@/lib/policy-library";
import { useAuth } from "@/lib/auth-context";

export default function ProfilePage() {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const [resending, setResending] = useState(false);
  const [resendStatus, setResendStatus] = useState("");

  const countsReader = useCallback(() => [readSaved().length, readHistory().length] as const, []);
  const [savedCount, historyCount] = usePolicyLibrary(countsReader, [0, 0]);

  const handleResendVerification = async () => {
    if (!user) return;
    setResending(true);
    setResendStatus("");
    try {
      const res = await fetch("/api/auth/resend-verification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: user.email }),
      });
      const data = await res.json().catch(() => ({}));
      setResendStatus(data.message || "Verification email sent.");
    } catch {
      setResendStatus("Failed to send verification email. Please try again.");
    } finally {
      setResending(false);
    }
  };

  return (
    <main className="mx-auto max-w-3xl px-4 py-8 pb-24 space-y-6">
      <div>
        <h1 className="font-display text-3xl font-extrabold text-[#171717]">Account &amp; Preferences</h1>
        <p className="mt-1.5 text-sm text-[#4b5563]">
          Manage your PolicyLens civic participation profile, credentials, and browser preferences.
        </p>
      </div>

      {/* User Authentication Card */}
      {user ? (
        <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-xs space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#f1f3f5]">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-[#143454] text-white flex items-center justify-center font-bold text-lg">
                {user.displayName.charAt(0).toUpperCase()}
              </div>
              <div>
                <h2 className="text-lg font-bold text-[#171717]">{user.displayName}</h2>
                <p className="text-xs text-[#666666] flex items-center gap-1.5 mt-0.5">
                  <Mail size={12} className="text-[#9ca3af]" />
                  <span>{user.email}</span>
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider px-2.5 py-1 rounded bg-[#e8eef5] text-[#143454]">
                {user.role}
              </span>
              <button
                type="button"
                onClick={logout}
                className="inline-flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3 py-1 text-xs font-semibold text-red-700 hover:bg-red-100"
              >
                <LogOut size={13} />
                <span>Sign Out</span>
              </button>
            </div>
          </div>

          {/* Verification Status */}
          <div className="rounded-lg p-3.5 border text-xs flex flex-wrap items-center justify-between gap-2.5 bg-[#f8fafc] border-[#e2e8f0]">
            <div className="flex items-center gap-2">
              {user.isEmailVerified ? (
                <>
                  <CheckCircle2 size={16} className="text-[#16834A]" />
                  <span className="font-semibold text-[#16834A]">Email Verified</span>
                  <span className="text-slate-400">·</span>
                  <span className="text-slate-600">Full civic commenting and participation privileges active</span>
                </>
              ) : (
                <>
                  <AlertTriangle size={16} className="text-amber-600" />
                  <span className="font-semibold text-amber-800">Email Unverified</span>
                  <span className="text-slate-400">·</span>
                  <span className="text-slate-600">Verify your email to unlock comment submissions</span>
                </>
              )}
            </div>

            {!user.isEmailVerified && (
              <button
                type="button"
                onClick={handleResendVerification}
                disabled={resending}
                className="rounded-md bg-[#143454] px-3 py-1 font-semibold text-white hover:bg-[#0e2a47] disabled:opacity-50"
              >
                {resending ? "Sending…" : "Resend Link"}
              </button>
            )}
          </div>

          {resendStatus && (
            <p className="text-xs text-slate-700 bg-blue-50 border border-blue-200 p-2.5 rounded-lg">
              {resendStatus}
            </p>
          )}
        </section>
      ) : (
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs text-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-[#f0f4f8] border border-[#d6e0ea] flex items-center justify-center mx-auto text-[#143454]">
            <User size={24} />
          </div>
          <h2 className="text-lg font-bold text-[#171717]">Sign in to participate</h2>
          <p className="text-xs text-[#666666] max-w-md mx-auto">
            Log in or create a Citizen account to share feedback on government policies, access personal submission records, and track updates.
          </p>
          <div className="pt-2 flex justify-center gap-3">
            <Link
              href="/login"
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#143454] px-4 py-2 text-xs font-semibold text-white hover:bg-[#0e2a47]"
            >
              <LogIn size={13} />
              <span>Sign In</span>
            </Link>
            <Link
              href="/register"
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#E87518] px-4 py-2 text-xs font-semibold text-white hover:bg-[#d4660f]"
            >
              <UserPlus size={13} />
              <span>Create Account</span>
            </Link>
          </div>
        </section>
      )}

      {/* On this browser section */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs space-y-3">
        <h2 className="text-sm font-bold text-[#171717] flex items-center gap-2">
          <Bookmark size={15} className="text-[#143454]" />
          <span>Local Library Activity</span>
        </h2>
        <div className="flex flex-wrap gap-4 text-xs">
          <Link
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[#143454] font-semibold hover:border-[#143454]"
            href="/saved"
          >
            <Bookmark size={14} className="text-[#E87518]" />
            <span>{savedCount} saved {savedCount === 1 ? "policy" : "policies"}</span>
          </Link>
          <Link
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[#143454] font-semibold hover:border-[#143454]"
            href="/history"
          >
            <History size={14} className="text-[#143454]" />
            <span>{historyCount} browsing {historyCount === 1 ? "record" : "records"}</span>
          </Link>
        </div>
        <p className="text-[11px] text-[#666666]">
          Saved policies and local viewing history are preserved in this browser.
        </p>
      </section>

      {/* Appearance theme */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs space-y-3">
        <h2 className="text-sm font-bold text-[#171717] flex items-center gap-2">
          <Palette size={15} className="text-[#143454]" />
          <span>Appearance</span>
        </h2>
        <p className="text-xs text-[#666666]">Choose the display contrast and theme for this browser.</p>
        <button
          onClick={toggleTheme}
          className="rounded-lg border border-slate-300 bg-white px-3.5 py-1.5 text-xs font-semibold text-[#171717] hover:bg-slate-50"
        >
          Switch to {theme === "dark" ? "light" : "dark"} theme
        </button>
      </section>
    </main>
  );
}
