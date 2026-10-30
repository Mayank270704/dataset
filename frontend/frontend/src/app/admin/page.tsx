"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  ShieldCheck,
  Users,
  FileText,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Loader2,
  LogIn,
  ShieldOff,
  RefreshCw,
  ScrollText,
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

interface ModerationComment {
  id: string;
  policySlug: string;
  stakeholder: string;
  originalText: string;
  status: string;
  moderationReason: string | null;
  spamScore: number;
  spamFlags: string[] | string;
  createdAt: string;
  authorId: string;
  authorEmail: string;
  authorName: string;
  authorStatus: string;
}

interface AdminUser {
  id: string;
  email: string;
  displayName: string;
  role: string;
  isEmailVerified: boolean;
  status: string;
  suspensionReason: string | null;
  createdAt: string;
  commentsCount: number;
}

interface AuditLog {
  id: string;
  eventType: string;
  actorId: string | null;
  resourceType: string | null;
  resourceId: string | null;
  actionResult: string;
  createdAt: string;
  metadata: Record<string, unknown> | null;
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper components
// ──────────────────────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const colours: Record<string, string> = {
    PENDING: "bg-amber-100 text-amber-800 border-amber-200",
    FLAGGED: "bg-orange-100 text-orange-800 border-orange-200",
    APPROVED: "bg-green-100 text-green-800 border-green-200",
    REJECTED: "bg-red-100 text-red-800 border-red-200",
    ACTIVE: "bg-green-100 text-green-800 border-green-200",
    RESTRICTED: "bg-amber-100 text-amber-800 border-amber-200",
    SUSPENDED: "bg-red-100 text-red-800 border-red-200",
  };
  return (
    <span
      className={`inline-block text-[11px] font-semibold px-2 py-0.5 rounded-full border ${colours[status] ?? "bg-gray-100 text-gray-700 border-gray-200"}`}
    >
      {status}
    </span>
  );
}

function SectionHeader({
  icon,
  title,
  subtitle,
  onRefresh,
  refreshing,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
  onRefresh?: () => void;
  refreshing?: boolean;
}) {
  return (
    <div className="flex items-start justify-between pb-4 border-b border-[#f1f3f5]">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#f0f4f8] text-[#143454]">
          {icon}
        </div>
        <div>
          <h2 className="font-display text-[18px] font-bold text-[#171717]">{title}</h2>
          {subtitle && <p className="text-[12px] text-[#6b7280] mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {onRefresh && (
        <button
          onClick={onRefresh}
          disabled={refreshing}
          className="inline-flex items-center gap-1.5 text-[12px] text-[#143454] hover:text-[#0e2a47] font-semibold disabled:opacity-50"
        >
          <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />
          Refresh
        </button>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Moderation Queue
// ──────────────────────────────────────────────────────────────────────────────

function ModerationQueue() {
  const [comments, setComments] = useState<ModerationComment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionStates, setActionStates] = useState<Record<string, "approving" | "rejecting" | null>>({});
  const [actionResults, setActionResults] = useState<Record<string, { ok: boolean; message: string }>>({});

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/moderation/comments?status=PENDING,FLAGGED", { cache: "no-store" });
      const data = await res.json();
      if (res.ok && data.success) {
        setComments(data.comments);
      } else {
        setError(data.message || "Failed to load moderation queue.");
      }
    } catch {
      setError("Network error loading moderation queue.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const moderate = async (id: string, action: "approve" | "reject") => {
    setActionStates((s) => ({ ...s, [id]: action === "approve" ? "approving" : "rejecting" }));
    try {
      const res = await fetch(`/api/moderation/comments/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: action === "approve" ? "APPROVE" : "REJECT",
          reason: action === "reject" ? "Rejected by moderator." : undefined,
        }),
      });
      const data = await res.json();
      setActionResults((r) => ({
        ...r,
        [id]: { ok: res.ok && data.success, message: data.message || (res.ok ? "Done." : "Failed.") },
      }));
      if (res.ok && data.success) {
        setComments((prev) => prev.filter((c) => c.id !== id));
      }
    } catch {
      setActionResults((r) => ({ ...r, [id]: { ok: false, message: "Network error." } }));
    } finally {
      setActionStates((s) => ({ ...s, [id]: null }));
    }
  };

  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 space-y-5">
      <SectionHeader
        icon={<ShieldCheck size={18} />}
        title="Moderation Queue"
        subtitle="Pending and flagged comments awaiting review"
        onRefresh={load}
        refreshing={loading}
      />

      {loading && (
        <div className="flex items-center gap-2 text-[13px] text-[#6b7280] py-4">
          <Loader2 size={14} className="animate-spin" /> Loading queue…
        </div>
      )}
      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-[13px] text-red-700 flex items-center gap-2">
          <AlertTriangle size={14} /> {error}
        </div>
      )}
      {!loading && !error && comments.length === 0 && (
        <p className="text-[13px] text-[#6b7280] py-4 text-center">
          No pending comments. Great job, moderators! 🎉
        </p>
      )}

      <div className="space-y-4">
        {comments.map((c) => {
          const flags = Array.isArray(c.spamFlags) ? c.spamFlags : JSON.parse(c.spamFlags || "[]");
          const state = actionStates[c.id];
          const res = actionResults[c.id];
          return (
            <div key={c.id} className="rounded-lg border border-[#e5e7eb] p-4 space-y-2.5 bg-[#fafbfc]">
              <div className="flex flex-wrap items-center gap-2 text-[12px]">
                <StatusBadge status={c.status} />
                <span className="text-[#6b7280]">Policy: <strong className="text-[#171717]">{c.policySlug}</strong></span>
                <span className="text-[#6b7280]">{new Date(c.createdAt).toLocaleString("en-IN")}</span>
                <span className="text-[#6b7280]">Spam score: <strong>{c.spamScore?.toFixed(2)}</strong></span>
              </div>
              <div className="text-[12px] text-[#4b5563]">
                <strong>Author:</strong> {c.authorName} ({c.authorEmail})
                {" "}<StatusBadge status={c.authorStatus} />
              </div>
              <p className="text-[13px] text-[#171717] bg-white rounded border border-[#e5e7eb] p-3 leading-relaxed">
                {c.originalText}
              </p>
              {flags.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {flags.map((f: string) => (
                    <span key={f} className="text-[11px] bg-orange-100 text-orange-800 border border-orange-200 px-2 py-0.5 rounded-full font-semibold">
                      {f}
                    </span>
                  ))}
                </div>
              )}
              {res && (
                <p className={`text-[12px] font-semibold ${res.ok ? "text-green-700" : "text-red-700"}`}>
                  {res.message}
                </p>
              )}
              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={() => moderate(c.id, "approve")}
                  disabled={!!state}
                  className="inline-flex items-center gap-1 rounded-lg bg-green-600 px-3 py-1.5 text-[12px] font-bold text-white hover:bg-green-700 disabled:opacity-50 transition-colors"
                >
                  {state === "approving" ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle size={12} />}
                  Approve
                </button>
                <button
                  onClick={() => moderate(c.id, "reject")}
                  disabled={!!state}
                  className="inline-flex items-center gap-1 rounded-lg bg-red-600 px-3 py-1.5 text-[12px] font-bold text-white hover:bg-red-700 disabled:opacity-50 transition-colors"
                >
                  {state === "rejecting" ? <Loader2 size={12} className="animate-spin" /> : <XCircle size={12} />}
                  Reject
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// User Management (Admin only)
// ──────────────────────────────────────────────────────────────────────────────

function UserManagement() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updatingId, setUpdatingId] = useState<string | null>(null);
  const [updateResults, setUpdateResults] = useState<Record<string, { ok: boolean; message: string }>>({});

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/users", { cache: "no-store" });
      const data = await res.json();
      if (res.ok && data.success) {
        setUsers(data.users);
      } else {
        setError(data.message || "Failed to load users.");
      }
    } catch {
      setError("Network error loading users.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const updateStatus = async (userId: string, status: string) => {
    setUpdatingId(userId);
    setUpdateResults((r) => {
      const copy = { ...r };
      delete copy[userId];
      return copy;
    });
    try {
      const res = await fetch("/api/admin/users", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId, status }),
      });
      const data = await res.json();
      setUpdateResults((r) => ({
        ...r,
        [userId]: { ok: res.ok && data.success, message: data.message || (res.ok ? "Updated." : "Failed.") },
      }));
      if (res.ok && data.success) {
        setUsers((prev) =>
          prev.map((u) => (u.id === userId ? { ...u, status: data.user.status } : u))
        );
      }
    } catch {
      setUpdateResults((r) => ({ ...r, [userId]: { ok: false, message: "Network error." } }));
    } finally {
      setUpdatingId(null);
    }
  };

  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 space-y-5">
      <SectionHeader
        icon={<Users size={18} />}
        title="User Management"
        subtitle="All registered users — Admin only"
        onRefresh={load}
        refreshing={loading}
      />

      {loading && (
        <div className="flex items-center gap-2 text-[13px] text-[#6b7280] py-4">
          <Loader2 size={14} className="animate-spin" /> Loading users…
        </div>
      )}
      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-[13px] text-red-700 flex items-center gap-2">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-[#e5e7eb] text-left text-[#6b7280]">
              <th className="pb-2 pr-4 font-semibold">User</th>
              <th className="pb-2 pr-4 font-semibold">Role</th>
              <th className="pb-2 pr-4 font-semibold">Status</th>
              <th className="pb-2 pr-4 font-semibold">Comments</th>
              <th className="pb-2 font-semibold">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#f1f3f5]">
            {users.map((u) => {
              const res = updateResults[u.id];
              return (
                <tr key={u.id} className="align-top">
                  <td className="py-3 pr-4">
                    <p className="font-semibold text-[#171717]">{u.displayName}</p>
                    <p className="text-[#6b7280] text-[11px]">{u.email}</p>
                    {!u.isEmailVerified && (
                      <span className="text-[10px] text-amber-600 font-semibold">Unverified</span>
                    )}
                  </td>
                  <td className="py-3 pr-4">
                    <span className="text-[#4b5563]">{u.role}</span>
                  </td>
                  <td className="py-3 pr-4">
                    <StatusBadge status={u.status} />
                  </td>
                  <td className="py-3 pr-4 text-[#4b5563]">{u.commentsCount}</td>
                  <td className="py-3">
                    <div className="flex flex-wrap gap-1.5 items-center">
                      {(["ACTIVE", "RESTRICTED", "SUSPENDED"] as const).map((s) => (
                        <button
                          key={s}
                          disabled={u.status === s || updatingId === u.id}
                          onClick={() => updateStatus(u.id, s)}
                          className={`rounded px-2 py-0.5 text-[11px] font-semibold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                            s === "ACTIVE"
                              ? "border-green-300 text-green-700 hover:bg-green-50"
                              : s === "RESTRICTED"
                              ? "border-amber-300 text-amber-700 hover:bg-amber-50"
                              : "border-red-300 text-red-700 hover:bg-red-50"
                          } ${u.status === s ? "opacity-30 cursor-default" : "cursor-pointer"}`}
                        >
                          {updatingId === u.id ? <Loader2 size={10} className="animate-spin inline" /> : null}
                          {" "}{s}
                        </button>
                      ))}
                    </div>
                    {res && (
                      <p className={`text-[11px] mt-1 font-semibold ${res.ok ? "text-green-700" : "text-red-700"}`}>
                        {res.message}
                      </p>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Audit Logs (Admin only)
// ──────────────────────────────────────────────────────────────────────────────

function AuditLogs() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/audit-logs?limit=50", { cache: "no-store" });
      const data = await res.json();
      if (res.ok && data.success) {
        setLogs(data.logs);
      } else {
        setError(data.message || "Failed to load audit logs.");
      }
    } catch {
      setError("Network error loading audit logs.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 space-y-5">
      <SectionHeader
        icon={<ScrollText size={18} />}
        title="Audit Logs"
        subtitle="Last 50 security and moderation events — Admin only"
        onRefresh={load}
        refreshing={loading}
      />

      {loading && (
        <div className="flex items-center gap-2 text-[13px] text-[#6b7280] py-4">
          <Loader2 size={14} className="animate-spin" /> Loading logs…
        </div>
      )}
      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-[13px] text-red-700 flex items-center gap-2">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-[#e5e7eb] text-left text-[#6b7280]">
              <th className="pb-2 pr-4 font-semibold">Event</th>
              <th className="pb-2 pr-4 font-semibold">Result</th>
              <th className="pb-2 pr-4 font-semibold">Resource</th>
              <th className="pb-2 font-semibold">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#f1f3f5]">
            {logs.map((log) => (
              <tr key={log.id} className="align-top">
                <td className="py-2 pr-4">
                  <span className="font-mono text-[11px] text-[#143454]">{log.eventType}</span>
                </td>
                <td className="py-2 pr-4">
                  <span
                    className={`text-[11px] font-semibold ${
                      log.actionResult === "SUCCESS"
                        ? "text-green-700"
                        : log.actionResult === "FAILURE" || log.actionResult === "REJECTED"
                        ? "text-red-700"
                        : "text-[#4b5563]"
                    }`}
                  >
                    {log.actionResult}
                  </span>
                </td>
                <td className="py-2 pr-4 text-[#6b7280]">
                  {log.resourceType && log.resourceId
                    ? `${log.resourceType}:${log.resourceId.slice(0, 8)}…`
                    : "—"}
                </td>
                <td className="py-2 text-[#6b7280]">
                  {new Date(log.createdAt).toLocaleString("en-IN")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Admin Dashboard Page
// ──────────────────────────────────────────────────────────────────────────────

export default function AdminPage() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <main className="content-width min-h-[calc(100svh-80px)] py-8 pb-24 md:py-12 flex items-center justify-center">
        <Loader2 size={24} className="animate-spin text-[#143454]" />
      </main>
    );
  }

  if (!user) {
    return (
      <main className="content-width min-h-[calc(100svh-80px)] py-8 pb-24 md:py-12 space-y-4">
        <p className="text-[10px] font-bold uppercase tracking-[.2em] text-[#E87518]">
          Restricted workspace
        </p>
        <h1 className="font-display mt-2 text-4xl font-extrabold text-[#171717]">
          Admin Dashboard
        </h1>
        <p className="text-sm text-[#4b5563] max-w-md">
          Please sign in with an account that has Moderator or Admin privileges to access this
          dashboard.
        </p>
        <Link
          href="/login?redirect=/admin"
          className="inline-flex items-center gap-2 rounded-full bg-[#143454] px-5 py-3 text-sm font-semibold text-white hover:bg-[#0e2a47] transition-colors"
        >
          <LogIn size={15} />
          Sign in to continue
        </Link>
      </main>
    );
  }

  if (user.role === "CITIZEN") {
    return (
      <main className="content-width min-h-[calc(100svh-80px)] py-8 pb-24 md:py-12 space-y-4">
        <p className="text-[10px] font-bold uppercase tracking-[.2em] text-[#E87518]">
          Access Denied
        </p>
        <h1 className="font-display mt-2 text-4xl font-extrabold text-[#171717]">
          Admin Dashboard
        </h1>
        <div className="rounded-xl border border-red-200 bg-red-50 p-6 max-w-lg flex items-start gap-3">
          <ShieldOff size={20} className="text-red-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-red-900">Insufficient permissions</p>
            <p className="text-[13px] text-red-700 mt-1">
              Your account (<strong>{user.displayName}</strong>) has the{" "}
              <strong>{user.role}</strong> role. Moderator or Admin role is required to access this
              workspace.
            </p>
          </div>
        </div>
        <Link href="/" className="inline-block text-[13px] text-[#143454] font-semibold underline">
          ← Return to home
        </Link>
      </main>
    );
  }

  // ── MODERATOR or ADMIN ─────────────────────────────────────────────────────
  const isAdmin = user.role === "ADMIN";

  return (
    <main className="content-width min-h-[calc(100svh-80px)] py-8 pb-24 md:py-12 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[.2em] text-[#E87518]">
            {isAdmin ? "Admin workspace" : "Moderator workspace"}
          </p>
          <h1 className="font-display mt-1 text-3xl sm:text-4xl font-extrabold text-[#171717]">
            Policy Publishing Studio
          </h1>
          <p className="mt-2 text-[13px] text-[#4b5563]">
            Signed in as <strong>{user.displayName}</strong> &middot;{" "}
            <span className="text-[#143454] font-semibold">{user.role}</span>
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 rounded-full bg-[#f0f4f8] border border-[#d6e0ea] px-3 py-1.5 text-[12px] font-semibold text-[#143454]">
            <FileText size={13} />
            <span>Moderation</span>
          </div>
          {isAdmin && (
            <>
              <div className="flex items-center gap-1.5 rounded-full bg-[#f0f4f8] border border-[#d6e0ea] px-3 py-1.5 text-[12px] font-semibold text-[#143454]">
                <Users size={13} />
                <span>Users</span>
              </div>
              <div className="flex items-center gap-1.5 rounded-full bg-[#f0f4f8] border border-[#d6e0ea] px-3 py-1.5 text-[12px] font-semibold text-[#143454]">
                <ScrollText size={13} />
                <span>Audit Logs</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Moderation Queue — all privileged roles */}
      <ModerationQueue />

      {/* User Management — Admin only */}
      {isAdmin && <UserManagement />}

      {/* Audit Logs — Admin only */}
      {isAdmin && <AuditLogs />}
    </main>
  );
}
