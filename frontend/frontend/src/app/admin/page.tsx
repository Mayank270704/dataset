import Link from "next/link";
import { KeyRound, ShieldCheck, UserPlus } from "lucide-react";
export default function AdminPage() {
  return <main className="content-width min-h-[calc(100svh-80px)] py-8 pb-24 md:py-12">
    <p className="text-[10px] font-bold uppercase tracking-[.2em] accent">Restricted workspace</p>
    <h1 className="font-display mt-2 text-4xl font-extrabold">Policy publishing studio.</h1>
    <p className="mt-3 max-w-xl text-sm leading-6 muted">A private workspace for publishing consultation information, verifying sources and moderating activity.</p>
    <section className="mt-8 max-w-2xl rounded-[24px] border p-6 sm:p-8" style={{background:"var(--surface)",borderColor:"var(--line)"}}>
      <div className="flex items-center gap-3"><div className="flex h-12 w-12 items-center justify-center rounded-2xl" style={{background:"var(--accent-soft)",color:"var(--accent)"}}><KeyRound size={21}/></div><div><h2 className="font-display text-xl font-bold">Invitation-only access</h2><p className="mt-1 text-xs muted">No public admin registration.</p></div></div>
      <div className="mt-6 space-y-4 text-xs leading-6 muted">
        <p className="flex gap-2"><ShieldCheck size={15} className="mt-0.5 shrink-0 accent"/>Admin role must be granted server-side to a pre-created or invited account. UI visibility alone is not an access control.</p>
        <p className="flex gap-2"><UserPlus size={15} className="mt-0.5 shrink-0 accent"/>Production publishing will require verified account, MFA, per-action authorization and an audit log.</p>
      </div>
      <Link href="/login" className="mt-7 inline-flex rounded-full px-5 py-3 text-xs font-semibold" style={{background:"var(--accent)",color:"var(--surface)"}}>Sign in to continue</Link>
      <p className="mt-4 text-[10px] muted">Prototype only. Admin authentication and publishing endpoints are not implemented yet.</p>
    </section>
  </main>;
}
