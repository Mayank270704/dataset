import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { ArrowUpRight } from "lucide-react";

export function FeaturePlaceholder({
  eyebrow,
  title,
  description,
  icon: Icon,
  detail,
  cta = "Back to the feed",
  href = "/"
}: {
  eyebrow: string;
  title: string;
  description: string;
  icon: LucideIcon;
  detail: string;
  cta?: string;
  href?: string;
}) {
  return (
    <main className="content-width flex min-h-[calc(100svh-80px)] items-center justify-center py-12 pb-24">
      <section className="w-full max-w-[620px] rounded-[28px] border p-7 sm:p-10" style={{background:"var(--surface)",borderColor:"var(--line)"}}>
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl" style={{background:"var(--accent-soft)",color:"var(--accent)"}}><Icon size={22} strokeWidth={1.6}/></div>
        <p className="mt-7 text-[10px] font-bold uppercase tracking-[.2em] accent">{eyebrow}</p>
        <h1 className="font-display mt-2 text-3xl font-extrabold leading-tight sm:text-4xl">{title}</h1>
        <p className="mt-4 max-w-lg text-sm leading-7 muted">{description}</p>
        <div className="mt-7 rounded-2xl p-4 sm:p-5" style={{background:"var(--surface-subtle)"}}>
          <p className="text-sm font-semibold">Coming in the account layer</p>
          <p className="mt-2 text-xs leading-6 muted">{detail}</p>
        </div>
        <Link href={href} className="mt-6 inline-flex items-center gap-2 rounded-full px-5 py-3 text-xs font-semibold" style={{background:"var(--accent)",color:"var(--surface)"}}>{cta}<ArrowUpRight size={14}/></Link>
      </section>
    </main>
  );
}
