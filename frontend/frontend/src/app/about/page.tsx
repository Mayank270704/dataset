import Link from "next/link";
import { ArrowLeft, BookOpen, CheckCircle2, Fingerprint, Scale, Building2 } from "lucide-react";
import { PortalIdentity } from "@/components/brand/portal-identity";

const principles = [
  {
    icon: BookOpen,
    title: "Primary Source Integrity",
    text: "Every policy update, direct benefit transfer metric, and welfare scheme published links directly to authenticated government releases from PIB and Union Ministries."
  },
  {
    icon: Scale,
    title: "Factual Public Information",
    text: "Distinguishes officially notified government programmes from proposed draft consultations and civic inputs without fabricated metrics or biased framing."
  },
  {
    icon: Fingerprint,
    title: "Non-partisan & Independent",
    text: "Maintained as an independent civic portal to make government policies transparent, accessible, and structured for citizen awareness and academic research."
  },
  {
    icon: CheckCircle2,
    title: "Verified Documentation",
    text: "Only authentic scheme guidelines, gazettes, and official notifications are catalogued, eliminating fake viral claims and misleading misinformation."
  }
];

export default function AboutPage() {
  return (
    <main className="max-w-4xl mx-auto px-4 py-8 pb-24 md:py-12">
      <div className="mb-6">
        <Link href="/" className="inline-flex items-center gap-1.5 text-xs text-[#666666] hover:text-[#E87518]">
          <ArrowLeft size={14} /> Back to Portal Home
        </Link>
      </div>

      <div className="mb-8 p-4 rounded-xl border border-[#e5e7eb] bg-white shadow-2xs">
        <PortalIdentity />
      </div>

      <div className="space-y-3">
        <span className="text-[11px] font-bold uppercase tracking-wider text-[#143454] bg-[#e8eef5] px-2.5 py-1 rounded">
          Civic Information Repository
        </span>
        <h1 className="font-display text-3xl sm:text-4xl font-extrabold text-[#171717] leading-tight">
          Transparent, Verified Public Policy Information for Every Citizen.
        </h1>
        <p className="text-[14px] leading-relaxed text-[#444444] max-w-2xl">
          The Public Policy Portal is an independent, non-partisan repository designed to organize
          and index official Government of India welfare schemes, administrative guidelines, and
          parliamentary policy updates directly from primary official sources.
        </p>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {principles.map(({ icon: Icon, title, text }) => (
          <section key={title} className="rounded-xl border border-[#e5e7eb] bg-white p-5 shadow-2xs">
            <Icon size={20} className="text-[#E87518]" />
            <h2 className="font-display mt-3 text-[17px] font-bold text-[#171717]">{title}</h2>
            <p className="mt-2 text-[13px] leading-relaxed text-[#666666]">{text}</p>
          </section>
        ))}
      </div>

      <div className="mt-8 rounded-xl border border-[#e5e7eb] bg-[#f8fafc] p-6 space-y-2">
        <div className="flex items-center gap-2 text-xs font-bold text-[#143454] uppercase tracking-wider">
          <Building2 size={16} className="text-[#E87518]" />
          <span>Independent Platform Disclaimer</span>
        </div>
        <p className="text-[13px] text-[#444444] leading-relaxed">
          This portal is an independent public-interest platform and is not operated by or affiliated with the Government of India or any Union Ministry. Official portals are accessible at <a href="https://www.india.gov.in" target="_blank" rel="noopener noreferrer" className="text-[#143454] font-semibold underline">india.gov.in</a> and <a href="https://www.pib.gov.in" target="_blank" rel="noopener noreferrer" className="text-[#143454] font-semibold underline">pib.gov.in</a>.
        </p>
      </div>
    </main>
  );
}
