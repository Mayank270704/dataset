import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, Building, Calendar, CheckCircle2, ExternalLink, Tag } from "lucide-react";
import { officialPolicies } from "@/features/feed/demo-data";
import { PublicOpinionSection } from "@/features/opinions/public-opinion-section";
import { PolicyVisitTracker } from "@/features/feed/policy-visit-tracker";
import { SavePolicyButton } from "@/features/feed/save-policy-button";

// Note: this feature also ships src/features/opinions/comments-data.ts (a static
// "public opinion summary" / "top opinions" / "improvement suggestions" demo dataset) and the
// matching PublicOpinionSummaryCard / TopOpinionsSection / PolicyImprovementSuggestions
// components. That dataset's own fields explicitly label it "QUARANTINED" and state the figures
// are unverified labels on Canadian Open Government consultation comments, and "none of these
// figures may be presented as PolicyLens results" -- it is the same research corpus this whole
// project studies, not analysis of these Indian government policy items. Rendering it here would
// misrepresent it, so those three components are deliberately NOT used on this page.

export default async function PolicyDetailPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = officialPolicies.find(p => p.slug === slug);
  if (!post) notFound();

  return (
    <main className="mx-auto max-w-4xl px-4 py-6 pb-24 sm:py-8">
      <Link href="/" className="inline-flex items-center gap-1.5 text-xs text-[#666666] hover:text-[#E87518]">
        <ArrowLeft size={14} /> Back to feed
      </Link>

      <article className="mt-4 rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6">
        <PolicyVisitTracker slug={post.slug} />
        <div className="flex flex-wrap items-center gap-2 text-[12px]">
          <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-semibold text-[#143454] bg-[#e8eef5] text-[11px] uppercase tracking-wider">
            <Tag size={11} className="text-[#E87518]" />
            {post.category}
          </span>
          <span className="inline-flex items-center gap-1 font-medium text-[#16834A] text-[12px]">
            <CheckCircle2 size={13} />
            {post.statusLabel}
          </span>
          <span className="flex items-center gap-1 text-[#666666]">
            <Calendar size={13} className="text-[#9ca3af]" />
            {post.publishedDate}
          </span>
        </div>

        <h1 className="font-display mt-3 text-2xl sm:text-3xl font-extrabold leading-tight text-[#171717]">
          {post.title}
        </h1>
        <p className="mt-2.5 text-[14px] sm:text-[15px] leading-relaxed text-[#374151]">{post.summary}</p>

        {post.keyHighlights.length > 0 && (
          <div className="mt-4 rounded-lg border border-[#e5e7eb] bg-[#f8fafc] p-3.5 text-[13px] text-[#334155]">
            <span className="block text-[12px] font-bold uppercase tracking-wider text-[#143454] mb-1.5">
              Key Programme Highlights
            </span>
            <ul className="space-y-1.5 list-disc list-inside text-[#475569]">
              {post.keyHighlights.map((h, i) => <li key={i}>{h}</li>)}
            </ul>
          </div>
        )}

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-[#f1f3f5] pt-3.5 text-[13px]">
          <div className="flex items-center gap-1.5 font-medium text-[#4b5563]">
            <Building size={14} className="text-[#143454]" />
            {post.ministry}
          </div>
          <div className="flex items-center gap-3"><SavePolicyButton slug={post.slug} /><a
            href={post.sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 font-semibold text-[#143454] hover:text-[#E87518]"
          >
            Official source <ExternalLink size={13} />
          </a></div>
        </div>
      </article>

      <div className="mt-6 space-y-6">
        <PublicOpinionSection
          policySlug={post.slug}
          policyTitle={post.title}
          policyId={post.id}
          openedAt={post.openedAt}
          commentingDeadline={post.commentingDeadline}
          initialConsultationStatus={post.consultationStatus}
        />
      </div>
    </main>
  );
}
