"use client";

import Link from "next/link";
import { ExternalLink, CheckCircle2, Building, Calendar, ArrowRight, Tag } from "lucide-react";
import type { PolicyPost } from "./types";
import { SavePolicyButton } from "./save-policy-button";

interface PolicyCardProps {
  post: PolicyPost;
  featured?: boolean;
}

export function PolicyCard({ post, featured = false }: PolicyCardProps) {
  return (
    <article
      className={`rounded-xl border bg-white transition-all duration-200 hover:shadow-md ${
        featured ? "border-[#143454]/30 shadow-xs" : "border-[#e5e7eb]"
      }`}
    >
      {/* Top Metadata Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#f1f3f5] px-4 py-2.5 sm:px-5 text-[12px] bg-[#fafbfc]">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-semibold text-[#143454] bg-[#e8eef5] text-[11px] uppercase tracking-wider">
            <Tag size={11} className="text-[#E87518]" />
            {post.category}
          </span>
          <span className="inline-flex items-center gap-1 font-medium text-[#16834A] text-[12px]">
            <CheckCircle2 size={13} />
            {post.statusLabel}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-[#666666] text-[12px]">
          <Calendar size={13} className="text-[#9ca3af]" />
          <span>{post.publishedDate}</span>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="p-4 sm:p-5">
        {/* Title */}
        <h2 className="font-display text-[18px] sm:text-[20px] font-bold leading-snug text-[#171717] hover:text-[#143454]">
          <Link href={`/policy/${post.slug}`}>{post.title}</Link>
        </h2>

        {/* Factual Summary */}
        <p className="mt-2.5 text-[14px] sm:text-[15px] leading-relaxed text-[#374151]">
          {post.summary}
        </p>

        {/* Key Highlights list if featured */}
        {featured && post.keyHighlights && post.keyHighlights.length > 0 && (
          <div className="mt-3.5 rounded-lg border border-[#e5e7eb] bg-[#f8fafc] p-3.5 text-[13px] text-[#334155]">
            <span className="block text-[12px] font-bold uppercase tracking-wider text-[#143454] mb-1.5">
              Key Programme Highlights:
            </span>
            <ul className="space-y-1.5 list-disc list-inside text-[#475569]">
              {post.keyHighlights.map((highlight, idx) => (
                <li key={idx} className="leading-relaxed">
                  {highlight}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Footer Actions & Ministry Citation */}
        <div className="mt-4 pt-3.5 border-t border-[#f1f3f5] flex flex-wrap items-center justify-between gap-3 text-[13px]">
          <div className="flex items-center gap-1.5 text-[#4b5563] font-medium">
            <Building size={14} className="text-[#143454] shrink-0" />
            <span className="text-[12px] sm:text-[13px]">{post.ministry}</span>
          </div>

          <div className="flex items-center gap-3">
            <SavePolicyButton slug={post.slug} />
            {/* Direct Official Publication Link */}
            <a
              href={post.sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 font-semibold text-[#143454] hover:text-[#E87518] text-[12px] transition-colors py-1 px-2 rounded hover:bg-[#f1f5f9]"
              title="Open verified official government publication in a new tab"
            >
              <span>Official source</span>
              <ExternalLink size={13} />
            </a>

            {/* View Policy Action */}
            <Link
              href={`/policy/${post.slug}`}
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#143454] px-3.5 py-1.5 text-[12px] font-semibold text-white shadow-2xs hover:bg-[#0e2a47] transition-colors"
            >
              <span>View Policy</span>
              <ArrowRight size={13} className="text-[#E87518]" />
            </Link>
          </div>
        </div>
      </div>
    </article>
  );
}
