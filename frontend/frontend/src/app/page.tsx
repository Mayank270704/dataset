"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { FlaskConical } from "lucide-react";
import { FeedToolbar, type SortTimeFilter, type TopicCategory } from "@/features/feed/feed-toolbar";
import { PolicyCard } from "@/features/feed/policy-card";
import { officialPolicies } from "@/features/feed/demo-data";

export default function HomePage() {
  const [sortTime, setSortTime] = useState<SortTimeFilter>("Latest");
  const [selectedTopic, setSelectedTopic] = useState<TopicCategory>("All topics");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("All languages");

  const filtered = useMemo(() => {
    let list = [...officialPolicies];
    if (selectedTopic !== "All topics") list = list.filter(p => p.category === selectedTopic);
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      list = list.filter(p => p.title.toLowerCase().includes(q) || p.summary.toLowerCase().includes(q));
    }
    if (selectedLanguage !== "All languages") {
      list = list.filter(p => p.language.toLowerCase().includes(selectedLanguage.toLowerCase()));
    }
    // "Trending" and "Last 24 hours" have no real engagement/timestamp data to sort or filter by
    // in this dataset, so all three tabs currently show the same chronological list.
    void sortTime;
    return list;
  }, [selectedTopic, searchQuery, selectedLanguage, sortTime]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-6 pb-24 sm:py-8">
      <div className="mb-4 rounded-xl border border-[#e5e7eb] bg-white p-4 sm:p-5">
        <h1 className="font-display text-xl sm:text-2xl font-extrabold text-[#171717]">
          Government Policy &amp; Programme Feed
        </h1>
        <p className="mt-1.5 text-[13px] sm:text-[14px] leading-relaxed text-[#4b5563]">
          Browse official Government of India policy and programme updates, sourced directly from
          PIB and Union Ministries. Share your opinion on any item below — PolicyLens runs a
          provisional stance/sentiment model on your comment and shows you its result.
        </p>
        <Link
          href="/inference"
          className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-[#d6e0ea] bg-[#f0f4f8] px-3 py-1.5 text-[12px] font-semibold text-[#143454] hover:bg-[#e8eef5]"
        >
          <FlaskConical size={13} />
          Open the research workspace (verified model evaluation &amp; corpus tools)
        </Link>
      </div>

      <FeedToolbar
        sortTime={sortTime}
        onSortTimeChange={setSortTime}
        selectedTopic={selectedTopic}
        onTopicChange={setSelectedTopic}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        selectedLanguage={selectedLanguage}
        onLanguageChange={setSelectedLanguage}
        onReset={() => {
          setSortTime("Latest");
          setSelectedTopic("All topics");
          setSearchQuery("");
          setSelectedLanguage("All languages");
        }}
        totalCount={filtered.length}
      />

      <div className="mt-4 space-y-4">
        {filtered.length === 0 ? (
          <p className="py-10 text-center text-sm text-[#666666]">No items match the current filters.</p>
        ) : (
          filtered.map((post, i) => <PolicyCard key={post.id} post={post} featured={i === 0} />)
        )}
      </div>
    </main>
  );
}
