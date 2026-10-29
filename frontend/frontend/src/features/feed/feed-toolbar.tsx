"use client";

import React from "react";
import { Globe, Search, X, SlidersHorizontal } from "lucide-react";

export type SortTimeFilter = "Latest" | "Trending" | "Last 24 hours";

export const TOPIC_CATEGORIES = [
  "All topics",
  "Environment",
  "Public services",
  "Economy",
  "Health",
  "Digital government",
  "Education"
] as const;

export type TopicCategory = (typeof TOPIC_CATEGORIES)[number];

export const LANGUAGES = [
  "All languages",
  "English",
  "Hindi",
  "Bengali",
  "Marathi",
  "Telugu",
  "Tamil",
  "Gujarati"
];

interface FeedToolbarProps {
  sortTime: SortTimeFilter;
  onSortTimeChange: (val: SortTimeFilter) => void;
  selectedTopic: TopicCategory;
  onTopicChange: (val: TopicCategory) => void;
  searchQuery: string;
  onSearchChange: (val: string) => void;
  selectedLanguage: string;
  onLanguageChange: (val: string) => void;
  onReset: () => void;
  totalCount: number;
}

export function FeedToolbar({
  sortTime,
  onSortTimeChange,
  selectedTopic,
  onTopicChange,
  searchQuery,
  onSearchChange,
  selectedLanguage,
  onLanguageChange,
  onReset,
  totalCount
}: FeedToolbarProps) {
  return (
    <div className="w-full rounded-xl border border-[#e5e7eb] bg-white p-3 sm:p-4 shadow-2xs space-y-3">
      {/* First Row — Sorting & Time (Left) + Search & Language (Right) */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        {/* Sorting and Time Buttons */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0" role="tablist" aria-label="Sort and time filter">
          {(["Latest", "Trending", "Last 24 hours"] as SortTimeFilter[]).map((tab) => {
            const isSelected = sortTime === tab;

            return (
              <button
                key={tab}
                type="button"
                role="tab"
                aria-selected={isSelected}
                onClick={() => onSortTimeChange(tab)}
                className={`shrink-0 rounded-md px-3.5 py-1.5 text-[13px] font-semibold transition-all cursor-pointer ${
                  isSelected
                    ? "bg-[#E87518] text-white shadow-xs"
                    : "bg-[#f3f4f6] text-[#444444] hover:bg-[#e5e7eb] hover:text-[#171717]"
                }`}
              >
                {tab === "Latest" ? "All items" : tab === "Trending" ? "Most recent" : tab}
              </button>
            );
          })}
        </div>

        {/* Right side of first row: Compact Search + Small Language Selector */}
        <div className="flex items-center gap-2">
          {/* Compact Search Control */}
          <div className="relative flex-1 sm:w-56 flex items-center">
            <Search size={14} className="absolute left-2.5 text-[#9ca3af] pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search policies..."
              className="w-full rounded-md border border-[#e5e7eb] bg-[#f9fafb] py-1.5 pl-8 pr-7 text-[13px] text-[#171717] outline-none transition-colors focus:border-[#E87518] focus:bg-white"
              aria-label="Search policy updates"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => onSearchChange("")}
                className="absolute right-2 text-[#9ca3af] hover:text-[#171717]"
                aria-label="Clear search"
              >
                <X size={13} />
              </button>
            )}
          </div>

          {/* Small Language Icon & Selector */}
          <div className="relative inline-flex items-center rounded-md border border-[#e5e7eb] bg-[#f9fafb] px-2 py-1.5 text-[12px] text-[#444444] hover:border-[#d1d5db]">
            <Globe size={14} className="mr-1.5 text-[#666666] shrink-0" aria-hidden="true" />
            <select
              value={selectedLanguage}
              onChange={(e) => onLanguageChange(e.target.value)}
              className="bg-transparent text-[12px] font-medium text-[#171717] outline-none cursor-pointer pr-1"
              aria-label="Select portal language"
            >
              {LANGUAGES.map((lang) => (
                <option key={lang} value={lang}>
                  {lang}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Second Row — Topic Categories */}
      <div className="border-t border-[#f1f3f5] pt-2.5">
        <div className="flex items-center justify-between gap-2">
          {/* Horizontally scrollable topic pills */}
          <div
            className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none w-full"
            role="tablist"
            aria-label="Topic categories"
          >
            {TOPIC_CATEGORIES.map((cat) => {
              const isSelected = selectedTopic === cat;

              return (
                <button
                  key={cat}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  onClick={() => onTopicChange(cat)}
                  className={`shrink-0 rounded-full px-3 py-1 text-[12px] font-medium transition-colors cursor-pointer border ${
                    isSelected
                      ? "border-[#E87518] bg-[#fff5eb] text-[#E87518] font-bold"
                      : "border-[#e5e7eb] bg-white text-[#555555] hover:border-[#d1d5db] hover:text-[#171717]"
                  }`}
                >
                  {cat}
                </button>
              );
            })}
          </div>

          {/* Reset button */}
          {(searchQuery || selectedTopic !== "All topics" || sortTime !== "Latest" || selectedLanguage !== "All languages") && (
            <button
              type="button"
              onClick={onReset}
              className="shrink-0 inline-flex items-center gap-1 text-[12px] font-semibold text-[#666666] hover:text-[#E87518] px-2 py-1 transition-colors"
              title="Reset all filters"
            >
              <SlidersHorizontal size={12} />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      {/* Results Count Banner */}
      <div className="flex items-center justify-between pt-1 text-[12px] text-[#666666]">
        <span>
          Showing <strong className="text-[#171717]">{totalCount}</strong> verified government programme updates
        </span>
        <span className="text-[#16834A] font-medium hidden sm:inline">
          Official Government Data
        </span>
      </div>
    </div>
  );
}
