"use client";
import { Bookmark, BookmarkCheck } from "lucide-react";
import { useCallback } from "react";
import { readSaved, toggleSaved, usePolicyLibrary } from "@/lib/policy-library";

export function SavePolicyButton({ slug }: { slug: string }) {
  const reader = useCallback(() => readSaved(), []);
  const saved = usePolicyLibrary(reader, []);
  const isSaved = saved.includes(slug);
  return <button type="button" onClick={() => toggleSaved(slug)} aria-pressed={isSaved} className="inline-flex min-h-9 items-center gap-2 rounded-lg border border-[#d6e0ea] px-3 py-1.5 text-xs font-semibold text-[#143454] hover:bg-[#f0f4f8]">
    {isSaved ? <BookmarkCheck size={15}/> : <Bookmark size={15}/>} {isSaved ? "Saved" : "Save policy"}
  </button>;
}
