"use client";

import { useEffect, useState } from "react";

export type VisitRecord = { slug: string; visitedAt: string; kind: "policy" | "inference" };
const SAVED_KEY = "policylens-saved-policy-slugs";
const HISTORY_KEY = "policylens-local-history";
const EVENT = "policylens-library-change";

function readJson<T>(key: string, fallback: T): T {
  try { const value = localStorage.getItem(key); return value ? JSON.parse(value) as T : fallback; }
  catch { return fallback; }
}
function notify() { window.dispatchEvent(new Event(EVENT)); }
export function readSaved(): string[] { return readJson<string[]>(SAVED_KEY, []).filter(x => typeof x === "string"); }
export function toggleSaved(slug: string) {
  const saved = readSaved();
  localStorage.setItem(SAVED_KEY, JSON.stringify(saved.includes(slug) ? saved.filter(x => x !== slug) : [...saved, slug]));
  notify();
}
export function readHistory(): VisitRecord[] { return readJson<VisitRecord[]>(HISTORY_KEY, []).filter(x => x && typeof x.slug === "string"); }
export function recordVisit(record: VisitRecord) {
  const rows = readHistory().filter(x => !(x.slug === record.slug && x.kind === record.kind));
  localStorage.setItem(HISTORY_KEY, JSON.stringify([record, ...rows].slice(0, 100)));
  notify();
}
export function clearHistory() { localStorage.removeItem(HISTORY_KEY); notify(); }
export function usePolicyLibrary<T>(reader: () => T, fallback: T): T {
  const [value, setValue] = useState<T>(fallback);
  useEffect(() => {
    const refresh = () => setValue(reader());
    refresh();
    window.addEventListener(EVENT, refresh);
    window.addEventListener("storage", refresh);
    return () => { window.removeEventListener(EVENT, refresh); window.removeEventListener("storage", refresh); };
  }, [reader]);
  return value;
}
