"use client";
import { useEffect } from "react";
import { recordVisit } from "@/lib/policy-library";
export function PolicyVisitTracker({ slug }: { slug: string }) {
  useEffect(() => { recordVisit({ slug, kind: "policy", visitedAt: new Date().toISOString() }); }, [slug]);
  return null;
}
