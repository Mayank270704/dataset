/**
 * spam-detector.test.ts
 *
 * Tests for pure functions in spam-detector.ts (no DB required).
 * Run with: jiti src/lib/spam-detector.test.ts
 */

import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import {
  normalizeCommentText,
  computeTextHash,
  analyzeCommentSpam,
} from "./spam-detector";

// ──────────────────────────────────────────────────────────────────────────────
// normalizeCommentText
// ──────────────────────────────────────────────────────────────────────────────

test("normalizeCommentText: lowercases text", () => {
  assert.strictEqual(normalizeCommentText("Hello World"), "hello world");
});

test("normalizeCommentText: strips punctuation and special chars", () => {
  const result = normalizeCommentText("Hello, World! @#$%");
  // Only letters, numbers, spaces remain
  assert.match(result, /^[a-z0-9 ]+$/);
  assert.ok(result.includes("hello"));
  assert.ok(result.includes("world"));
});

test("normalizeCommentText: collapses multiple whitespace into single space", () => {
  const result = normalizeCommentText("one   two\t\tthree\n\nfour");
  assert.strictEqual(result, "one two three four");
});

test("normalizeCommentText: trims leading and trailing whitespace", () => {
  assert.strictEqual(normalizeCommentText("  hello world  "), "hello world");
});

test("normalizeCommentText: preserves unicode letters and numbers", () => {
  const result = normalizeCommentText("नमस्ते 123 world");
  assert.ok(result.includes("123"));
  assert.ok(result.includes("world"));
});

// ──────────────────────────────────────────────────────────────────────────────
// computeTextHash
// ──────────────────────────────────────────────────────────────────────────────

test("computeTextHash: produces a deterministic SHA-256 hex string", () => {
  const text = "hello world";
  const expected = crypto.createHash("sha256").update(text).digest("hex");
  assert.strictEqual(computeTextHash(text), expected);
});

test("computeTextHash: same input always yields same hash", () => {
  const text = "policy comment text here";
  assert.strictEqual(computeTextHash(text), computeTextHash(text));
});

test("computeTextHash: different inputs yield different hashes", () => {
  assert.notStrictEqual(computeTextHash("abc"), computeTextHash("def"));
});

test("computeTextHash: returns 64-char hex string", () => {
  const hash = computeTextHash("test");
  assert.strictEqual(hash.length, 64);
  assert.match(hash, /^[0-9a-f]{64}$/);
});

// ──────────────────────────────────────────────────────────────────────────────
// analyzeCommentSpam
// ──────────────────────────────────────────────────────────────────────────────

test("analyzeCommentSpam: REJECTS comments that are too short (< 10 chars)", () => {
  const result = analyzeCommentSpam("Hi");
  assert.strictEqual(result.status, "REJECTED");
  assert.ok(result.flags.includes("TOO_SHORT"));
});

test("analyzeCommentSpam: REJECTS empty string", () => {
  const result = analyzeCommentSpam("");
  assert.strictEqual(result.status, "REJECTED");
  assert.ok(result.flags.includes("TOO_SHORT"));
});

test("analyzeCommentSpam: REJECTS comments that are too long (> 2000 chars)", () => {
  const longText = "a".repeat(2001);
  const result = analyzeCommentSpam(longText);
  assert.strictEqual(result.status, "REJECTED");
  assert.ok(result.flags.includes("TOO_LONG"));
});

test("analyzeCommentSpam: REJECTS comments that are all special chars (MEANINGLESS_CHARS)", () => {
  // 15+ special chars — after normalization, almost nothing left
  const result = analyzeCommentSpam("!!!!!!!!!!!!!!!!!!");
  assert.strictEqual(result.status, "REJECTED");
  assert.ok(result.flags.includes("MEANINGLESS_CHARS") || result.flags.includes("TOO_SHORT"));
});

test("analyzeCommentSpam: REJECTS or FLAGS comments containing promotional URLs", () => {
  const result = analyzeCommentSpam("Check my website at https://spam.com for great deals now!");
  assert.ok(result.status === "REJECTED" || result.status === "FLAGGED");
  assert.ok(result.flags.some((f) => f.includes("UNAUTHORIZED_EXTERNAL_LINK")));
});

test("analyzeCommentSpam: REJECTS or FLAGS comments containing promotional keywords", () => {
  const result = analyzeCommentSpam("Buy now and get free bonus items delivered to your door today!");
  assert.ok(result.status === "REJECTED" || result.status === "FLAGGED");
  assert.ok(result.flags.some((f) => f.startsWith("PROMOTIONAL_KEYWORD")));
});

test("analyzeCommentSpam: APPROVES a clean, substantive policy comment", () => {
  const result = analyzeCommentSpam(
    "I believe this policy will have a positive impact on rural communities. The infrastructure investment should be prioritized."
  );
  assert.strictEqual(result.status, "APPROVED");
  assert.strictEqual(result.flags.length, 0);
  assert.ok(result.score < 0.4);
});

test("analyzeCommentSpam: adds BRAND_NEW_ACCOUNT_ACTIVITY flag when isAccountNew=true", () => {
  const result = analyzeCommentSpam(
    "This is a perfectly reasonable policy comment that meets all length requirements.",
    undefined,
    true
  );
  assert.ok(result.flags.includes("BRAND_NEW_ACCOUNT_ACTIVITY"));
  assert.ok(result.score >= 0.2);
});

test("analyzeCommentSpam: adds BRAND_NEW_ACCOUNT_ACTIVITY for very recently created account", () => {
  const justNow = new Date(Date.now() - 60 * 1000); // 1 minute ago (within 5-minute threshold)
  const result = analyzeCommentSpam(
    "This is a good comment about policy changes and their implications for citizens.",
    justNow
  );
  assert.ok(result.flags.includes("BRAND_NEW_ACCOUNT_ACTIVITY"));
});

test("analyzeCommentSpam: does NOT add new account flag for established accounts", () => {
  const oldDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
  const result = analyzeCommentSpam(
    "This policy change affects many citizens and needs careful evaluation before implementation.",
    oldDate
  );
  assert.ok(!result.flags.includes("BRAND_NEW_ACCOUNT_ACTIVITY"));
});

test("analyzeCommentSpam: REJECTS or FLAGS highly repetitive word patterns", () => {
  // 12 words, only 1 unique → ratio = 1/12 ≈ 0.08 < 0.3 → REPETITIVE_WORDS
  const repetitive = "good ".repeat(12).trim();
  const result = analyzeCommentSpam(repetitive);
  assert.ok(result.status === "REJECTED" || result.status === "FLAGGED");
  assert.ok(result.flags.includes("REPETITIVE_WORDS"));
});

test("analyzeCommentSpam: REJECTS comments with excessive repetitive characters", () => {
  // 7+ same chars in a row → REPETITIVE_CHARACTERS score += 0.4
  const result = analyzeCommentSpam("This policy is aaaaaaaaaa terrible approach to governance.");
  assert.ok(result.flags.includes("REPETITIVE_CHARACTERS"));
  assert.ok(result.score >= 0.4);
});

test("analyzeCommentSpam: normalizedHash is consistent and deterministic", () => {
  const text = "A straightforward policy comment for testing hash consistency.";
  const r1 = analyzeCommentSpam(text);
  const r2 = analyzeCommentSpam(text);
  assert.strictEqual(r1.normalizedHash, r2.normalizedHash);
  assert.strictEqual(r1.normalizedHash.length, 64);
});

test("analyzeCommentSpam: score is a number between 0 and 1 (or above 1 if multiple triggers)", () => {
  const result = analyzeCommentSpam(
    "This policy comment is completely normal and has good content for review."
  );
  assert.ok(typeof result.score === "number");
  assert.ok(result.score >= 0);
});
