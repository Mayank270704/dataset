import test from "node:test";
import assert from "node:assert/strict";
import {
  isValidSlugFormat,
  isKnownPolicySlug,
  resolveSafeDataFilePath,
  SecurityError,
  BENCHMARK_PILOT_SLUG
} from "./security";

test("Slug format validation: accepts valid kebab-case policy identifiers", () => {
  const validSlugs = [
    "prahaar-counter-terrorism-policy",
    "special-campaign-6-mha",
    "national-drug-control-vision-2026-2029",
    "ai-enabled-cyber-crime-prevention",
    "immigration-visa-foreigners-registration-tracking",
    "policy-101",
    "mha-reform-act-2026"
  ];

  for (const slug of validSlugs) {
    assert.strictEqual(isValidSlugFormat(slug), true, `Expected valid: ${slug}`);
  }
});

test("Slug format validation: rejects directory traversal and path manipulation attempts", () => {
  const traversalAttempts = [
    "../",
    "../../",
    "../../etc/passwd",
    "..%2f..%2f",
    "..%252f..%252f",
    ".",
    "..",
    "/root/policy",
    "c:\\windows\\system32",
    "submissions/../../secret",
    "policy/subfolder"
  ];

  for (const attempt of traversalAttempts) {
    assert.strictEqual(isValidSlugFormat(attempt), false, `Expected rejected: ${attempt}`);
  }
});

test("Slug format validation: rejects special characters, spaces, and formatting anomalies", () => {
  const malformed = [
    "",
    "a",
    "ab",
    "-leading-hyphen",
    "trailing-hyphen-",
    "double--hyphen",
    "UPPERCASE-SLUG",
    "MixedCase-slug",
    "policy with spaces",
    "policy_with_underscores",
    "policy@name",
    "policy#1",
    "policy$var",
    "policy.json",
    "policy\0nullbyte",
    "a".repeat(101) // exceeds 100 char limit
  ];

  for (const item of malformed) {
    assert.strictEqual(isValidSlugFormat(item), false, `Expected rejected: ${item}`);
  }
});

test("Slug format validation: rejects non-string types", () => {
  assert.strictEqual(isValidSlugFormat(null), false);
  assert.strictEqual(isValidSlugFormat(undefined), false);
  assert.strictEqual(isValidSlugFormat(12345), false);
  assert.strictEqual(isValidSlugFormat({}), false);
  assert.strictEqual(isValidSlugFormat(["slug"]), false);
});

test("Known policy registry: identifies authorized policy items and benchmark", () => {
  assert.strictEqual(isKnownPolicySlug("prahaar-counter-terrorism-policy"), true);
  assert.strictEqual(isKnownPolicySlug("special-campaign-6-mha"), true);
  assert.strictEqual(isKnownPolicySlug(BENCHMARK_PILOT_SLUG), true);

  // Unknown slugs
  assert.strictEqual(isKnownPolicySlug("unknown-nonexistent-policy"), false);
  assert.strictEqual(isKnownPolicySlug("../../etc/passwd"), false);
});

test("Safe file path resolution: constrains paths strictly within data subfolder", () => {
  const safePath = resolveSafeDataFilePath("submissions", "prahaar-counter-terrorism-policy");
  assert.match(safePath, /[/\\]data[/\\]submissions[/\\]prahaar-counter-terrorism-policy\.json$/);

  const analysisPath = resolveSafeDataFilePath("analysis", "special-campaign-6-mha");
  assert.match(analysisPath, /[/\\]data[/\\]analysis[/\\]special-campaign-6-mha\.json$/);

  const jobsPath = resolveSafeDataFilePath("jobs", "national-drug-control-vision-2026-2029");
  assert.match(jobsPath, /[/\\]data[/\\]jobs[/\\]national-drug-control-vision-2026-2029\.json$/);
});

test("Safe file path resolution: throws SecurityError on traversal attempts", () => {
  assert.throws(
    () => resolveSafeDataFilePath("submissions", "../../evil"),
    (err: unknown) => err instanceof SecurityError
  );

  assert.throws(
    () => resolveSafeDataFilePath("analysis", "../jobs/override"),
    (err: unknown) => err instanceof SecurityError
  );
});
