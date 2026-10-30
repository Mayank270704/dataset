/**
 * auth-security.test.ts
 *
 * Comprehensive tests for auth-crypto functions.
 * Run with: jiti src/lib/auth-security.test.ts
 */

import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import {
  isValidEmail,
  normalizeEmail,
  validatePassword,
  generateSecureToken,
  hashToken,
} from "./auth-crypto";

// ──────────────────────────────────────────────────────────────────────────────
// Email validation
// ──────────────────────────────────────────────────────────────────────────────

test("isValidEmail: accepts standard valid email addresses", () => {
  const valid = [
    "user@example.com",
    "alice.bob@domain.org",
    "user+tag@mail.co.in",
    "test123@sub.example.gov.in",
    "a@b.co",
  ];
  for (const email of valid) {
    assert.strictEqual(isValidEmail(email), true, `Expected valid: ${email}`);
  }
});

test("isValidEmail: rejects obviously invalid email addresses", () => {
  const invalid = [
    "",
    "notanemail",
    "@nodomain.com",
    "missing@",
    "two@@domain.com",
    "sp aces@example.com",
    "a@b",           // single-segment TLD rejected by our regex
    null,
    undefined,
    12345,
    {},
  ];
  for (const email of invalid) {
    assert.strictEqual(isValidEmail(email), false, `Expected invalid: ${String(email)}`);
  }
});

test("isValidEmail: rejects emails that are too short or too long", () => {
  // too short: total length < 5
  assert.strictEqual(isValidEmail("a@b"), false);
  // too long: total length > 254
  const longLocal = "a".repeat(250) + "@example.com"; // > 254 chars
  assert.strictEqual(isValidEmail(longLocal), false);
});

// ──────────────────────────────────────────────────────────────────────────────
// normalizeEmail
// ──────────────────────────────────────────────────────────────────────────────

test("normalizeEmail: lowercases and trims whitespace", () => {
  assert.strictEqual(normalizeEmail("  USER@EXAMPLE.COM  "), "user@example.com");
  assert.strictEqual(normalizeEmail("Alice@Domain.ORG"), "alice@domain.org");
  assert.strictEqual(normalizeEmail("test@test.com"), "test@test.com");
});

// ──────────────────────────────────────────────────────────────────────────────
// Password validation
// ──────────────────────────────────────────────────────────────────────────────

test("validatePassword: accepts strong passwords", () => {
  const strong = [
    "Secure1@Pass",
    "MyP@ssw0rd!",
    "C0mpl3x$ecret",
    "ValidPass1!",
  ];
  for (const pw of strong) {
    const result = validatePassword(pw);
    assert.strictEqual(result.valid, true, `Expected valid password: ${pw}, errors: ${result.errors.join(", ")}`);
    assert.strictEqual(result.errors.length, 0);
  }
});

test("validatePassword: rejects passwords that are too short (< 8 chars)", () => {
  const result = validatePassword("Ab1!");
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.some((e) => e.includes("8 characters")));
});

test("validatePassword: rejects passwords with no uppercase letter", () => {
  const result = validatePassword("nouppercase1@");
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.some((e) => e.toLowerCase().includes("uppercase")));
});

test("validatePassword: rejects passwords with no number", () => {
  const result = validatePassword("NoNumber@Pass");
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.some((e) => e.includes("number")));
});

test("validatePassword: rejects passwords with no special character", () => {
  const result = validatePassword("NoSpecial1Pass");
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.some((e) => e.toLowerCase().includes("special")));
});

test("validatePassword: rejects passwords with no lowercase letter", () => {
  const result = validatePassword("NOLOWER1!");
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.some((e) => e.toLowerCase().includes("lowercase")));
});

test("validatePassword: rejects passwords that exceed 128 characters", () => {
  const tooLong = "Aa1!" + "x".repeat(130);
  const result = validatePassword(tooLong);
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.some((e) => e.includes("128 characters")));
});

test("validatePassword: rejects non-string inputs", () => {
  for (const bad of [null, undefined, 123, [], {}]) {
    const result = validatePassword(bad);
    assert.strictEqual(result.valid, false);
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// hashToken
// ──────────────────────────────────────────────────────────────────────────────

test("hashToken: produces a deterministic SHA-256 hex string", () => {
  const token = "test-token-abc";
  const expected = crypto.createHash("sha256").update(token).digest("hex");
  assert.strictEqual(hashToken(token), expected);
  // Same input always yields same output
  assert.strictEqual(hashToken(token), hashToken(token));
  // Different inputs yield different hashes
  assert.notStrictEqual(hashToken("abc"), hashToken("def"));
});

test("hashToken: produces a 64-character hex string (SHA-256 output)", () => {
  const hash = hashToken("some-opaque-token");
  assert.strictEqual(hash.length, 64);
  assert.match(hash, /^[0-9a-f]{64}$/);
});

// ──────────────────────────────────────────────────────────────────────────────
// generateSecureToken
// ──────────────────────────────────────────────────────────────────────────────

test("generateSecureToken: generates tokens that are not equal across calls", () => {
  const t1 = generateSecureToken();
  const t2 = generateSecureToken();
  const t3 = generateSecureToken();
  assert.notStrictEqual(t1, t2);
  assert.notStrictEqual(t2, t3);
  assert.notStrictEqual(t1, t3);
});

test("generateSecureToken: default 32-byte token encodes to ~43 base64url chars", () => {
  // 32 bytes → ceil(32 * 4/3) ≈ 43 chars in base64url (no padding)
  const token = generateSecureToken();
  // base64url from 32 bytes: 43 chars (no padding)
  assert.ok(token.length >= 42 && token.length <= 44, `Unexpected length: ${token.length}`);
  // base64url alphabet only
  assert.match(token, /^[A-Za-z0-9_-]+$/);
});

test("generateSecureToken: respects custom byteLength parameter", () => {
  const t16 = generateSecureToken(16);
  const t64 = generateSecureToken(64);
  // 16 bytes → ~22 chars; 64 bytes → ~86 chars
  assert.ok(t16.length >= 20 && t16.length <= 24, `16-byte token length unexpected: ${t16.length}`);
  assert.ok(t64.length >= 84 && t64.length <= 88, `64-byte token length unexpected: ${t64.length}`);
});
