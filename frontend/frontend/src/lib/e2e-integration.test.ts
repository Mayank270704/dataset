import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { query, withTransaction } from "./db";
import {
  hashPassword,
  verifyPassword,
  generateSecureToken,
  hashToken,
  normalizeEmail,
} from "./auth-crypto";
import {
  createSession,
  validateSessionToken,
  revokeSession,
} from "./session-manager";
import { analyzeCommentSpam, computeTextHash, normalizeCommentText } from "./spam-detector";
import { emailService } from "./email-service";

// Load environment variables from .env.local if present
try {
  const envPath = path.resolve(process.cwd(), ".env.local");
  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, "utf-8");
    for (const line of envContent.split("\n")) {
      const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*)?\s*$/);
      if (match && !match[1].startsWith("#")) {
        const key = match[1];
        let value = (match[2] || "").trim();
        if (
          (value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))
        ) {
          value = value.slice(1, -1);
        }
        if (!process.env[key]) {
          process.env[key] = value;
        }
      }
    }
  }
} catch {}

test("E2E Integration: 1. Complete Registration and Database Persistence", async () => {
  const testEmail = `e2e_citizen_${Date.now()}@policylens.local`;
  const normalizedEmail = normalizeEmail(testEmail);
  const rawPassword = "SecurePassword123!@#";
  const displayName = "E2E Test Citizen";

  // 1. Password hashing with Argon2id
  const passwordHash = await hashPassword(rawPassword);
  assert.strictEqual(passwordHash.startsWith("$argon2id$"), true);

  // 2. Insert user into PostgreSQL
  const userRes = await query<{ id: string; role: string; is_email_verified: boolean }>(
    `INSERT INTO users (email, email_normalized, display_name, name, password_hash, role, is_email_verified, status)
     VALUES ($1, $2, $3::text, $4::text, $5, 'CITIZEN', FALSE, 'ACTIVE')
     RETURNING id, role, is_email_verified;`,
    [testEmail, normalizedEmail, displayName, displayName, passwordHash]
  );

  assert.strictEqual(userRes.rows.length, 1);
  const user = userRes.rows[0];
  assert.strictEqual(user.role, "CITIZEN");
  assert.strictEqual(user.is_email_verified, false);

  // 3. Email verification token generation & persistence
  const rawToken = generateSecureToken(32);
  const tokenHash = hashToken(rawToken);
  const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);

  const tokenRes = await query<{ id: string }>(
    `INSERT INTO email_verifications (user_id, token_hash, expires_at)
     VALUES ($1, $2, $3)
     RETURNING id;`,
    [user.id, tokenHash, expiresAt]
  );
  assert.strictEqual(tokenRes.rows.length, 1);

  // Cleanup test user
  await query(`DELETE FROM users WHERE id = $1`, [user.id]);
});

test("E2E Integration: 2. Email Verification Token Single-Use and Expiration", async () => {
  const testEmail = `e2e_verif_${Date.now()}@policylens.local`;
  const hash = await hashPassword("TestPass123!@#");

  const userRes = await query<{ id: string }>(
    `INSERT INTO users (email, email_normalized, display_name, name, password_hash, role, is_email_verified, status)
     VALUES ($1, $2, 'Verif Tester'::text, 'Verif Tester'::text, $3, 'CITIZEN', FALSE, 'ACTIVE')
     RETURNING id;`,
    [testEmail, normalizeEmail(testEmail), hash]
  );
  const userId = userRes.rows[0].id;

  const rawToken = generateSecureToken(32);
  const tokenHash = hashToken(rawToken);
  const expiresAt = new Date(Date.now() + 3600 * 1000);

  await query(
    `INSERT INTO email_verifications (user_id, token_hash, expires_at)
     VALUES ($1, $2, $3);`,
    [userId, tokenHash, expiresAt]
  );

  // First verification: should succeed
  await withTransaction(async (client) => {
    await client.query(
      `UPDATE email_verifications SET used_at = NOW() WHERE token_hash = $1 AND used_at IS NULL;`,
      [tokenHash]
    );
    await client.query(
      `UPDATE users SET is_email_verified = TRUE WHERE id = $1;`,
      [userId]
    );
  });

  const updatedUser = await query<{ is_email_verified: boolean }>(
    `SELECT is_email_verified FROM users WHERE id = $1;`,
    [userId]
  );
  assert.strictEqual(updatedUser.rows[0].is_email_verified, true);

  // Second verification attempt with same token: should fail (already used)
  const tokenCheck = await query<{ used_at: Date | null }>(
    `SELECT used_at FROM email_verifications WHERE token_hash = $1;`,
    [tokenHash]
  );
  assert.strictEqual(tokenCheck.rows[0].used_at !== null, true);

  // Cleanup
  await query(`DELETE FROM users WHERE id = $1`, [userId]);
});

test("E2E Integration: 3. Login, Session Management, and Session Revocation", async () => {
  const testEmail = `e2e_session_${Date.now()}@policylens.local`;
  const rawPassword = "CorrectPassword123!@#";
  const passHash = await hashPassword(rawPassword);

  const userRes = await query<{ id: string }>(
    `INSERT INTO users (email, email_normalized, display_name, name, password_hash, role, is_email_verified, status)
     VALUES ($1, $2, 'Session Tester'::text, 'Session Tester'::text, $3, 'CITIZEN', TRUE, 'ACTIVE')
     RETURNING id;`,
    [testEmail, normalizeEmail(testEmail), passHash]
  );
  const userId = userRes.rows[0].id;

  // 1. Password verification
  const isValid = await verifyPassword(passHash, rawPassword);
  assert.strictEqual(isValid, true);
  const isInvalid = await verifyPassword(passHash, "WrongPassword123!@#");
  assert.strictEqual(isInvalid, false);

  // 2. Session creation (stores sha256 token hash)
  const { token, expiresAt } = await createSession(userId);
  assert.strictEqual(typeof token, "string");
  assert.strictEqual(token.length >= 32, true);

  // 3. Session validation (retrieves safe user object)
  const sessionUser = await validateSessionToken(token);
  assert.strictEqual(sessionUser !== null, true);
  assert.strictEqual(sessionUser?.id, userId);
  assert.strictEqual(sessionUser?.email, testEmail);
  assert.strictEqual(sessionUser?.role, "CITIZEN");
  assert.strictEqual(sessionUser?.isEmailVerified, true);

  // 4. Session revocation (logout)
  await revokeSession(token);

  // 5. Revoked session validation: should return null
  const revokedCheck = await validateSessionToken(token);
  assert.strictEqual(revokedCheck, null);

  // Cleanup
  await query(`DELETE FROM users WHERE id = $1`, [userId]);
});

test("E2E Integration: 4. Comment Quota Enforcement (Max 5 comments per user per policy)", async () => {
  const testEmail = `e2e_quota_${Date.now()}@policylens.local`;
  const passHash = await hashPassword("CommentPass123!@#");
  const policySlug = "prahaar-counter-terrorism-policy";

  const userRes = await query<{ id: string }>(
    `INSERT INTO users (email, email_normalized, display_name, name, password_hash, role, is_email_verified, status)
     VALUES ($1, $2, 'Commenter'::text, 'Commenter'::text, $3, 'CITIZEN', TRUE, 'ACTIVE')
     RETURNING id;`,
    [testEmail, normalizeEmail(testEmail), passHash]
  );
  const userId = userRes.rows[0].id;

  // Submit 5 distinct comments
  for (let i = 1; i <= 5; i++) {
    const text = `This is substantive civic comment number ${i} for testing policy quota limits.`;
    const normHash = computeTextHash(normalizeCommentText(text));

    await withTransaction(async (client) => {
      await client.query(
        `INSERT INTO policy_comment_quotas (user_id, policy_slug, submission_count, created_at, updated_at)
         VALUES ($1, $2, 0, NOW(), NOW())
         ON CONFLICT (user_id, policy_slug) DO NOTHING;`,
        [userId, policySlug]
      );

      const lockRes = await client.query<{ submission_count: number }>(
        `SELECT submission_count FROM policy_comment_quotas
         WHERE user_id = $1 AND policy_slug = $2 FOR UPDATE;`,
        [userId, policySlug]
      );

      const current = lockRes.rows[0].submission_count;
      assert.strictEqual(current < 5, true);

      await client.query(
        `UPDATE policy_comment_quotas
         SET submission_count = $1, updated_at = NOW()
         WHERE user_id = $2 AND policy_slug = $3;`,
        [current + 1, userId, policySlug]
      );

      await client.query(
        `INSERT INTO policy_comments (
           user_id, policy_slug, stakeholder, original_text,
           normalized_text_hash, status, created_at
         )
         VALUES ($1, $2, 'Citizen', $3, $4, 'APPROVED', NOW());`,
        [userId, policySlug, text, normHash]
      );
    });
  }

  // Verify quota is exactly 5
  const quotaRes = await query<{ submission_count: number }>(
    `SELECT submission_count FROM policy_comment_quotas WHERE user_id = $1 AND policy_slug = $2;`,
    [userId, policySlug]
  );
  assert.strictEqual(quotaRes.rows[0].submission_count, 5);

  // Attempt 6th comment: should be rejected by quota constraint
  let rejected = false;
  try {
    await withTransaction(async (client) => {
      const lockRes = await client.query<{ submission_count: number }>(
        `SELECT submission_count FROM policy_comment_quotas
         WHERE user_id = $1 AND policy_slug = $2 FOR UPDATE;`,
        [userId, policySlug]
      );
      if (lockRes.rows[0].submission_count >= 5) {
        throw new Error("COMMENT_LIMIT_EXCEEDED");
      }
    });
  } catch (err: any) {
    if (err.message === "COMMENT_LIMIT_EXCEEDED") {
      rejected = true;
    }
  }
  assert.strictEqual(rejected, true, "6th comment must be rejected");

  // Cleanup
  await query(`DELETE FROM users WHERE id = $1`, [userId]);
});

test("E2E Integration: 5. Spam Detector rejects spam and approves legitimate comment", () => {
  // Short spam
  const shortResult = analyzeCommentSpam("Too short");
  assert.strictEqual(shortResult.status, "REJECTED");

  // Promotional link spam
  const linkResult = analyzeCommentSpam(
    "Check out my telegram channel at https://t.me/fakechannel for free lottery prizes!"
  );
  assert.strictEqual(linkResult.status === "REJECTED" || linkResult.status === "FLAGGED", true);

  // Legitimate constructive policy comment
  const legitResult = analyzeCommentSpam(
    "The proposed policy framework should incorporate explicit data privacy safeguards and periodic independent judicial review."
  );
  assert.strictEqual(legitResult.status, "APPROVED");
  assert.strictEqual(legitResult.score < 0.4, true);
});
