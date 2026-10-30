import crypto from "crypto";

export interface SpamAnalysisResult {
  status: "APPROVED" | "FLAGGED" | "REJECTED" | "PENDING";
  score: number;
  flags: string[];
  reason?: string;
  normalizedHash: string;
}

export function normalizeCommentText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, "") // keep unicode letters and numbers
    .replace(/\s+/g, " ")
    .trim();
}

export function computeTextHash(normalizedText: string): string {
  return crypto.createHash("sha256").update(normalizedText).digest("hex");
}

const SUSPICIOUS_URL_PATTERNS = [
  /https?:\/\//i,
  /www\.[a-z0-9-]+\.[a-z]{2,}/i,
  /\b[a-z0-9-]+\.(?:com|net|org|xyz|top|biz|info|cc|tk|ru|online|live|click|shop)\b/i,
  /t\.me\/[a-z0-9_]+/i,
  /wa\.me\/[0-9]+/i,
  /bit\.ly\/[a-z0-9]+/i,
];

const PROMOTIONAL_KEYWORDS = [
  "buy now", "free bonus", "whatsapp number", "call now", "earn money", "crypto",
  "telegram", "discount code", "click here", "subscribe to", "guaranteed returns",
  "lottery", "contact me on", "loan approved", "sugar daddy", "casino"
];

export function analyzeCommentSpam(
  text: string,
  userCreatedAt?: Date,
  isAccountNew = false
): SpamAnalysisResult {
  const flags: string[] = [];
  let score = 0.0;
  const trimmed = text.trim();

  // 1. Minimum and Maximum Length Validation
  if (trimmed.length < 10) {
    return {
      status: "REJECTED",
      score: 1.0,
      flags: ["TOO_SHORT"],
      reason: "Comment is too short (minimum 10 characters).",
      normalizedHash: computeTextHash(normalizeCommentText(trimmed)),
    };
  }

  if (trimmed.length > 2000) {
    return {
      status: "REJECTED",
      score: 1.0,
      flags: ["TOO_LONG"],
      reason: "Comment exceeds maximum length of 2,000 characters.",
      normalizedHash: computeTextHash(normalizeCommentText(trimmed)),
    };
  }

  const normalized = normalizeCommentText(trimmed);
  const normalizedHash = computeTextHash(normalized);

  if (normalized.length < 5) {
    return {
      status: "REJECTED",
      score: 1.0,
      flags: ["MEANINGLESS_CHARS"],
      reason: "Comment does not contain enough recognizable words or characters.",
      normalizedHash,
    };
  }

  // 2. Character repetition check (e.g. "aaaaaaaa", "!!!!!!", "?????")
  if (/(.)\1{6,}/.test(trimmed)) {
    flags.push("REPETITIVE_CHARACTERS");
    score += 0.4;
  }

  // 3. Repeated word patterns (e.g. "good good good good good good")
  const words = normalized.split(/\s+/);
  const uniqueWords = new Set(words);
  if (words.length >= 8 && uniqueWords.size / words.length < 0.3) {
    flags.push("REPETITIVE_WORDS");
    score += 0.5;
  }

  // 4. Excessive uppercase ratio (e.g. ALL CAPS SHOUTING)
  const lettersOnly = trimmed.replace(/[^a-zA-Z]/g, "");
  if (lettersOnly.length >= 15) {
    const uppercaseCount = trimmed.replace(/[^A-Z]/g, "").length;
    if (uppercaseCount / lettersOnly.length > 0.8) {
      flags.push("EXCESSIVE_CAPS");
      score += 0.2;
    }
  }

  // 5. Promotional link / URL detection
  for (const pattern of SUSPICIOUS_URL_PATTERNS) {
    if (pattern.test(trimmed)) {
      flags.push("UNAUTHORIZED_EXTERNAL_LINK");
      score += 0.6;
      break;
    }
  }

  // 6. Known promotional spam phrases
  const lower = trimmed.toLowerCase();
  for (const phrase of PROMOTIONAL_KEYWORDS) {
    if (lower.includes(phrase)) {
      flags.push(`PROMOTIONAL_KEYWORD:${phrase}`);
      score += 0.4;
      break;
    }
  }

  // 7. New account burst / Sybil signal
  if (isAccountNew || (userCreatedAt && Date.now() - userCreatedAt.getTime() < 5 * 60 * 1000)) {
    flags.push("BRAND_NEW_ACCOUNT_ACTIVITY");
    score += 0.2;
  }

  // Decision logic
  if (score >= 0.8) {
    return {
      status: "REJECTED",
      score,
      flags,
      reason: `Automated spam filter rejected submission: ${flags.join(", ")}`,
      normalizedHash,
    };
  }

  if (score >= 0.4) {
    return {
      status: "FLAGGED",
      score,
      flags,
      reason: `Flagged for moderator review due to risk indicators: ${flags.join(", ")}`,
      normalizedHash,
    };
  }

  return {
    status: "APPROVED",
    score,
    flags,
    normalizedHash,
  };
}

