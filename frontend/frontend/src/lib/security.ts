import path from "path";
import fs from "fs";
import { officialPolicies } from "@/features/feed/demo-data";

/**
 * Strict slug format for policy identifiers.
 * Only lowercase alphanumeric characters separated by single hyphens.
 * Prevents directory traversal sequences (e.g. .., /, \), control characters, and special characters.
 */
const POLICY_SLUG_REGEX = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export const BENCHMARK_PILOT_SLUG = "digital-public-infrastructure-india-stack";

export class SecurityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SecurityError";
  }
}

/**
 * Validates that a policy slug matches the strict format:
 * - Must be a string.
 * - Must be between 3 and 100 characters in length.
 * - Must strictly match lowercase kebab-case (no dots, slashes, or special characters).
 */
export function isValidSlugFormat(slug: unknown): slug is string {
  if (typeof slug !== "string") {
    return false;
  }
  if (slug.length < 3 || slug.length > 100) {
    return false;
  }
  return POLICY_SLUG_REGEX.test(slug);
}

/**
 * Verifies that the slug corresponds to a known policy registered in the application
 * or the authorized pilot benchmark record.
 */
export function isKnownPolicySlug(slug: string): boolean {
  if (!isValidSlugFormat(slug)) {
    return false;
  }
  if (slug === BENCHMARK_PILOT_SLUG) {
    return true;
  }
  return officialPolicies.some((p) => p.slug === slug);
}

/**
 * Resolves the canonical base data directory for the repository.
 */
export function getBaseDataDirectory(): string {
  const projectRoot = path.resolve(process.cwd(), "..");
  if (fs.existsSync(path.join(process.cwd(), "data"))) {
    return path.resolve(process.cwd(), "data");
  }
  return path.resolve(projectRoot, "data");
}

/**
 * Resolves a safe file path inside one of the recognized data subdirectories:
 * - Ensures the slug passes format validation.
 * - Resolves the canonical target directory.
 * - Asserts that the canonical file path cannot escape the intended directory boundary.
 *
 * @throws SecurityError if slug format is invalid or directory traversal is detected.
 */
export function resolveSafeDataFilePath(
  subfolder: "submissions" | "jobs" | "analysis",
  slug: string
): string {
  if (!isValidSlugFormat(slug)) {
    throw new SecurityError(`Invalid policy slug format: '${String(slug)}'`);
  }

  const baseDataDir = getBaseDataDirectory();
  const targetDir = path.resolve(baseDataDir, subfolder);

  // Canonicalize directory with trailing path separator for strict prefix matching
  const canonicalTargetDir = targetDir.endsWith(path.sep)
    ? targetDir
    : targetDir + path.sep;

  // Canonicalize the target file path
  const targetFile = path.resolve(targetDir, `${slug}.json`);

  // Reject if the resolved path escapes the intended subfolder
  if (!targetFile.startsWith(canonicalTargetDir)) {
    throw new SecurityError(
      `Directory traversal attempt blocked: '${slug}' escapes '${subfolder}' directory`
    );
  }

  return targetFile;
}
