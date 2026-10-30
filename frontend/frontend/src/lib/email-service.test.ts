import test from "node:test";
import assert from "node:assert/strict";
import { AppEmailService } from "./email-service";

test("EmailService: dispatches via Resend when RESEND_API_KEY is configured", async () => {
  const originalEnv = { ...process.env };
  const originalFetch = globalThis.fetch;

  try {
    process.env.RESEND_API_KEY = "re_test_dummy_key_123";
    process.env.EMAIL_FROM = "PolicyLens <onboarding@resend.dev>";

    let requestedUrl = "";
    let requestedMethod = "";
    let authHeader = "";
    let requestBody: any = null;

    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const urlStr = url.toString();
      if (init?.method === "GET") {
        // Status check GET request
        return {
          ok: false,
          status: 401,
          json: async () => ({
            statusCode: 401,
            message: "This API key is restricted to only send emails",
          }),
        } as Response;
      }

      requestedUrl = urlStr;
      requestedMethod = init?.method || "";
      authHeader = (init?.headers as Record<string, string>)?.[
        "Authorization"
      ] || "";
      requestBody = JSON.parse((init?.body as string) || "{}");

      return {
        ok: true,
        status: 200,
        json: async () => ({ id: "resend_msg_test_abc123" }),
      } as Response;
    }) as typeof fetch;

    const service = new AppEmailService();
    const result = await service.sendEmail({
      to: "citizen@policylens.gov.in",
      subject: "Test Civic Notification",
      text: "Plaintext body",
      html: "<p>HTML body</p>",
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.accepted, true);
    assert.strictEqual(result.messageId, "resend_msg_test_abc123");
    assert.strictEqual(requestedUrl, "https://api.resend.com/emails");
    assert.strictEqual(requestedMethod, "POST");
    assert.strictEqual(authHeader, "Bearer re_test_dummy_key_123");
    assert.strictEqual(requestBody.from, "PolicyLens <onboarding@resend.dev>");
    assert.deepStrictEqual(requestBody.to, ["citizen@policylens.gov.in"]);
    assert.strictEqual(requestBody.subject, "Test Civic Notification");
  } finally {
    process.env = originalEnv;
    globalThis.fetch = originalFetch;
  }
});

test("EmailService: strips quotes around EMAIL_FROM if present in env", async () => {
  const originalEnv = { ...process.env };
  const originalFetch = globalThis.fetch;

  try {
    process.env.RESEND_API_KEY = "re_test_key";
    process.env.EMAIL_FROM = '"PolicyLens <onboarding@resend.dev>"';

    let capturedFrom = "";
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      if (init?.method === "GET") {
        return { ok: false, status: 401, json: async () => ({}) } as Response;
      }
      const body = JSON.parse((init?.body as string) || "{}");
      capturedFrom = body.from;
      return {
        ok: true,
        status: 200,
        json: async () => ({ id: "msg_quoted_env_test" }),
      } as Response;
    }) as typeof fetch;

    const service = new AppEmailService();
    await service.sendEmail({
      to: "citizen@policylens.gov.in",
      subject: "Test",
      text: "Test",
      html: "<p>Test</p>",
    });

    assert.strictEqual(capturedFrom, "PolicyLens <onboarding@resend.dev>");
  } finally {
    process.env = originalEnv;
    globalThis.fetch = originalFetch;
  }
});

test("EmailService: handles Resend error response safely without throwing", async () => {
  const originalEnv = { ...process.env };
  const originalFetch = globalThis.fetch;

  try {
    process.env.RESEND_API_KEY = "re_test_key";

    globalThis.fetch = (async () => {
      return {
        ok: false,
        status: 422,
        statusText: "Unprocessable Entity",
        json: async () => ({
          statusCode: 422,
          message: "The from field must be a valid email address.",
          name: "validation_error",
        }),
      } as Response;
    }) as typeof fetch;

    const service = new AppEmailService();
    const result = await service.sendEmail({
      to: "citizen@policylens.gov.in",
      subject: "Test Failure",
      text: "Test",
      html: "<p>Test</p>",
    });

    assert.strictEqual(result.success, false);
    assert.strictEqual(result.accepted, false);
    assert.strictEqual(
      result.error,
      "The from field must be a valid email address."
    );
  } finally {
    process.env = originalEnv;
    globalThis.fetch = originalFetch;
  }
});

test("EmailService: fails with clear error when RESEND_API_KEY is unset (no false success)", async () => {
  const originalEnv = { ...process.env };
  try {
    delete process.env.RESEND_API_KEY;

    const service = new AppEmailService();
    const result = await service.sendEmail({
      to: "citizen@policylens.gov.in",
      subject: "Unconfigured Test",
      text: "Body",
      html: "<p>Body</p>",
    });

    assert.strictEqual(result.success, false);
    assert.strictEqual(result.accepted, false);
    assert.match(result.error || "", /RESEND_API_KEY.*not configured/);
  } finally {
    process.env = originalEnv;
  }
});

test("EmailService: sendVerificationEmail formats verification link and dispatches successfully", async () => {
  const originalEnv = { ...process.env };
  const originalFetch = globalThis.fetch;

  try {
    process.env.RESEND_API_KEY = "re_dummy";
    process.env.NEXT_PUBLIC_SITE_URL = "https://policylens.gov.in";

    let capturedHtml = "";
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      if (init?.method === "GET") {
        return { ok: false, status: 401, json: async () => ({}) } as Response;
      }
      const body = JSON.parse((init?.body as string) || "{}");
      capturedHtml = body.html;
      return {
        ok: true,
        status: 200,
        json: async () => ({ id: "msg_verification_123" }),
      } as Response;
    }) as typeof fetch;

    const service = new AppEmailService();
    const result = await service.sendVerificationEmail(
      "citizen@policylens.gov.in",
      "secure_token_abc_xyz"
    );

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.accepted, true);
    assert.strictEqual(result.messageId, "msg_verification_123");
    assert.match(
      capturedHtml,
      /https:\/\/policylens\.gov\.in\/verify-email\?token=secure_token_abc_xyz/
    );
  } finally {
    process.env = originalEnv;
    globalThis.fetch = originalFetch;
  }
});
