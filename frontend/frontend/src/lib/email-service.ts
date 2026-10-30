export interface EmailMessage {
  to: string;
  subject: string;
  html: string;
  text: string;
}

export type DeliveryStatus =
  | "accepted_by_provider"
  | "queued"
  | "sent"
  | "delivered"
  | "delivery_delayed"
  | "bounced"
  | "complained"
  | "failed";

export interface EmailDeliveryResult {
  success: boolean;
  accepted: boolean;
  delivered: boolean;
  deliveryStatus: DeliveryStatus;
  messageId?: string;
  details?: string;
  error?: string;
  statusCode?: number;
}

export interface EmailProvider {
  sendEmail(message: EmailMessage): Promise<EmailDeliveryResult>;
}

function cleanEnvString(val?: string): string {
  if (!val) return "";
  const trimmed = val.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1).trim();
  }
  return trimmed;
}

/**
 * Resend Email Service Integration.
 * Strictly adheres to truth in delivery:
 * - Does not interpret HTTP 200 or message ID as confirmed mailbox delivery.
 * - Confirms provider acceptance vs mailbox delivery.
 * - Queries delivery status via Resend API when supported.
 * - Never returns false success or mocks responses.
 */
export class AppEmailService implements EmailProvider {
  private getResendApiKey(): string {
    return cleanEnvString(process.env.RESEND_API_KEY);
  }

  private getFromAddress(): string {
    const from = cleanEnvString(process.env.EMAIL_FROM);
    return from || "PolicyLens <onboarding@resend.dev>";
  }

  /**
   * Dispatches an email using the Resend REST API (https://api.resend.com/emails).
   */
  async sendEmail(message: EmailMessage): Promise<EmailDeliveryResult> {
    const resendApiKey = this.getResendApiKey();
    const fromAddress = this.getFromAddress();

    if (!resendApiKey) {
      const err = "RESEND_API_KEY environment variable is not configured. Email delivery aborted.";
      console.error(`[EMAIL_SERVICE] Dispatch aborted for ${message.to}: ${err}`);
      return {
        success: false,
        accepted: false,
        delivered: false,
        deliveryStatus: "failed",
        error: err,
        statusCode: 500,
      };
    }

    try {
      const response = await fetch("https://api.resend.com/emails", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${resendApiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          from: fromAddress,
          to: [message.to],
          subject: message.subject,
          html: message.html,
          text: message.text,
        }),
      });

      const data = (await response.json().catch(() => ({}))) as {
        id?: string;
        message?: string;
        name?: string;
        statusCode?: number;
      };

      if (!response.ok || !data.id) {
        const errorMessage = data.message || `HTTP ${response.status} ${response.statusText}`;
        console.error(`[EMAIL_SERVICE] Resend API rejected dispatch for ${message.to} (Status ${response.status}): ${errorMessage}`);
        return {
          success: false,
          accepted: false,
          delivered: false,
          deliveryStatus: "failed",
          error: errorMessage,
          statusCode: response.status,
        };
      }

      console.log(`[EMAIL_SERVICE] Provider accepted email for ${message.to} (Message ID: ${data.id})`);

      // Attempt to inspect actual delivery status via supported Resend API
      const statusCheck = await this.checkDeliveryStatus(data.id, resendApiKey);

      return {
        success: true,
        accepted: true,
        delivered: statusCheck.delivered,
        deliveryStatus: statusCheck.deliveryStatus,
        messageId: data.id,
        details: statusCheck.details,
      };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Network error during email dispatch";
      console.error(`[EMAIL_SERVICE] Exception dispatching email to ${message.to}: ${errorMsg}`);
      return {
        success: false,
        accepted: false,
        delivered: false,
        deliveryStatus: "failed",
        error: errorMsg,
        statusCode: 502,
      };
    }
  }

  /**
   * Retrieves actual delivery status of an accepted email using the Resend GET /emails/{id} API.
   */
  async checkDeliveryStatus(
    messageId: string,
    apiKey?: string
  ): Promise<{ delivered: boolean; deliveryStatus: DeliveryStatus; details: string }> {
    const key = apiKey || this.getResendApiKey();
    if (!key) {
      return {
        delivered: false,
        deliveryStatus: "accepted_by_provider",
        details: "API key unset; cannot verify delivery status.",
      };
    }

    try {
      const res = await fetch(`https://api.resend.com/emails/${encodeURIComponent(messageId)}`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${key}`,
        },
      });

      const data = (await res.json().catch(() => ({}))) as {
        last_event?: string;
        message?: string;
        statusCode?: number;
      };

      if (res.ok && data.last_event) {
        const lastEvent = data.last_event as DeliveryStatus;
        return {
          delivered: lastEvent === "delivered",
          deliveryStatus: lastEvent,
          details: `Resend event status: ${lastEvent}`,
        };
      }

      // If API key is sending-only (HTTP 401 restricted_api_key)
      if (res.status === 401 || data.message?.includes("restricted to only send")) {
        return {
          delivered: false,
          deliveryStatus: "accepted_by_provider",
          details:
            "Provider accepted and queued message. Resend API key has Sending Access only; status polling requires Full Access key.",
        };
      }

      return {
        delivered: false,
        deliveryStatus: "accepted_by_provider",
        details: data.message || "Accepted and queued by provider for delivery to destination MTA.",
      };
    } catch (err: any) {
      return {
        delivered: false,
        deliveryStatus: "accepted_by_provider",
        details: `Could not poll status: ${err?.message || "Unknown error"}`,
      };
    }
  }

  /**
   * Sends email verification with accurate host origin.
   * Guarantees that localhost:3000 is never used in production.
   */
  async sendVerificationEmail(
    email: string,
    token: string,
    baseUrl?: string
  ): Promise<EmailDeliveryResult> {
    let origin = baseUrl;

    if (!origin) {
      origin =
        process.env.APP_URL ||
        process.env.NEXT_PUBLIC_SITE_URL ||
        process.env.NEXT_PUBLIC_APP_URL;
    }

    // Strict safeguard: Never output localhost in production
    if (
      process.env.NODE_ENV === "production" &&
      (!origin || origin.includes("localhost"))
    ) {
      origin = process.env.NEXT_PUBLIC_SITE_URL || "https://policylens.gov.in";
    }

    if (!origin) {
      origin = "http://localhost:3000";
    }

    const cleanOrigin = origin.replace(/\/$/, "");
    const verificationUrl = `${cleanOrigin}/verify-email?token=${encodeURIComponent(token)}`;

    return this.sendEmail({
      to: email,
      subject: "Verify your email for PolicyLens",
      text: `Welcome to PolicyLens. Please verify your email by clicking the link: ${verificationUrl}`,
      html: `
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 24px; border: 1px solid #e2e8f0; border-radius: 8px;">
          <h2 style="color: #143454; margin-top: 0;">PolicyLens Verification</h2>
          <p style="color: #334155; line-height: 1.6;">Thank you for registering on PolicyLens. Please verify your email address to participate in civic discussions.</p>
          <div style="margin: 28px 0;">
            <a href="${verificationUrl}" style="background-color: #143454; color: #ffffff !important; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600; display: inline-block;">
              Verify Email Address
            </a>
          </div>
          <p style="color: #64748b; font-size: 13px; line-height: 1.5;">This link will expire in 24 hours. If the button above does not work, copy and paste this link into your browser:<br/><span style="color: #143454; word-break: break-all;">${verificationUrl}</span></p>
          <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;" />
          <p style="color: #94a3b8; font-size: 11px;">If you did not create an account on PolicyLens, you can safely ignore this email.</p>
        </div>
      `,
    });
  }
}

export const emailService = new AppEmailService();
