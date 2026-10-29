"use client";

import { PortalIdentity } from "./portal-identity";

export function Wordmark({ compact = false }: { compact?: boolean }) {
  return <PortalIdentity compact={compact} />;
}
