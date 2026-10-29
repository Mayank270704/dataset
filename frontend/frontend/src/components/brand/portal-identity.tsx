"use client";

import Link from "next/link";
import Image from "next/image";

interface PortalIdentityProps {
  compact?: boolean;
  className?: string;
}

export function PortalIdentity({ compact = false, className = "" }: PortalIdentityProps) {
  return (
    <Link
      href="/"
      className={`inline-flex items-center gap-3.5 group text-left ${className}`}
      aria-label="Public Policy Portal home"
    >
      <div className="flex h-12 w-10 shrink-0 items-center justify-center">
        <Image
          src="/logo.png"
          alt="State Emblem of India"
          width={38}
          height={64}
          priority
          className="h-12 w-auto object-contain"
        />
      </div>
      <div className="flex flex-col leading-tight">
        <span className="text-[11px] font-semibold tracking-wide text-[#444444] font-hindi">
          सार्वजनिक नीति पोर्टल
        </span>
        <span className="font-display text-[18px] sm:text-[20px] font-extrabold tracking-tight text-[#171717]">
          PUBLIC POLICY PORTAL
        </span>
        {!compact && (
          <span className="text-[10px] sm:text-[11px] font-medium text-[#666666] tracking-normal">
            National Schemes &amp; Policy Updates Repository
          </span>
        )}
      </div>
    </Link>
  );
}
