"use client";

import React from "react";

/**
 * Original landmark-inspired architectural icon inspired by the Ashoka Stambh capital
 * and Indian civic architecture. Features a stylized capital silhouette, circular abacus
 * with Chakra motif, fluted lotus base, and subtle tricolour accent.
 */
export function StambhIcon({ size = 36, className = "" }: { size?: number; className?: string }) {
  return (
    <div className={`relative inline-flex items-center justify-center shrink-0 ${className}`} style={{ width: size, height: size }}>
      <svg
        viewBox="0 0 48 48"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
        aria-hidden="true"
      >
        {/* Soft background glow */}
        <rect width="48" height="48" rx="10" fill="currentColor" fillOpacity="0.04" />

        {/* Capital Top - Stylized Quad-direction architectural crown */}
        <path
          d="M16 12C16 9.5 19 8 24 8C29 8 32 9.5 32 12C32 14.5 30 16 28 16H20C18 16 16 14.5 16 12Z"
          fill="#171717"
        />
        {/* Central pinnacle motif */}
        <path d="M24 5L25.5 8H22.5L24 5Z" fill="#E87518" />

        {/* Left and Right Guardian Silhouettes */}
        <path
          d="M13 14C11.5 14 11 15.5 12 17C13 18.5 15 18 16 17L15 14H13Z"
          fill="#171717"
        />
        <path
          d="M35 14C36.5 14 37 15.5 36 17C35 18.5 33 18 32 17L33 14H35Z"
          fill="#171717"
        />

        {/* Abacus Band */}
        <rect x="12" y="19" width="24" height="6" rx="2" fill="#143454" />

        {/* Ashoka Chakra Wheel Motif in Abacus */}
        <circle cx="24" cy="22" r="2.2" stroke="#FFFFFF" strokeWidth="0.8" fill="none" />
        <circle cx="24" cy="22" r="0.6" fill="#FFFFFF" />
        {/* Wheel Spokes */}
        <line x1="24" y1="19.8" x2="24" y2="24.2" stroke="#FFFFFF" strokeWidth="0.5" />
        <line x1="21.8" y1="22" x2="26.2" y2="22" stroke="#FFFFFF" strokeWidth="0.5" />
        <line x1="22.5" y1="20.5" x2="25.5" y2="23.5" stroke="#FFFFFF" strokeWidth="0.5" />
        <line x1="25.5" y1="20.5" x2="22.5" y2="23.5" stroke="#FFFFFF" strokeWidth="0.5" />

        {/* Fluted Lotus Base */}
        <path
          d="M15 27C17 31 19 33 24 33C29 33 31 31 33 27H15Z"
          fill="#171717"
        />
        {/* Lotus fluting lines */}
        <line x1="20" y1="28" x2="21" y2="32" stroke="#FFFFFF" strokeWidth="0.6" strokeOpacity="0.4" />
        <line x1="24" y1="28" x2="24" y2="32.5" stroke="#FFFFFF" strokeWidth="0.6" strokeOpacity="0.4" />
        <line x1="28" y1="28" x2="27" y2="32" stroke="#FFFFFF" strokeWidth="0.6" strokeOpacity="0.4" />

        {/* Pedestal plinth */}
        <rect x="11" y="35" width="26" height="3" rx="1.5" fill="#171717" />
        <rect x="9" y="39" width="30" height="2" rx="1" fill="#444444" />

        {/* Tricolour Accent Line at Bottom */}
        <rect x="13" y="43" width="7" height="1.8" rx="0.9" fill="#E87518" />
        <rect x="21" y="43" width="6" height="1.8" rx="0.9" fill="#FFFFFF" stroke="#D1D5DB" strokeWidth="0.3" />
        <rect x="28" y="43" width="7" height="1.8" rx="0.9" fill="#16834A" />
      </svg>
    </div>
  );
}
