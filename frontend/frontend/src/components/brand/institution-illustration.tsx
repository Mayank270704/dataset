"use client";

import React from "react";

/**
 * Clean architectural CSS/SVG illustration of an Indian public institution facade.
 * Features classical columns, portico pediment, dome with pinnacle, Ashoka Chakra,
 * and Indian tricolour pennant. Replaces random stock photos with clean vector art.
 */
export function InstitutionIllustration({ className = "" }: { className?: string }) {
  return (
    <div className={`relative overflow-hidden rounded-xl border border-[#e5e7eb] bg-[#f8fafc] p-4 flex items-center justify-center ${className}`}>
      <svg
        viewBox="0 0 540 220"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-auto max-h-[190px]"
        aria-label="Illustration of Indian Public Policy Institution"
      >
        {/* Soft Sky Background */}
        <rect width="540" height="220" rx="8" fill="#F4F6F8" />
        <circle cx="270" cy="110" r="90" fill="#E8EEF5" fillOpacity="0.6" />

        {/* Central Dome & Cupola */}
        <path d="M220 90C220 62 242 40 270 40C298 40 320 62 320 90H220Z" fill="#143454" />
        {/* Dome Ribs */}
        <line x1="270" y1="40" x2="270" y2="90" stroke="#FFFFFF" strokeWidth="1.5" strokeOpacity="0.4" />
        <path d="M245 48C252 60 255 75 255 90" stroke="#FFFFFF" strokeWidth="1" strokeOpacity="0.3" />
        <path d="M295 48C288 60 285 75 285 90" stroke="#FFFFFF" strokeWidth="1" strokeOpacity="0.3" />

        {/* Dome Pinnacle & Tricolour Finial */}
        <rect x="268" y="24" width="4" height="16" fill="#143454" />
        <circle cx="270" cy="22" r="3.5" fill="#E87518" />
        {/* Tricolour Flag */}
        <line x1="270" y1="8" x2="270" y2="24" stroke="#444444" strokeWidth="1.5" />
        <path d="M270 8H288L284 13L288 18H270V8Z" fill="#E87518" />
        <rect x="270" y="11.5" width="16" height="3" fill="#FFFFFF" />
        <rect x="270" y="14.5" width="16" height="3.5" fill="#16834A" />

        {/* Upper Attic Architrave */}
        <rect x="200" y="88" width="140" height="12" fill="#204A74" rx="1" />

        {/* Main Triangular Pediment */}
        <path d="M170 100L270 65L370 100H170Z" fill="#143454" />
        {/* Pediment Relief with Chakra */}
        <circle cx="270" cy="85" r="9" fill="#1A4168" stroke="#FFFFFF" strokeWidth="1" />
        <circle cx="270" cy="85" r="2.5" fill="#E87518" />

        {/* Entablature */}
        <rect x="60" y="100" width="420" height="14" fill="#0E2A47" />
        <rect x="70" y="104" width="400" height="2" fill="#E87518" />

        {/* Left Wing Colonnade */}
        <rect x="70" y="114" width="110" height="66" fill="#DCE3EC" />
        {/* Left Columns */}
        {[80, 100, 120, 140, 160].map((x) => (
          <g key={`l-col-${x}`}>
            <rect x={x} y="114" width="10" height="66" fill="#F8FAFC" stroke="#CBD5E1" strokeWidth="1" />
            <rect x={x - 2} y="114" width="14" height="4" fill="#143454" />
            <rect x={x - 2} y="176" width="14" height="4" fill="#143454" />
          </g>
        ))}

        {/* Right Wing Colonnade */}
        <rect x="360" y="114" width="110" height="66" fill="#DCE3EC" />
        {/* Right Columns */}
        {[370, 390, 410, 430, 450].map((x) => (
          <g key={`r-col-${x}`}>
            <rect x={x} y="114" width="10" height="66" fill="#F8FAFC" stroke="#CBD5E1" strokeWidth="1" />
            <rect x={x - 2} y="114" width="14" height="4" fill="#143454" />
            <rect x={x - 2} y="176" width="14" height="4" fill="#143454" />
          </g>
        ))}

        {/* Central Grand Portico */}
        <rect x="180" y="114" width="180" height="66" fill="#CBD5E1" />
        {/* Central Entrance Portal */}
        <path d="M250 145C250 134 259 125 270 125C281 125 290 134 290 145V180H250V145Z" fill="#0E2A47" />
        {/* Portico Columns */}
        {[190, 215, 240, 290, 315, 340].map((x) => (
          <g key={`c-col-${x}`}>
            <rect x={x} y="114" width="11" height="66" fill="#FFFFFF" stroke="#94A3B8" strokeWidth="1" />
            <rect x={x - 2} y="114" width="15" height="5" fill="#E87518" />
            <rect x={x - 2} y="175" width="15" height="5" fill="#143454" />
          </g>
        ))}

        {/* Podium Base Steps */}
        <rect x="50" y="180" width="440" height="10" fill="#64748B" rx="1" />
        <rect x="40" y="190" width="460" height="8" fill="#475569" rx="1" />
        <rect x="30" y="198" width="480" height="8" fill="#334155" rx="1" />

        {/* Foreground Plaza Line */}
        <line x1="20" y1="206" x2="520" y2="206" stroke="#E87518" strokeWidth="2" />
        <rect x="230" y="210" width="80" height="3" fill="#16834A" rx="1.5" />
      </svg>
    </div>
  );
}
