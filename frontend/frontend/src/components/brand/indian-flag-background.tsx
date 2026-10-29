"use client";

import React from "react";

/**
 * IndianFlagBackground
 *
 * Renders an authentic Indian National Flag (Tiranga) overlay in the background
 * with subtle, low-opacity styling:
 * - Top band: India Saffron (#FF9933)
 * - Middle band: White (#FFFFFF) with a 24-spoke Navy Blue Ashoka Chakra (#143454)
 * - Bottom band: India Green (#138808)
 * - Ambient atmospheric gradient wash across viewport edges
 * - Soft radial edge feathering so it gracefully fades into the portal theme
 * - Zero pointer event interference (pointer-events-none)
 */
export function IndianFlagBackground() {
  return (
    <div
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none"
    >
      {/* 1. Ambient atmospheric tricolour glow on viewport edges */}
      <div
        className="absolute -top-24 left-1/2 -translate-x-1/2 w-[120vw] h-80 opacity-[0.045] dark:opacity-[0.03] blur-3xl pointer-events-none"
        style={{
          background: "radial-gradient(ellipse at top, #FF9933 0%, rgba(255, 153, 51, 0.4) 45%, transparent 75%)"
        }}
      />
      <div
        className="absolute -bottom-24 left-1/2 -translate-x-1/2 w-[120vw] h-80 opacity-[0.045] dark:opacity-[0.03] blur-3xl pointer-events-none"
        style={{
          background: "radial-gradient(ellipse at bottom, #138808 0%, rgba(19, 136, 8, 0.4) 45%, transparent 75%)"
        }}
      />

      {/* 2. Centered Indian National Flag Watermark */}
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div
          className="w-[90vw] max-w-[840px] aspect-[3/2] opacity-[0.055] dark:opacity-[0.04] transition-opacity duration-300"
          style={{
            maskImage: "radial-gradient(ellipse 88% 78% at 50% 50%, black 45%, rgba(0, 0, 0, 0.6) 70%, transparent 98%)",
            WebkitMaskImage: "radial-gradient(ellipse 88% 78% at 50% 50%, black 45%, rgba(0, 0, 0, 0.6) 70%, transparent 98%)"
          }}
        >
          <svg
            viewBox="0 0 900 600"
            className="w-full h-full shadow-2xl rounded-2xl"
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              {/* Subtle lighting / fabric ripple filter */}
              <linearGradient id="flag-sheen" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.08" />
                <stop offset="25%" stopColor="#000000" stopOpacity="0.04" />
                <stop offset="50%" stopColor="#ffffff" stopOpacity="0.1" />
                <stop offset="75%" stopColor="#000000" stopOpacity="0.04" />
                <stop offset="100%" stopColor="#ffffff" stopOpacity="0.06" />
              </linearGradient>
            </defs>

            {/* Saffron Band */}
            <rect x="0" y="0" width="900" height="200" fill="#FF9933" />

            {/* White Band */}
            <rect x="0" y="200" width="900" height="200" fill="#FFFFFF" />

            {/* Green Band */}
            <rect x="0" y="400" width="900" height="200" fill="#138808" />

            {/* Ashoka Chakra in Center of White Band */}
            <g id="ashoka-chakra" transform="translate(450, 300)">
              {/* Outer Chakra Rim */}
              <circle r="86" fill="none" stroke="#143454" strokeWidth="5.5" />

              {/* Inner Hub */}
              <circle r="20" fill="#143454" />
              <circle r="8" fill="#FFFFFF" />

              {/* 24 Spoke Radii */}
              {Array.from({ length: 24 }).map((_, i) => (
                <line
                  key={`spoke-${i}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="-86"
                  stroke="#143454"
                  strokeWidth="3.2"
                  strokeLinecap="round"
                  transform={`rotate(${i * 15})`}
                />
              ))}

              {/* 24 Decorative Circular Nodes at the Spoke Intervals */}
              {Array.from({ length: 24 }).map((_, i) => (
                <circle
                  key={`node-${i}`}
                  cx="0"
                  cy="-80"
                  r="2.6"
                  fill="#143454"
                  transform={`rotate(${i * 15 + 7.5})`}
                />
              ))}
            </g>

            {/* Sheen Overlay */}
            <rect x="0" y="0" width="900" height="600" fill="url(#flag-sheen)" />
          </svg>
        </div>
      </div>
    </div>
  );
}
