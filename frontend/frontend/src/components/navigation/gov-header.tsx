"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { House, History, Bookmark, Compass, UserRound, Menu, X, ExternalLink, ShieldCheck } from "lucide-react";
import { PortalIdentity } from "@/components/brand/portal-identity";
import { ThemeToggle } from "@/components/theme/theme-toggle";

const NAV_ITEMS = [
  { label: "Home", href: "/", icon: House },
  { label: "History", href: "/history", icon: History },
  { label: "Saved", href: "/saved", icon: Bookmark },
  { label: "Explore", href: "/discover", icon: Compass },
  { label: "Profile", href: "/profile", icon: UserRound },
];

export function GovHeader() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="w-full bg-white border-b border-[#e5e7eb] sticky top-0 z-40 shadow-xs">
      {/* Top Tricolour Ribbon */}
      <div className="h-[3px] w-full bg-gradient-to-r from-[#E87518] via-white to-[#16834A]" aria-hidden="true" />

      {/* Top Utility Bar (Screen Reader, Accessibility, National Portal Links) */}
      <div className="bg-[#f8f9fa] border-b border-[#e5e7eb] text-[11px] text-[#444444] px-4 py-1">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-medium text-[#171717] flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full bg-[#16834A]" aria-hidden="true" />
              Public Information Portal
            </span>
            <span className="text-[#9ca3af] hidden sm:inline">|</span>
            <span className="text-[#666666] hidden sm:inline">
              Civic Policy &amp; Programme Repository
            </span>
          </div>
          <div className="flex items-center gap-3">
            <a
              href="https://www.india.gov.in/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[#E87518] transition-colors hidden md:inline-flex items-center gap-1"
            >
              National Portal of India <ExternalLink size={10} />
            </a>
            <span className="text-[#9ca3af] hidden md:inline">|</span>
            <a
              href="https://www.pib.gov.in/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[#E87518] transition-colors hidden sm:inline-flex items-center gap-1"
            >
              PIB <ExternalLink size={10} />
            </a>
            <span className="text-[#9ca3af] hidden sm:inline">|</span>
            <ThemeToggle />
          </div>
        </div>
      </div>

      {/* Main White Masthead */}
      <div className="max-w-7xl mx-auto px-4 py-2.5 sm:py-3.5 flex items-center justify-between gap-4">
        {/* Left: Identity Block with Provided Logo */}
        <PortalIdentity />

        {/* Right: Institutional Badges (Swachh Bharat & Digital India) */}
        <div className="hidden lg:flex items-center gap-5">
          {/* Swachh Bharat Clean Line Art Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#e5e7eb] bg-[#f9fafb]">
            <svg width="42" height="20" viewBox="0 0 60 26" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Swachh Bharat Mission">
              <circle cx="16" cy="13" r="10" stroke="#171717" strokeWidth="2.2" />
              <circle cx="44" cy="13" r="10" stroke="#171717" strokeWidth="2.2" />
              <path d="M26 13H34" stroke="#171717" strokeWidth="2.2" />
              <path d="M6 13C3 13 1 10 1 7" stroke="#171717" strokeWidth="1.8" strokeLinecap="round" />
              <path d="M54 13C57 13 59 10 59 7" stroke="#171717" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
            <div className="flex flex-col text-left leading-none">
              <span className="text-[10px] font-bold text-[#171717]">स्वच्छ भारत</span>
              <span className="text-[8px] text-[#666666] font-medium">एक कदम स्वच्छता की ओर</span>
            </div>
          </div>

          {/* Digital India Initiative Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#e5e7eb] bg-[#f9fafb]">
            <div className="w-6 h-6 rounded-full bg-[#143454] flex items-center justify-center text-white">
              <ShieldCheck size={14} className="text-[#E87518]" />
            </div>
            <div className="flex flex-col text-left leading-none">
              <span className="text-[10px] font-bold text-[#143454]">डिजिटल भारत</span>
              <span className="text-[8px] text-[#666666] font-medium">Digital Public Data</span>
            </div>
          </div>
        </div>

        {/* Mobile menu toggle button */}
        <button
          type="button"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 rounded-lg text-[#171717] hover:bg-[#f3f4f6] focus:outline-none focus:ring-2 focus:ring-[#143454]"
          aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={mobileMenuOpen}
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Full-width Dark Navy Blue Navigation Bar */}
      <nav className="w-full bg-[#143454] text-white" aria-label="Primary Portal Navigation">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between">
          {/* Desktop Nav Links with 16–18px font size and comfortable spacing */}
          <ul className="hidden md:flex items-center gap-2 sm:gap-4 md:gap-6 text-[16px] sm:text-[17px] font-semibold tracking-normal">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

              return (
                <li key={item.label}>
                  <Link
                    href={item.href}
                    className={`inline-flex items-center gap-2.5 px-4 py-3.5 transition-colors border-b-4 ${
                      isActive
                        ? "bg-[#0e2a47] text-white border-[#E87518]"
                        : "text-[#f1f5f9] hover:bg-[#1a4168] hover:text-white border-transparent"
                    }`}
                  >
                    <Icon size={18} className={isActive ? "text-[#E87518]" : "text-[#cbd5e1]"} strokeWidth={isActive ? 2.2 : 1.8} />
                    <span>{item.label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {/* Quick External Portals Link on right */}
          <div className="hidden md:flex items-center gap-2 py-2 text-[12px] text-[#cbd5e1]">
            <span className="bg-[#1a4168] px-3 py-1.5 rounded text-white font-medium flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#E87518]" />
              Official Government Portal
            </span>
          </div>

          {/* Mobile bar summary */}
          <div className="flex md:hidden items-center justify-between w-full py-3 text-[15px] font-semibold text-white">
            <span className="flex items-center gap-2">
              <House size={18} className="text-[#E87518]" />
              Portal Navigation
            </span>
            <span className="text-[12px] font-normal text-[#94a3b8]">{mobileMenuOpen ? "Close ▲" : "Menu ▼"}</span>
          </div>
        </div>

        {/* Mobile Dropdown Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-[#1e4770] bg-[#0e2a47] px-4 py-3 space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.label}
                  href={item.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg text-[16px] font-medium transition-colors ${
                    isActive
                      ? "bg-[#143454] text-white border-l-4 border-[#E87518]"
                      : "text-[#cbd5e1] hover:bg-[#1a4168] hover:text-white"
                  }`}
                >
                  <Icon size={20} className={isActive ? "text-[#E87518]" : "text-[#cbd5e1]"} strokeWidth={isActive ? 2.2 : 1.8} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
            <div className="pt-3 mt-2 border-t border-[#1a4168] flex items-center justify-between text-[12px] text-[#94a3b8]">
              <a
                href="https://www.india.gov.in"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white inline-flex items-center gap-1"
              >
                india.gov.in <ExternalLink size={11} />
              </a>
              <a
                href="https://www.pib.gov.in"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white inline-flex items-center gap-1"
              >
                pib.gov.in <ExternalLink size={11} />
              </a>
            </div>
          </div>
        )}
      </nav>
    </header>
  );
}
