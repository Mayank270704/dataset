"use client";
import Link from "next/link";
import { Bookmark, Compass, History, House, UserRound } from "lucide-react";
const links = [{ href: "/", label: "Home", icon: House }, { href: "/history", label: "History", icon: History }, { href: "/saved", label: "Saved", icon: Bookmark }, { href: "/discover", label: "Explore", icon: Compass }, { href: "/profile", label: "Profile", icon: UserRound }];
export function SiteNav() { return <nav className="fixed inset-x-0 bottom-0 z-40 grid grid-cols-5 border-t border-slate-200 bg-white px-2 pb-[max(env(safe-area-inset-bottom),8px)] pt-2 md:hidden" aria-label="Portal navigation">{links.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className="flex min-h-11 flex-col items-center justify-center gap-1 text-[11px] text-slate-600"><Icon size={18}/><span>{label}</span></Link>)}</nav>; }
