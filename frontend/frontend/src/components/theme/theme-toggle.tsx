"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "./theme-provider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const label = theme === "light" ? "Switch to dark mode" : "Switch to light mode";
  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={label}
      title={label}
      className="flex h-10 w-10 items-center justify-center rounded-full border transition-colors"
      style={{ borderColor: "var(--line)", background: "var(--surface)" }}
    >
      {theme === "light" ? <Moon size={17} strokeWidth={1.8} /> : <Sun size={17} strokeWidth={1.8} />}
    </button>
  );
}
