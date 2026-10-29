"use client";

import { createContext, useContext, useEffect, useMemo, useSyncExternalStore } from "react";

type Theme = "light" | "dark";
type ThemeContextValue = { theme: Theme; toggleTheme: () => void };
const ThemeContext = createContext<ThemeContextValue | null>(null);

const STORAGE_KEY = "policylens-theme";
let currentTheme: Theme = "light";
let initialized = false;
const listeners = new Set<() => void>();

function getThemeSnapshot(): Theme {
  if (typeof window === "undefined") return "light";
  if (!initialized) {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    const preference = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    currentTheme = saved === "dark" || saved === "light" ? saved : preference;
    initialized = true;
  }
  return currentTheme;
}

function getServerSnapshot(): Theme {
  return "light";
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key !== STORAGE_KEY) return;
    const next = event.newValue === "dark" ? "dark" : "light";
    currentTheme = next;
    initialized = true;
    listeners.forEach((callback) => callback());
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

function updateTheme(next: Theme) {
  currentTheme = next;
  initialized = true;
  document.documentElement.dataset.theme = next;
  window.localStorage.setItem(STORAGE_KEY, next);
  listeners.forEach((listener) => listener());
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useSyncExternalStore(subscribe, getThemeSnapshot, getServerSnapshot);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const value = useMemo<ThemeContextValue>(() => ({
    theme,
    toggleTheme: () => updateTheme(theme === "light" ? "dark" : "light")
  }), [theme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used within ThemeProvider");
  return context;
}
