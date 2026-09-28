"use client";

import { LogOut, Menu, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

import { useAuth } from "@/components/AuthProvider";
import { HistorySidebar } from "@/components/HistorySidebar";
import { Logo } from "@/components/Logo";
import { ThemeToggle } from "@/components/ThemeProvider";

/** Layout for logged-in pages: history sidebar on the left, top bar, scrollable content. */
export function AppShell({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (!menuOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [menuOpen]);

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60 md:flex">
        <div className="flex h-14 items-center border-b border-slate-200 px-4 dark:border-slate-800">
          <Link href="/dashboard">
            <Logo />
          </Link>
        </div>
        <HistorySidebar />
      </aside>

      {menuOpen && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="History">
          <button className="absolute inset-0 bg-slate-900/50" aria-label="Close menu" onClick={() => setMenuOpen(false)} />
          <aside className="absolute inset-y-0 left-0 flex w-80 max-w-[85%] flex-col bg-white shadow-xl dark:bg-slate-900">
            <div className="flex h-14 items-center justify-between border-b border-slate-200 px-4 dark:border-slate-800">
              <Logo />
              <button className="rounded p-1 text-slate-500" onClick={() => setMenuOpen(false)} aria-label="Close menu">
                <X className="h-5 w-5" aria-hidden />
              </button>
            </div>
            <HistorySidebar onNavigate={() => setMenuOpen(false)} />
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white/80 px-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
          <div className="flex items-center gap-2">
            <button
              className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 md:hidden"
              onClick={() => setMenuOpen(true)}
              aria-label="Open history"
            >
              <Menu className="h-5 w-5" aria-hidden />
            </button>
            <Link href="/dashboard" className="md:hidden">
              <Logo />
            </Link>
          </div>
          <div className="flex items-center gap-1">
            <span className="mr-2 hidden max-w-[16rem] truncate text-sm text-slate-600 dark:text-slate-300 sm:inline">
              {user?.full_name || user?.email}
            </span>
            <ThemeToggle />
            <button
              onClick={logout}
              className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200"
              aria-label="Log out"
              title="Log out"
            >
              <LogOut className="h-5 w-5" aria-hidden />
            </button>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
