"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";

import { useAuth } from "@/components/AuthProvider";
import { Logo } from "@/components/Logo";
import { ThemeToggle } from "@/components/ThemeProvider";
import { Alert, Button } from "@/components/ui";

const INPUT =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100";

/** Login and registration form (the same card, two modes). */
export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const { login, register, token, loading } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const isRegister = mode === "register";

  useEffect(() => {
    if (!loading && token) {
      router.replace("/dashboard");
    }
  }, [loading, token, router]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (isRegister && (password.length < 8 || !/[A-Za-z]/.test(password) || !/\d/.test(password))) {
      setError("Use at least 8 characters with at least one letter and one digit.");
      return;
    }
    setSubmitting(true);
    try {
      if (isRegister) {
        await register(email, password, fullName);
      } else {
        await login(email, password);
      }
      router.replace("/dashboard");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <div className="absolute right-4 top-4">
        <ThemeToggle />
      </div>
      <div className="mb-8 flex flex-col items-center gap-3 text-center">
        <Logo />
        <p className="max-w-sm text-sm text-slate-500 dark:text-slate-400">
          A team of AI agents that searches real papers, writes a cited literature review and fact-checks every claim.
        </p>
      </div>
      <div className="w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <h1 className="mb-5 text-xl font-semibold text-slate-900 dark:text-white">
          {isRegister ? "Create your account" : "Welcome back"}
        </h1>
        <form className="space-y-4" onSubmit={(event) => void handleSubmit(event)}>
          {isRegister && (
            <label className="block space-y-1.5">
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Name (optional)</span>
              <input className={INPUT} value={fullName} onChange={(e) => setFullName(e.target.value)} autoComplete="name" />
            </label>
          )}
          <label className="block space-y-1.5">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Email</span>
            <input
              className={INPUT}
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              placeholder="you@university.edu"
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Password</span>
            <input
              className={INPUT}
              type="password"
              required
              minLength={isRegister ? 8 : 1}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={isRegister ? "new-password" : "current-password"}
            />
            {isRegister && (
              <span className="block text-xs text-slate-500 dark:text-slate-400">
                At least 8 characters, including a letter and a digit.
              </span>
            )}
          </label>
          {error && <Alert tone="error">{error}</Alert>}
          <Button type="submit" className="w-full" loading={submitting}>
            {isRegister ? "Create account" : "Log in"}
          </Button>
        </form>
        <p className="mt-5 text-center text-sm text-slate-500 dark:text-slate-400">
          {isRegister ? "Already have an account? " : "New to ResearchPilot? "}
          <Link href={isRegister ? "/login" : "/register"} className="font-medium text-brand-600 hover:underline dark:text-brand-400">
            {isRegister ? "Log in" : "Create an account"}
          </Link>
        </p>
      </div>
    </div>
  );
}
