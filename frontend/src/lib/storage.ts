/**
 * Browser storage for the login token and theme.
 *
 * The JWT is kept in localStorage so the SPA can send it as a Bearer header.
 * (An httpOnly cookie is safer against XSS but needs same-site cookie setup
 * between the frontend and backend domains; see docs/architecture.md.)
 */

const TOKEN_KEY = "researchpilot.token";
export const THEME_KEY = "researchpilot.theme";

function safe<T>(action: () => T, fallback: T): T {
  try {
    return action();
  } catch {
    return fallback;
  }
}

export function loadToken(): string | null {
  return safe(() => window.localStorage.getItem(TOKEN_KEY), null);
}

export function saveToken(token: string): void {
  safe(() => window.localStorage.setItem(TOKEN_KEY, token), undefined);
}

export function clearToken(): void {
  safe(() => window.localStorage.removeItem(TOKEN_KEY), undefined);
}

export function loadTheme(): "light" | "dark" | null {
  const value = safe(() => window.localStorage.getItem(THEME_KEY), null);
  return value === "light" || value === "dark" ? value : null;
}

export function saveTheme(theme: "light" | "dark"): void {
  safe(() => window.localStorage.setItem(THEME_KEY, theme), undefined);
}
