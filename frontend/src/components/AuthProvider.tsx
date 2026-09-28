"use client";

/**
 * Holds the logged-in user and their JWT, and exposes an API client bound to
 * that token. The token survives page reloads (localStorage) and is validated
 * against /api/auth/me on start-up; any 401 logs the user out.
 */

import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { ApiError, createApi, login as apiLogin, register as apiRegister, type Api } from "@/lib/api";
import { clearToken, loadToken, saveToken } from "@/lib/storage";
import type { TokenResponse, User } from "@/lib/types";

interface AuthContextValue {
  user: User | null;
  token: string | null;
  loading: boolean;
  api: Api;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const logout = useCallback(() => {
    clearToken();
    setToken(null);
    setUser(null);
    router.replace("/login");
  }, [router]);

  const api = useMemo(() => createApi(token, logout), [token, logout]);

  useEffect(() => {
    const saved = loadToken();
    if (!saved) {
      setLoading(false);
      return;
    }
    createApi(saved)
      .me()
      .then((me) => {
        setToken(saved);
        setUser(me);
      })
      .catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
        } else {
          // Server unreachable: keep the token so the user is still logged in once it is back.
          setToken(saved);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const accept = useCallback((response: TokenResponse) => {
    saveToken(response.access_token);
    setToken(response.access_token);
    setUser(response.user);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => accept(await apiLogin(email, password)),
    [accept],
  );

  const register = useCallback(
    async (email: string, password: string, fullName?: string) => accept(await apiRegister(email, password, fullName)),
    [accept],
  );

  const value = useMemo(
    () => ({ user, token, loading, api, login, register, logout }),
    [user, token, loading, api, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}
