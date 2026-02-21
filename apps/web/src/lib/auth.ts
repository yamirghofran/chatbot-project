import { useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "./api";
import type { User } from "./types";

export function getToken(): string | null {
  return localStorage.getItem("bookdb_token");
}

export function setToken(token: string): void {
  localStorage.setItem("bookdb_token", token);
}

export function clearToken(): void {
  localStorage.removeItem("bookdb_token");
}

export function isAuthenticated(): boolean {
  return getToken() !== null;
}

export function useCurrentUser() {
  return useQuery<User | null>({
    queryKey: ["me"],
    queryFn: async () => {
      if (!getToken()) return null;
      try {
        return await api.getMe();
      } catch {
        clearToken();
        return null;
      }
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
}
