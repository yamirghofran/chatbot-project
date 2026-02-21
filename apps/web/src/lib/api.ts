import type { Book, User, List, ActivityItem, RatedBook, BookStats, Review } from "./types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8001";

function getToken(): string | null {
  return localStorage.getItem("bookdb_token");
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string> | undefined),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    let message = text;
    try {
      const json = JSON.parse(text);
      if (json.detail) message = String(json.detail);
    } catch {
      // use raw text
    }
    throw new Error(message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export function setToken(token: string): void {
  localStorage.setItem("bookdb_token", token);
}

export function clearToken(): void {
  localStorage.removeItem("bookdb_token");
}

export function getStoredToken(): string | null {
  return getToken();
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function register(
  email: string,
  password: string,
  name: string,
  username: string,
): Promise<TokenResponse> {
  return apiFetch<TokenResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, name, username }),
  });
}

export async function getMe(): Promise<User> {
  const data = await apiFetch<{
    id: string;
    handle: string;
    displayName: string;
    avatarUrl?: string | null;
  }>("/auth/me");
  return {
    id: data.id,
    handle: data.handle,
    displayName: data.displayName,
    avatarUrl: data.avatarUrl ?? undefined,
  };
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

export async function getRecommendations(limit = 20): Promise<Book[]> {
  return apiFetch<Book[]>(`/discovery/recommendations?limit=${limit}`);
}

export async function getStaffPicks(limit = 6): Promise<Book[]> {
  return apiFetch<Book[]>(`/discovery/staff-picks?limit=${limit}`);
}

export async function getActivityFeed(limit = 10): Promise<ActivityItem[]> {
  return apiFetch<ActivityItem[]>(`/discovery/activity?limit=${limit}`);
}

// ---------------------------------------------------------------------------
// Books
// ---------------------------------------------------------------------------

export async function searchBooks(q: string, limit = 20): Promise<Book[]> {
  return apiFetch<Book[]>(`/books/search?q=${encodeURIComponent(q)}&limit=${limit}`);
}

export interface BookDetail extends Book {
  stats?: BookStats;
  publicationYear?: number | null;
  isbn13?: string | null;
}

export async function getBook(id: string | number): Promise<BookDetail> {
  return apiFetch<BookDetail>(`/books/${id}`);
}

export async function getBookReviews(id: string | number, limit = 20): Promise<Review[]> {
  return apiFetch<Review[]>(`/books/${id}/reviews?limit=${limit}`);
}

export async function getRelatedBooks(id: string | number, limit = 6): Promise<Book[]> {
  return apiFetch<Book[]>(`/books/${id}/related?limit=${limit}`);
}

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------

export async function getUser(id: string | number): Promise<User> {
  const data = await apiFetch<{
    id: string;
    handle: string;
    displayName: string;
    avatarUrl?: string | null;
  }>(`/users/${id}`);
  return {
    id: data.id,
    handle: data.handle,
    displayName: data.displayName,
    avatarUrl: data.avatarUrl ?? undefined,
  };
}

export async function getUserRatings(
  id: string | number,
  limit = 50,
  sort: "recent" | "rating" = "recent",
): Promise<RatedBook[]> {
  return apiFetch<RatedBook[]>(`/users/${id}/ratings?limit=${limit}&sort=${sort}`);
}

export async function getUserLists(id: string | number): Promise<List[]> {
  return apiFetch<List[]>(`/users/${id}/lists`);
}

export async function getUserActivity(id: string | number, limit = 10): Promise<ActivityItem[]> {
  return apiFetch<ActivityItem[]>(`/users/${id}/activity?limit=${limit}`);
}

// ---------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------

export async function getList(id: string | number): Promise<List> {
  return apiFetch<List>(`/lists/${id}`);
}

export async function createList(name: string, description?: string): Promise<{ id: string; name: string }> {
  return apiFetch<{ id: string; name: string }>("/me/lists", {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
}

export async function updateList(
  id: string | number,
  body: { name?: string; description?: string },
): Promise<List> {
  return apiFetch<List>(`/lists/${id}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export async function deleteList(id: string | number): Promise<void> {
  return apiFetch<void>(`/lists/${id}`, { method: "DELETE" });
}

export async function addBookToList(listId: string | number, bookId: string | number): Promise<void> {
  return apiFetch<void>(`/lists/${listId}/books/${bookId}`, { method: "POST" });
}

export async function removeBookFromList(listId: string | number, bookId: string | number): Promise<void> {
  return apiFetch<void>(`/lists/${listId}/books/${bookId}`, { method: "DELETE" });
}

export async function reorderList(listId: string | number, bookIds: (string | number)[]): Promise<void> {
  return apiFetch<void>(`/lists/${listId}/reorder`, {
    method: "PUT",
    body: JSON.stringify(bookIds.map(Number)),
  });
}

// ---------------------------------------------------------------------------
// Me
// ---------------------------------------------------------------------------

export async function getMyLists(): Promise<List[]> {
  return apiFetch<List[]>("/me/lists");
}

export async function getMyShell(): Promise<Book[]> {
  return apiFetch<Book[]>("/me/shell");
}

export async function upsertRating(bookId: string | number, rating: number): Promise<void> {
  return apiFetch<void>(`/me/ratings?book_id=${bookId}&rating=${rating}`, { method: "POST" });
}

export async function deleteRating(bookId: string | number): Promise<void> {
  return apiFetch<void>(`/me/ratings/${bookId}`, { method: "DELETE" });
}

export async function addToShell(bookId: string | number): Promise<void> {
  return apiFetch<void>(`/me/shell/${bookId}`, { method: "POST" });
}

export async function removeFromShell(bookId: string | number): Promise<void> {
  return apiFetch<void>(`/me/shell/${bookId}`, { method: "DELETE" });
}
