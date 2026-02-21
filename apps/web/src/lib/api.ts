import type {
  Book,
  User,
  List,
  ActivityItem,
  RatedBook,
  BookStats,
  Review,
} from "./types";

const BASE =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://localhost:8001";

function getToken(): string | null {
  return localStorage.getItem("bookdb_token");
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const method = (init.method ?? "GET").toUpperCase();
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string> | undefined),
  };
  // Avoid CORS preflight on simple GET/HEAD requests.
  if (method !== "GET" && method !== "HEAD" && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
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

export async function login(
  email: string,
  password: string,
): Promise<TokenResponse> {
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

type UserApiData = {
  id: string;
  handle: string;
  displayName: string;
  avatarUrl?: string | null;
};
function mapUser(data: UserApiData): User {
  return {
    id: data.id,
    handle: data.handle,
    displayName: data.displayName,
    avatarUrl: data.avatarUrl ?? undefined,
  };
}

export async function getMe(): Promise<User> {
  return mapUser(await apiFetch<UserApiData>("/auth/me"));
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

export async function searchBooks(q: string, limit = 10): Promise<Book[]> {
  return apiFetch<Book[]>(
    `/books/search?q=${encodeURIComponent(q)}&limit=${limit}`,
  );
}

export interface BookDetail extends Book {
  stats?: BookStats;
  publicationYear?: number | null;
  isbn13?: string | null;
}

export async function getBook(id: string | number): Promise<BookDetail> {
  return apiFetch<BookDetail>(`/books/${id}`);
}

export interface ReviewsPage {
  items: Review[];
  total: number;
}

export async function getBookReviews(
  id: string | number,
  limit = 20,
  offset = 0,
): Promise<ReviewsPage> {
  return apiFetch<ReviewsPage>(
    `/books/${id}/reviews?limit=${limit}&offset=${offset}`,
  );
}

export async function postReview(
  bookId: string | number,
  text: string,
): Promise<Review> {
  return apiFetch<Review>(`/books/${bookId}/reviews`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export async function deleteReview(reviewId: string | number): Promise<void> {
  return apiFetch<void>(`/reviews/${reviewId}`, { method: "DELETE" });
}

export async function likeReview(reviewId: string | number): Promise<void> {
  return apiFetch<void>(`/reviews/${reviewId}/likes`, { method: "POST" });
}

export async function unlikeReview(reviewId: string | number): Promise<void> {
  return apiFetch<void>(`/reviews/${reviewId}/likes`, { method: "DELETE" });
}

export async function postReviewComment(
  reviewId: string | number,
  text: string,
): Promise<Review["replies"][number]> {
  return apiFetch(`/reviews/${reviewId}/comments`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export async function deleteReviewComment(
  reviewId: string | number,
  commentId: string | number,
): Promise<void> {
  return apiFetch<void>(`/reviews/${reviewId}/comments/${commentId}`, {
    method: "DELETE",
  });
}

export async function getRelatedBooks(
  id: string | number,
  limit = 6,
): Promise<Book[]> {
  return apiFetch<Book[]>(`/books/${id}/related?limit=${limit}`);
}

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------

export async function getUser(id: string | number): Promise<User> {
  return mapUser(await apiFetch<UserApiData>(`/user/${id}`));
}

export async function getUserRatings(
  id: string | number,
  limit = 50,
  sort: "recent" | "rating" = "recent",
): Promise<RatedBook[]> {
  return apiFetch<RatedBook[]>(
    `/user/${id}/ratings?limit=${limit}&sort=${sort}`,
  );
}

export async function getUserLists(id: string | number): Promise<List[]> {
  return apiFetch<List[]>(`/user/${id}/lists`);
}

export async function getUserFavorites(
  id: string | number,
  limit = 3,
): Promise<Book[]> {
  return apiFetch<Book[]>(`/user/${id}/favorites?limit=${limit}`);
}

export async function getUserActivity(
  id: string | number,
  limit = 10,
): Promise<ActivityItem[]> {
  return apiFetch<ActivityItem[]>(`/user/${id}/activity?limit=${limit}`);
}

// ---------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------

export async function getList(id: string | number): Promise<List> {
  return apiFetch<List>(`/lists/${id}`);
}

export async function createList(
  name: string,
  description?: string,
): Promise<{ id: string; name: string }> {
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

export async function addBookToList(
  listId: string | number,
  bookId: string | number,
): Promise<void> {
  return apiFetch<void>(`/lists/${listId}/books/${bookId}`, { method: "POST" });
}

export async function removeBookFromList(
  listId: string | number,
  bookId: string | number,
): Promise<void> {
  return apiFetch<void>(`/lists/${listId}/books/${bookId}`, {
    method: "DELETE",
  });
}

export async function reorderList(
  listId: string | number,
  bookIds: (string | number)[],
): Promise<void> {
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

export async function getMyFavorites(limit = 3): Promise<Book[]> {
  return apiFetch<Book[]>(`/me/favorites?limit=${limit}`);
}

export async function addToMyFavorites(bookId: string | number): Promise<void> {
  return apiFetch<void>(`/me/favorites/${bookId}`, { method: "POST" });
}

export async function removeFromMyFavorites(
  bookId: string | number,
): Promise<void> {
  return apiFetch<void>(`/me/favorites/${bookId}`, { method: "DELETE" });
}

export async function getMyRating(
  bookId: string | number,
): Promise<{ rating: number | null }> {
  return apiFetch<{ rating: number | null }>(`/me/ratings/${bookId}`);
}

export async function upsertRating(
  bookId: string | number,
  rating: number,
): Promise<void> {
  return apiFetch<void>(`/me/ratings?book_id=${bookId}&rating=${rating}`, {
    method: "POST",
  });
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
