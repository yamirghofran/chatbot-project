import type { Book, ChatMessage, ChatSession, ChatSessionDetail } from "./types";

const BASE =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://localhost:8001";

function getToken(): string | null {
  return localStorage.getItem("bookdb_token");
}

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

async function chatFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = { ...authHeaders(), ...(init.headers as Record<string, string>) };
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
// Session CRUD
// ---------------------------------------------------------------------------

export async function createSession(title?: string): Promise<ChatSession> {
  return chatFetch<ChatSession>("/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ title: title ?? null }),
  });
}

export async function listSessions(limit = 20): Promise<ChatSession[]> {
  return chatFetch<ChatSession[]>(`/chat/sessions?limit=${limit}`);
}

export async function getSession(id: string): Promise<ChatSessionDetail> {
  return chatFetch<ChatSessionDetail>(`/chat/sessions/${id}`);
}

export async function deleteSession(id: string): Promise<void> {
  return chatFetch<void>(`/chat/sessions/${id}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// SSE message stream
// ---------------------------------------------------------------------------

export interface SendMessageCallbacks {
  onToken: (text: string) => void;
  onToolCall: (tool: string, input: unknown) => void;
  onToolResult: (tool: string, books: Book[], source: string) => void;
  onBookCards: (books: Book[]) => void;
  onDone: (messageId: string, referencedBookIds: number[], modelUsed?: string) => void;
  onError: (error: Error) => void;
}

export async function sendMessage(
  sessionId: string,
  content: string,
  callbacks: SendMessageCallbacks,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/chat/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ content }),
    });
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    callbacks.onError(new Error(text));
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError(new Error("No response body"));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      let currentEvent = "";
      let dataLines: string[] = [];

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
          dataLines = [];
        } else if (line.startsWith("data: ")) {
          dataLines.push(line.slice(6));
        } else if (line === "" && currentEvent && dataLines.length > 0) {
          const raw = dataLines.join("\n");
          try {
            const data = JSON.parse(raw);
            dispatchEvent(currentEvent, data, callbacks);
          } catch {
            // skip malformed events
          }
          currentEvent = "";
          dataLines = [];
        }
      }
    }
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
  }
}

function dispatchEvent(
  event: string,
  data: Record<string, unknown>,
  cb: SendMessageCallbacks,
) {
  switch (event) {
    case "token":
      cb.onToken(String(data.text ?? ""));
      break;
    case "tool_call":
      cb.onToolCall(String(data.tool ?? ""), data.input);
      break;
    case "tool_result":
      cb.onToolResult(
        String(data.tool ?? ""),
        (data.books ?? []) as Book[],
        String(data.source ?? ""),
      );
      break;
    case "book_cards":
      cb.onBookCards((data.books ?? []) as Book[]);
      break;
    case "done":
      cb.onDone(
        String(data.message_id ?? ""),
        (data.referenced_book_ids ?? []) as number[],
        data.model_used ? String(data.model_used) : undefined,
      );
      break;
    default:
      break;
  }
}
