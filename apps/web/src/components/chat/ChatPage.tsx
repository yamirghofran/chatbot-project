import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Book, ChatMessage, ChatSession } from "@/lib/types";
import * as chatApi from "@/lib/chat-api";
import { ChatInput } from "./ChatInput";
import { ChatSidebar } from "./ChatSidebar";
import { MessageList } from "./MessageList";

export function ChatPage() {
  const queryClient = useQueryClient();

  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [streamingBooks, setStreamingBooks] = useState<Book[]>([]);
  const [streamingMsgId, setStreamingMsgId] = useState<string | undefined>();
  const abortRef = useRef(false);

  const sessionsQuery = useQuery({
    queryKey: ["chatSessions"],
    queryFn: () => chatApi.listSessions(),
  });

  const sessions = sessionsQuery.data ?? [];

  // Load session messages when active session changes
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      return;
    }
    chatApi.getSession(activeSessionId).then((detail) => {
      setMessages(detail.messages);
    }).catch(() => {
      setMessages([]);
    });
  }, [activeSessionId]);

  const handleNewSession = useCallback(async () => {
    try {
      const session = await chatApi.createSession();
      setActiveSessionId(session.id);
      setMessages([]);
      queryClient.invalidateQueries({ queryKey: ["chatSessions"] });
    } catch {
      // ignore
    }
  }, [queryClient]);

  const handleDeleteSession = useCallback(
    async (id: string) => {
      try {
        await chatApi.deleteSession(id);
        if (activeSessionId === id) {
          setActiveSessionId(null);
          setMessages([]);
        }
        queryClient.invalidateQueries({ queryKey: ["chatSessions"] });
      } catch {
        // ignore
      }
    },
    [activeSessionId, queryClient],
  );

  const handleSend = useCallback(
    async (content: string) => {
      let sessionId = activeSessionId;

      // Auto-create session if none active
      if (!sessionId) {
        try {
          const session = await chatApi.createSession(content.slice(0, 100));
          sessionId = session.id;
          setActiveSessionId(sessionId);
          queryClient.invalidateQueries({ queryKey: ["chatSessions"] });
        } catch {
          return;
        }
      }

      // Add optimistic user message
      const userMsg: ChatMessage = {
        id: `temp-user-${Date.now()}`,
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      // Prepare streaming assistant message
      const assistantId = `temp-assistant-${Date.now()}`;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setIsStreaming(true);
      setStreamingText("");
      setStreamingBooks([]);
      setStreamingMsgId(assistantId);
      abortRef.current = false;

      let fullText = "";
      let allBooks: Book[] = [];

      await chatApi.sendMessage(sessionId, content, {
        onToken: (text) => {
          if (abortRef.current) return;
          fullText += text;
          setStreamingText(fullText);
        },
        onToolCall: () => {
          // Could show a loading state here
        },
        onToolResult: (_tool, books) => {
          if (abortRef.current) return;
          allBooks = [...allBooks, ...books];
          setStreamingBooks(allBooks);
        },
        onBookCards: (books) => {
          if (abortRef.current) return;
          if (books.length > 0) {
            allBooks = books;
            setStreamingBooks(allBooks);
          }
        },
        onDone: (messageId, referencedBookIds, modelUsed) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    id: messageId || m.id,
                    content: fullText,
                    referencedBooks: allBooks,
                    referencedBookIds,
                    modelUsed,
                  }
                : m,
            ),
          );
          setIsStreaming(false);
          setStreamingMsgId(undefined);
          queryClient.invalidateQueries({ queryKey: ["chatSessions"] });
        },
        onError: (err) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `Error: ${err.message}` }
                : m,
            ),
          );
          setIsStreaming(false);
          setStreamingMsgId(undefined);
        },
      });
    },
    [activeSessionId, queryClient],
  );

  const handleStop = useCallback(() => {
    abortRef.current = true;
    setIsStreaming(false);
    setStreamingMsgId(undefined);
  }, []);

  return (
    <div className="flex h-[calc(100dvh-65px)] -mx-4 -my-8">
      {/* Sidebar */}
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSessionId ?? undefined}
        onSelectSession={setActiveSessionId}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        <MessageList
          messages={messages}
          streamingMessageId={streamingMsgId}
          streamingText={streamingText}
          streamingBooks={streamingBooks}
        />

        <div className="shrink-0 border-t border-border px-4 py-3">
          <ChatInput
            onSend={handleSend}
            onStop={handleStop}
            isStreaming={isStreaming}
            disabled={false}
          />
        </div>
      </div>
    </div>
  );
}
