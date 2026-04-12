import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Book, ChatMessage, ComparisonResult, ToolTrace, UserPreferences } from "@/lib/types";
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
  const [streamingComparison, setStreamingComparison] = useState<ComparisonResult | undefined>();
  const [streamingSource, setStreamingSource] = useState<string | undefined>();
  const [streamingToolName, setStreamingToolName] = useState<string | undefined>();
  const [sessionPreferences, setSessionPreferences] = useState<UserPreferences | undefined>();
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const abortRef = useRef(false);
  const skipNextLoadRef = useRef(false);

  const sessionsQuery = useQuery({
    queryKey: ["chatSessions"],
    queryFn: () => chatApi.listSessions(),
  });

  const sessions = sessionsQuery.data ?? [];

  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      setSessionPreferences(undefined);
      setIsLoadingSession(false);
      return;
    }
    if (skipNextLoadRef.current) {
      skipNextLoadRef.current = false;
      return;
    }
    setIsLoadingSession(true);
    chatApi.getSession(activeSessionId).then((detail) => {
      setMessages(detail.messages);
      setSessionPreferences(detail.preferences ?? undefined);
    }).catch(() => {
      setMessages([]);
      setSessionPreferences(undefined);
    }).finally(() => {
      setIsLoadingSession(false);
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

      if (!sessionId) {
        try {
          const session = await chatApi.createSession(content.slice(0, 100));
          sessionId = session.id;
          skipNextLoadRef.current = true;
          setActiveSessionId(sessionId);
          queryClient.invalidateQueries({ queryKey: ["chatSessions"] });
        } catch {
          return;
        }
      }

      const userMsg: ChatMessage = {
        id: `temp-user-${Date.now()}`,
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

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
      setStreamingComparison(undefined);
      setStreamingSource(undefined);
      setStreamingToolName(undefined);
      setStreamingMsgId(assistantId);
      abortRef.current = false;

      let fullText = "";
      let allBooks: Book[] = [];
      let lastComparison: ComparisonResult | undefined;
      let lastSource: string | undefined;
      let lastToolName: string | undefined;
      let lastToolInput: unknown;

      await chatApi.sendMessage(sessionId, content, {
        onToken: (text) => {
          if (abortRef.current) return;
          fullText += text;
          setStreamingText(fullText);
        },
        onToolCall: (tool, input) => {
          if (abortRef.current) return;
          lastToolName = tool;
          lastToolInput = input;
          setStreamingToolName(tool);
        },
        onToolResult: (_tool, books, source) => {
          if (abortRef.current) return;
          allBooks = [...allBooks, ...books];
          setStreamingBooks(allBooks);
          lastSource = source;
          setStreamingSource(source);
          setStreamingToolName(undefined);
        },
        onComparison: (comparison) => {
          if (abortRef.current) return;
          lastComparison = comparison;
          setStreamingComparison(comparison);
        },
        onBookCards: (books) => {
          if (abortRef.current) return;
          if (books.length > 0) {
            allBooks = books;
            setStreamingBooks(allBooks);
          }
        },
        onDone: (messageId, referencedBookIds, modelUsed) => {
          const toolTrace: ToolTrace | undefined = lastToolName
            ? { tool: lastToolName, input: lastToolInput, output: {}, source: lastSource }
            : undefined;

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
                    toolName: lastToolName,
                    toolTrace,
                    comparison: lastComparison,
                  }
                : m,
            ),
          );
          setIsStreaming(false);
          setStreamingMsgId(undefined);
          setStreamingComparison(undefined);
          setStreamingSource(undefined);
          setStreamingToolName(undefined);
          queryClient.invalidateQueries({ queryKey: ["chatSessions"] });
          // Re-fetch session to get updated preferences
          if (sessionId) {
            chatApi.getSession(sessionId).then((detail) => {
              setSessionPreferences(detail.preferences ?? undefined);
            }).catch(() => {});
          }
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
          setStreamingToolName(undefined);
        },
      });
    },
    [activeSessionId, queryClient],
  );

  const handleStop = useCallback(() => {
    abortRef.current = true;
    setIsStreaming(false);
    setStreamingMsgId(undefined);
    setStreamingToolName(undefined);
  }, []);

  const handleClearPreferences = useCallback(async () => {
    if (!activeSessionId) return;
    try {
      await chatApi.updatePreferences(activeSessionId, null);
      setSessionPreferences(undefined);
    } catch {
      // ignore
    }
  }, [activeSessionId]);

  return (
    <div className="flex h-full">
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSessionId ?? undefined}
        onSelectSession={setActiveSessionId}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        preferences={sessionPreferences}
        onClearPreferences={handleClearPreferences}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <MessageList
          messages={messages}
          isLoadingSession={isLoadingSession}
          streamingMessageId={streamingMsgId}
          streamingText={streamingText}
          streamingBooks={streamingBooks}
          streamingComparison={streamingComparison}
          streamingSource={streamingSource}
          streamingToolName={streamingToolName}
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
