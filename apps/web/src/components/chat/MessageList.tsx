import { useEffect, useRef } from "react";
import type { Book, ChatMessage, ComparisonResult } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

export type MessageListProps = {
  messages: ChatMessage[];
  streamingMessageId?: string;
  streamingText?: string;
  streamingBooks?: Book[];
  streamingComparison?: ComparisonResult;
  streamingSource?: string;
  streamingToolName?: string;
};

export function MessageList({
  messages,
  streamingMessageId,
  streamingText,
  streamingBooks,
  streamingComparison,
  streamingSource,
  streamingToolName,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, streamingText]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        <div className="text-center space-y-2">
          <p className="text-lg font-medium text-foreground">BookDB Assistant</p>
          <p>Ask me about books — I can search, recommend, and compare.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
      {messages.map((msg) => {
        const isStreaming = msg.id === streamingMessageId;
        return (
          <MessageBubble
            key={msg.id}
            message={msg}
            isStreaming={isStreaming}
            streamingText={isStreaming ? streamingText : undefined}
            books={isStreaming ? streamingBooks : undefined}
            comparison={isStreaming ? streamingComparison : undefined}
            source={isStreaming ? streamingSource : msg.toolTrace?.source}
            activeToolName={isStreaming ? streamingToolName : undefined}
          />
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}
