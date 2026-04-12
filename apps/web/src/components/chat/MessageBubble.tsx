import { Sparkles, User } from "lucide-react";
import { useMemo } from "react";
import type { Book, ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";
import { InlineBookCard } from "./InlineBookCard";
import { ToolTrace } from "./ToolTrace";

// Lightweight markdown: bold, italic, inline code
function renderInlineMarkdown(text: string): string {
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  html = html.replace(/`([^`]+?)`/g, "<code>$1</code>");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  return html;
}

function isUnorderedList(block: string): boolean {
  return block.split(/\n/).every((l) => !l.trim() || /^\s*[-*]\s+/.test(l));
}

function isOrderedList(block: string): boolean {
  return block.split(/\n/).every((l) => !l.trim() || /^\s*\d+[.)]\s+/.test(l));
}

function markdownToHtml(raw: string): string {
  // Normalise HTML line-break tags the LLM sometimes emits
  const text = raw.replace(/<br\s*\/?>/gi, "\n");

  const paragraphs = text.split(/\n{2,}/);
  return paragraphs
    .map((p) => {
      const trimmed = p.trim();
      if (!trimmed) return "";

      if (isUnorderedList(trimmed)) {
        const items = trimmed
          .split(/\n/)
          .filter((l) => l.trim())
          .map((l) => `<li>${renderInlineMarkdown(l.replace(/^\s*[-*]\s+/, ""))}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
      }

      if (isOrderedList(trimmed)) {
        const items = trimmed
          .split(/\n/)
          .filter((l) => l.trim())
          .map((l) => `<li>${renderInlineMarkdown(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`)
          .join("");
        return `<ol>${items}</ol>`;
      }

      return `<p>${renderInlineMarkdown(trimmed.replace(/\n/g, "<br/>"))}</p>`;
    })
    .join("");
}

export type MessageBubbleProps = {
  message: ChatMessage;
  books?: Book[];
  isStreaming?: boolean;
  streamingText?: string;
};

export function MessageBubble({
  message,
  books = [],
  isStreaming = false,
  streamingText,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const displayText = isStreaming && streamingText != null ? streamingText : message.content;
  const html = useMemo(() => markdownToHtml(displayText || ""), [displayText]);

  const displayBooks = books.length > 0 ? books : message.referencedBooks ?? [];

  return (
    <div className={cn("flex gap-3 max-w-full", isUser ? "flex-row-reverse" : "flex-row")}>
      {/* Avatar */}
      <div
        className={cn(
          "size-7 rounded-full flex items-center justify-center shrink-0 mt-0.5",
          isUser ? "bg-primary/10" : "bg-primary/5",
        )}
      >
        {isUser ? (
          <User className="size-3.5 text-primary" />
        ) : (
          <Sparkles className="size-3.5 text-primary" />
        )}
      </div>

      {/* Content */}
      <div
        className={cn(
          "min-w-0 max-w-[85%] space-y-3",
          isUser ? "items-end" : "items-start",
        )}
      >
        {/* Tool trace */}
        {message.toolTrace && (
          <ToolTrace
            toolName={message.toolTrace.tool}
            source={message.toolTrace.source ?? undefined}
          />
        )}

        {/* Text */}
        {displayText && (
          <div
            className={cn(
              "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
              isUser
                ? "bg-primary text-primary-foreground ml-auto"
                : "bg-input/30",
              "[&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold [&_em]:italic",
              "[&_code]:rounded [&_code]:bg-background/50 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs",
              "[&_ul]:list-disc [&_ul]:pl-4 [&_ul]:mb-2 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:mb-2 [&_li]:mb-0.5",
            )}
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )}

        {/* Streaming cursor */}
        {isStreaming && (
          <span className="inline-block w-1.5 h-4 bg-primary/60 rounded-full animate-pulse ml-1" />
        )}

        {/* Book cards */}
        {displayBooks.length > 0 && (
          <div className="flex gap-2 overflow-x-auto pb-1 -mx-1 px-1">
            {displayBooks.map((book) => (
              <InlineBookCard key={book.id} book={book} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
