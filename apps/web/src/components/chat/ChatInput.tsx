import { ArrowUp, Square } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

export type ChatInputProps = {
  onSend: (message: string) => void;
  onStop?: () => void;
  disabled?: boolean;
  isStreaming?: boolean;
  placeholder?: string;
};

export function ChatInput({
  onSend,
  onStop,
  disabled = false,
  isStreaming = false,
  placeholder = "Ask about books...",
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  function handleSubmit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    inputRef.current?.focus();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isStreaming) return;
      handleSubmit();
    }
  }

  return (
    <div className="relative">
      <textarea
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        rows={1}
        disabled={disabled}
        className={cn(
          "w-full resize-none rounded-[16px] border border-input bg-background py-3 pl-4 pr-12 text-sm text-foreground",
          "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          "min-h-[48px] max-h-[200px]",
          disabled && "opacity-50 cursor-not-allowed",
        )}
        style={{ fieldSizing: "content" } as React.CSSProperties}
      />
      <button
        type="button"
        onClick={isStreaming ? onStop : handleSubmit}
        aria-label={isStreaming ? "Stop generating" : "Send message"}
        disabled={!isStreaming && (!value.trim() || disabled)}
        className={cn(
          "absolute right-2.5 bottom-2.5 size-8 inline-flex items-center justify-center rounded-full transition-all",
          isStreaming
            ? "bg-destructive text-white hover:bg-destructive/90"
            : "bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-30 disabled:cursor-not-allowed",
        )}
      >
        {isStreaming ? <Square className="size-3.5" /> : <ArrowUp className="size-4" />}
      </button>
    </div>
  );
}
