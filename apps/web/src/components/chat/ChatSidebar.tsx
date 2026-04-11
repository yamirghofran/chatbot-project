import { MessageSquarePlus, Trash2 } from "lucide-react";
import type { ChatSession } from "@/lib/types";
import { cn } from "@/lib/utils";

export type ChatSidebarProps = {
  sessions: ChatSession[];
  activeSessionId?: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
};

export function ChatSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}: ChatSidebarProps) {
  return (
    <aside className="w-64 shrink-0 border-r border-border flex flex-col bg-background">
      <div className="p-3">
        <button
          type="button"
          onClick={onNewSession}
          className="w-full flex items-center gap-2 rounded-lg border border-input px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:border-ring/40 transition-colors"
        >
          <MessageSquarePlus className="size-4" />
          New chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {sessions.map((session) => (
          <div
            key={session.id}
            className={cn(
              "group flex items-center gap-1 rounded-lg px-3 py-2 text-sm cursor-pointer transition-colors",
              session.id === activeSessionId
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
            )}
          >
            <button
              type="button"
              onClick={() => onSelectSession(session.id)}
              className="flex-1 text-left truncate"
            >
              {session.title}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(session.id);
              }}
              className="opacity-0 group-hover:opacity-100 shrink-0 p-0.5 rounded hover:bg-destructive/10 hover:text-destructive transition-all"
              aria-label="Delete session"
            >
              <Trash2 className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
