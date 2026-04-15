import { ChevronDown, Database, GitCompare, Search, Sparkles, Wrench } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const TOOL_CONFIG: Record<string, { label: string; icon: typeof Wrench }> = {
  search_books: { label: "Searching the catalogue", icon: Search },
  get_book_details: { label: "Fetching book details", icon: Database },
  get_related_books: { label: "Finding similar books", icon: Search },
  get_recommendations: { label: "Getting recommendations", icon: Sparkles },
  compare_books: { label: "Comparing books", icon: GitCompare },
  recommend_via_mcp: { label: "Consulting recommendation engine", icon: Sparkles },
};

function getSourceLabel(source?: string): string | undefined {
  if (!source) return undefined;
  if (source.startsWith("local_fallback")) return "Used local recommendations (external engine unavailable)";
  return source.replace(/_/g, " ");
}

export type ToolTraceProps = {
  toolName: string;
  source?: string;
  isLoading?: boolean;
};

export function ToolTrace({ toolName, source, isLoading = false }: ToolTraceProps) {
  const [open, setOpen] = useState(false);

  const config = TOOL_CONFIG[toolName];
  const label = config?.label ?? toolName.replace(/_/g, " ");
  const Icon = config?.icon ?? Wrench;
  const sourceLabel = getSourceLabel(source);

  return (
    <div className="text-xs text-muted-foreground">
      <button
        type="button"
        onClick={() => !isLoading && setOpen((prev) => !prev)}
        className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
      >
        <Icon className={cn("size-3", isLoading && "animate-spin")} />
        <span>
          {isLoading ? (
            <>
              {label}
              <span className="inline-flex ml-1 gap-0.5">
                <span className="size-1 rounded-full bg-current animate-bounce [animation-delay:0ms]" />
                <span className="size-1 rounded-full bg-current animate-bounce [animation-delay:150ms]" />
                <span className="size-1 rounded-full bg-current animate-bounce [animation-delay:300ms]" />
              </span>
            </>
          ) : (
            `Used ${label.toLowerCase()}`
          )}
        </span>
        {!isLoading && (
          <ChevronDown
            className={cn("size-3 transition-transform", open && "rotate-180")}
          />
        )}
      </button>

      {open && sourceLabel && (
        <div className="mt-1 ml-4.5 text-muted-foreground/70">
          Source: {sourceLabel}
        </div>
      )}
    </div>
  );
}
