import { ChevronDown, ChevronRight, Info } from "lucide-react";
import { useState } from "react";
import type { ToolTrace } from "@/lib/types";
export type WhyPanelProps = {
  toolTrace: ToolTrace;
};

function extractQueryFromInput(input: unknown): string | undefined {
  if (input && typeof input === "object" && "query" in input) {
    return String((input as Record<string, unknown>).query);
  }
  return undefined;
}

function extractReviewExcerpts(output: unknown): string[] {
  if (!output || typeof output !== "object") return [];
  const out = output as Record<string, unknown>;
  const data = out.data as Record<string, unknown> | undefined;
  if (!data) return [];
  const reviews = data.reviews as Array<Record<string, unknown>> | undefined;
  if (!Array.isArray(reviews)) return [];
  return reviews
    .slice(0, 3)
    .map((r) => String(r.review ?? r.review_text ?? "").slice(0, 200))
    .filter((t) => t.length > 10);
}

export function WhyPanel({ toolTrace }: WhyPanelProps) {
  const [open, setOpen] = useState(false);

  const query = extractQueryFromInput(toolTrace.input);
  const reviews = extractReviewExcerpts(toolTrace.output);
  const source = toolTrace.source;
  const toolName = toolTrace.tool;

  const hasContent = query || reviews.length > 0 || source;
  if (!hasContent) return null;

  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
      >
        <Info className="size-3" />
        Why this book?
        {open ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
      </button>

      {open && (
        <div className="mt-1.5 rounded-lg border border-border/60 bg-accent/10 px-3 py-2 text-xs space-y-1.5">
          {source && (
            <div>
              <span className="font-medium text-muted-foreground">Found via:</span>{" "}
              <span className="capitalize">{source.replace(/_/g, " ")}</span>
              {toolName && (
                <span className="text-muted-foreground"> ({toolName.replace(/_/g, " ")})</span>
              )}
            </div>
          )}

          {query && (
            <div>
              <span className="font-medium text-muted-foreground">Query:</span>{" "}
              <span className="italic">"{query}"</span>
            </div>
          )}

          {reviews.length > 0 && (
            <div>
              <span className="font-medium text-muted-foreground">What readers say:</span>
              <ul className="mt-1 space-y-1 pl-3">
                {reviews.map((r, i) => (
                  <li key={i} className="text-muted-foreground leading-snug">
                    "{r.length < 200 ? r : r + "…"}"
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
