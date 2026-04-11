import { ChevronDown, Wrench } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

export type ToolTraceProps = {
  toolName: string;
  source?: string;
  isLoading?: boolean;
};

export function ToolTrace({ toolName, source, isLoading = false }: ToolTraceProps) {
  const [open, setOpen] = useState(false);

  const label = toolName.replace(/_/g, " ");

  return (
    <div className="text-xs text-muted-foreground">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
      >
        <Wrench className="size-3" />
        <span>
          {isLoading ? `Calling ${label}...` : `Used ${label}`}
          {source && !isLoading && (
            <span className="ml-1 text-muted-foreground/60">({source})</span>
          )}
        </span>
        {!isLoading && (
          <ChevronDown
            className={cn("size-3 transition-transform", open && "rotate-180")}
          />
        )}
      </button>
    </div>
  );
}
