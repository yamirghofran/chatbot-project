import { ChevronDown, ChevronRight, X } from "lucide-react";
import { useState } from "react";
import type { UserPreferences } from "@/lib/types";
import { cn } from "@/lib/utils";

export type PreferencesPanelProps = {
  preferences: UserPreferences;
  onClear: () => void;
};

function PreferenceChip({ children, variant = "positive" }: { children: React.ReactNode; variant?: "positive" | "negative" | "neutral" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full",
        variant === "positive" && "bg-green-500/10 text-green-700 dark:text-green-400",
        variant === "negative" && "bg-red-500/10 text-red-700 dark:text-red-400",
        variant === "neutral" && "bg-accent text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

function hasPreferences(prefs: UserPreferences): boolean {
  return !!(
    prefs.liked_genres?.length ||
    prefs.disliked_genres?.length ||
    prefs.max_pages ||
    prefs.standalone_only ||
    prefs.preferred_mood ||
    prefs.liked_books?.length ||
    prefs.disliked_books?.length ||
    prefs.other_constraints?.length
  );
}

export function PreferencesPanel({ preferences, onClear }: PreferencesPanelProps) {
  const [open, setOpen] = useState(false);

  if (!hasPreferences(preferences)) return null;

  return (
    <div className="border-t border-border px-3 py-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors w-full"
      >
        {open ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        <span className="font-medium">Your reading preferences</span>
      </button>

      {open && (
        <div className="mt-2 space-y-2 text-xs">
          <div className="flex flex-wrap gap-1.5">
            {preferences.liked_genres?.map((g) => (
              <PreferenceChip key={g} variant="positive">✓ {g}</PreferenceChip>
            ))}
            {preferences.disliked_genres?.map((g) => (
              <PreferenceChip key={g} variant="negative">✗ {g}</PreferenceChip>
            ))}
            {preferences.preferred_mood && (
              <PreferenceChip variant="neutral">🌙 {preferences.preferred_mood}</PreferenceChip>
            )}
            {preferences.max_pages && (
              <PreferenceChip variant="neutral">📏 Under {preferences.max_pages} pages</PreferenceChip>
            )}
            {preferences.standalone_only && (
              <PreferenceChip variant="neutral">📖 Standalone only</PreferenceChip>
            )}
          </div>

          {preferences.liked_books?.length ? (
            <div className="flex flex-wrap gap-1.5">
              {preferences.liked_books.map((b) => (
                <PreferenceChip key={b} variant="positive">👍 {b}</PreferenceChip>
              ))}
            </div>
          ) : null}

          {preferences.disliked_books?.length ? (
            <div className="flex flex-wrap gap-1.5">
              {preferences.disliked_books.map((b) => (
                <PreferenceChip key={b} variant="negative">👎 {b}</PreferenceChip>
              ))}
            </div>
          ) : null}

          {preferences.other_constraints?.length ? (
            <div className="flex flex-wrap gap-1.5">
              {preferences.other_constraints.map((c) => (
                <PreferenceChip key={c} variant="neutral">{c}</PreferenceChip>
              ))}
            </div>
          ) : null}

          <button
            type="button"
            onClick={onClear}
            className="inline-flex items-center gap-1 text-muted-foreground hover:text-destructive transition-colors mt-1"
          >
            <X className="size-3" />
            Clear preferences
          </button>
        </div>
      )}
    </div>
  );
}
