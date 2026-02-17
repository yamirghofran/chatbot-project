import { Search } from "lucide-react";
import { cn } from "@/lib/utils";

export type SearchBarProps = {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  showIcon?: boolean;
  onKeyDown?: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  inputRef?: React.Ref<HTMLInputElement>;
  className?: string;
  inputClassName?: string;
};

export function SearchBar({
  value,
  onChange,
  placeholder = "Search books, authors, lists...",
  showIcon = true,
  onKeyDown,
  inputRef,
  className,
  inputClassName,
}: SearchBarProps) {
  return (
    <div className={cn("relative", className)}>
      {showIcon && (
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
      )}
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        className={cn(
          "w-full rounded-[12px] supports-[corner-shape:squircle]:rounded-[70px] supports-[corner-shape:squircle]:[corner-shape:squircle] border border-input bg-background py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          showIcon ? "px-9" : "px-3",
          inputClassName,
        )}
      />
    </div>
  );
}
