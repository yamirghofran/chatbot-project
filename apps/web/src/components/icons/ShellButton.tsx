import { cn } from "@/lib/utils";
import { TurtleShellIcon } from "./TurtleShellIcon";

export type ShellButtonProps = {
  isShelled?: boolean;
  onClick?: () => void;
  className?: string;
};

export function ShellButton({
  isShelled = false,
  onClick,
  className,
}: ShellButtonProps) {
  return (
    <button
      type="button"
      className={cn(
        "size-9 inline-flex items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground",
        isShelled && "bg-secondary text-primary hover:text-primary",
        className,
      )}
      onClick={onClick}
      aria-label={isShelled ? "Remove from shell" : "Add to shell"}
    >
      <TurtleShellIcon
        filled={isShelled}
        className={cn("size-6", isShelled && "text-primary")}
      />
    </button>
  );
}
