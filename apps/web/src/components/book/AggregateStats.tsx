import { Star } from "lucide-react";
import type { BookStats } from "@/lib/types";
import { TurtleShellIcon } from "@/components/icons/TurtleShellIcon";

export type AggregateStatsProps = {
  stats: BookStats;
};

export function AggregateStats({ stats }: AggregateStatsProps) {
  return (
    <div className="flex items-center gap-4 text-xs text-muted-foreground">
      <span className="flex items-center gap-1">
        <Star className="size-3.5 text-[#FFCC00] fill-[#FFCC00]" />
        <span className="font-medium text-foreground">{stats.averageRating}</span>
        <span>({stats.ratingCount})</span>
      </span>
      <span className="flex items-center gap-1">
        <TurtleShellIcon className="size-3.5 text-primary" />
        <span className="font-medium text-foreground">{stats.shellCount}</span>
      </span>
    </div>
  );
}
