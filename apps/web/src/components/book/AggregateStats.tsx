import { Heart, Star } from "lucide-react";
import type { BookStats } from "@/lib/types";

export type AggregateStatsProps = {
  stats: BookStats;
};

export function AggregateStats({ stats }: AggregateStatsProps) {
  return (
    <div className="flex items-center gap-4 text-xs text-muted-foreground">
      <span className="flex items-center gap-1">
        <Star className="size-3.5 text-[#FFCC00] fill-[#FFCC00]" />
        <span className="font-medium text-foreground">{stats.averageRating}</span>
      </span>
      <span>
        <span className="font-medium text-foreground">{stats.ratingCount}</span> ratings
      </span>
      <span className="flex items-center gap-1">
        <Heart className="size-3.5 text-red-500 fill-red-500" />
        <span className="font-medium text-foreground">{stats.loveCount}</span>
      </span>
    </div>
  );
}
