import { Star } from "lucide-react";
import type { RatedBook } from "@/lib/types";

export type RatingsDistributionProps = {
  ratedBooks: RatedBook[];
};

export function RatingsDistribution({ ratedBooks }: RatingsDistributionProps) {
  if (ratedBooks.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4">No ratings yet.</p>
    );
  }

  const counts = [5, 4, 3, 2, 1].map((star) => ({
    star,
    count: ratedBooks.filter((rb) => Math.round(rb.rating) === star).length,
  }));

  const maxCount = Math.max(...counts.map((c) => c.count), 1);
  const total = ratedBooks.length;
  const average = ratedBooks.reduce((sum, rb) => sum + rb.rating, 0) / total;

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <div className="flex items-center gap-1">
          <Star className="size-5 text-[#FFCC00] fill-[#FFCC00]" />
          <span className="text-2xl font-semibold text-foreground">{average.toFixed(1)}</span>
        </div>
        <span className="text-sm text-muted-foreground">{total} ratings</span>
      </div>
      <div className="space-y-1.5">
        {counts.map(({ star, count }) => (
          <div key={star} className="flex items-center gap-2 text-sm">
            <span className="w-12 text-right text-muted-foreground shrink-0">
              {star} star{star !== 1 ? "s" : ""}
            </span>
            <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-foreground rounded-full transition-all"
                style={{ width: `${(count / maxCount) * 100}%` }}
              />
            </div>
            <span className="w-6 text-right text-muted-foreground shrink-0">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
