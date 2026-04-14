import type { Book, ComparisonResult } from "@/lib/types";
import { cn } from "@/lib/utils";

export type ComparisonTableProps = {
  comparison: ComparisonResult;
  books?: Book[];
};

export function ComparisonTable({ comparison, books = [] }: ComparisonTableProps) {
  const bookCount = comparison.dimensions[0]?.values.length ?? 0;
  if (bookCount === 0) return null;

  const bookHeaders = comparison.bookIds.map((id, i) => {
    const book = books.find((b) => String(b.id) === String(id));
    return {
      id,
      title: book?.title ?? `Book ${i + 1}`,
      author: book?.author ?? "",
      coverUrl: book?.coverUrl,
    };
  });

  return (
    <div className="rounded-xl border border-border overflow-hidden text-sm">
      <table className="w-full">
        <thead>
          <tr className="bg-accent/30">
            <th className="px-3 py-2.5 text-left font-medium text-muted-foreground w-28" />
            {bookHeaders.map((bh) => (
              <th key={bh.id} className="px-3 py-2.5 text-left">
                <div className="flex items-center gap-2">
                  {bh.coverUrl && (
                    <img
                      src={bh.coverUrl}
                      alt=""
                      className="w-8 h-12 rounded object-cover shrink-0"
                    />
                  )}
                  <div className="min-w-0">
                    <div className="font-semibold truncate">{bh.title}</div>
                    {bh.author && (
                      <div className="text-xs text-muted-foreground truncate">{bh.author}</div>
                    )}
                  </div>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {comparison.dimensions.map((dim, i) => (
            <tr
              key={dim.name}
              className={cn(
                "border-t border-border",
                i % 2 === 0 ? "bg-background" : "bg-accent/10",
              )}
            >
              <td className="px-3 py-2 font-medium text-muted-foreground align-top">
                {dim.name}
              </td>
              {dim.values.map((val, vi) => (
                <td key={vi} className="px-3 py-2 align-top">
                  {val}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {comparison.verdict && (
        <div className="border-t border-border bg-accent/20 px-3 py-2.5 text-sm">
          <span className="font-medium">Verdict:</span> {comparison.verdict}
        </div>
      )}
    </div>
  );
}
