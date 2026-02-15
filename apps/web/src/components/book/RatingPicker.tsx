import { useState } from "react";
import { Star } from "lucide-react";

export type RatingPickerProps = {
  value?: number;
  onChange?: (value: number) => void;
};

export function RatingPicker({ value, onChange }: RatingPickerProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const display = hovered ?? value ?? 0;

  return (
    <div
      className="inline-flex items-center gap-0.5"
      onMouseLeave={() => setHovered(null)}
    >
      {[1, 2, 3, 4, 5].map((star) => {
        const fill =
          display >= star ? 1 : display >= star - 0.5 ? 0.5 : 0;

        return (
          <span key={star} className="relative size-7 cursor-pointer">
            <span
              className="absolute inset-y-0 left-0 w-1/2 z-10"
              onMouseEnter={() => setHovered(star - 0.5)}
              onClick={() => onChange?.(star - 0.5)}
              role="button"
              aria-label={`Rate ${star - 0.5} out of 5`}
            />
            <span
              className="absolute inset-y-0 right-0 w-1/2 z-10"
              onMouseEnter={() => setHovered(star)}
              onClick={() => onChange?.(star)}
              role="button"
              aria-label={`Rate ${star} out of 5`}
            />

            {/* Empty star (always rendered as base layer) */}
            <Star className="absolute inset-0 size-7 text-border" />

            {/* Filled overlay (clipped for half support) */}
            {fill > 0 && (
              <span
                className="absolute inset-0 overflow-hidden"
                style={{ width: fill === 1 ? "100%" : "50%" }}
              >
                <Star
                  className="size-7"
                  style={{ fill: "#FFCC00", color: "#FFCC00" }}
                />
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
