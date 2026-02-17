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
        const isFilled = display >= star;

        return (
          <button
            key={star}
            type="button"
            className="relative size-7 cursor-pointer"
            onMouseEnter={() => setHovered(star)}
            onClick={() => onChange?.(star)}
            aria-label={`Rate ${star} out of 5`}
          >
            {/* Empty star (always rendered as base layer) */}
            <Star className="absolute inset-0 size-7 text-border" />

            {/* Filled overlay */}
            {isFilled && (
              <span className="absolute inset-0 overflow-hidden">
                <Star
                  className="size-7"
                  style={{ fill: "#FFCC00", color: "#FFCC00" }}
                />
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
