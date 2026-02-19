import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[12px] supports-[corner-shape:squircle]:rounded-[70px] supports-[corner-shape:squircle]:[corner-shape:squircle] text-sm font-medium transition-all hover:drop-shadow-[0_0.5px_1px_rgba(0,0,0,0.06)] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 cursor-pointer dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground hover:bg-primary/90 shadow-[inset_0_-1px_2px_0_rgba(0,0.8,0,0.06),inset_0_-2px_4px_0_rgba(0,0.7,0.7,0.42),inset_0_1px_0_0_rgba(255,255,255,0.16)] hover:shadow-[inset_0_-1px_2px_0_rgba(0,0,0,0.28),inset_0_-3px_5px_0_rgba(0,0,0,0.14),inset_0_1px_0_0_rgba(255,255,255,0.18)]",
        destructive:
          "bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
        outline:
          "border bg-background hover:bg-accent hover:text-accent-foreground shadow-[inset_0_-1px_2px_0_rgba(0,0,0,0.08),inset_0_1px_0_0_rgba(255,255,255,0.22)] hover:shadow-[inset_0_-1px_2px_0_rgba(0,0,0,0.06),inset_0_1px_0_0_rgba(255,255,255,0.06)] dark:bg-input/30 dark:border-input dark:hover:bg-input/50",
        secondary:
          "bg-secondary text-foreground hover:bg-secondary/80 shadow-[inset_0_-1px_2px_0_rgba(0,0,0,0.01),inset_0_-2px_3px_0_rgba(0,0,0,0.06),inset_0_1px_0_0_rgba(255,255,255,0.24)] hover:shadow-[inset_0_-1px_2px_0_rgba(0,0,0,0.02),inset_0_-3px_4px_0_rgba(0,0,0,0.08),inset_0_1px_0_0_rgba(255,255,255,0.28)] border",
        ghost:
          "hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        xs: "h-6 gap-1 px-2 text-xs has-[>svg]:px-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 px-6 has-[>svg]:px-4",
        icon: "size-9",
        "icon-xs": "size-6 [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-8",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  }) {
  const Comp = asChild ? Slot.Root : "button";

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };
