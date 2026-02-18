import type { SVGProps } from "react";

type TurtleShellIconProps = SVGProps<SVGSVGElement> & {
  filled?: boolean;
};

export function TurtleShellIcon({
  filled = false,
  ...props
}: TurtleShellIconProps) {
  return (
    <svg
      viewBox="0 -65 690 390"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      {filled && (
        <ellipse
          cx="282"
          cy="163"
          rx="220"
          ry="105"
          fill="currentColor"
          opacity="0.24"
        />
      )}
      <g>
        <path
          d="M65.2866 262.32C38.0802 -15.1389 350.1 -62.9836 467.146 143.19C477.475 161.385 528.286 246.881 498.999 270.094C473.294 290.464 326.795 285.626 238.187 285.626C186.998 285.626 137.304 285.626 89.7964 285.626"
          stroke="currentColor"
          strokeOpacity="0.9"
          strokeWidth="32"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M513.64 186.962C668.389 16.0034 668.389 186.962 507.981 266.289"
          stroke="currentColor"
          strokeOpacity="0.9"
          strokeWidth="32"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M582.808 158.21C583.181 157.033 583.555 155.848 583.928 154.671"
          stroke="currentColor"
          strokeOpacity="0.9"
          strokeWidth="32"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M476.242 298.312C471.498 302.971 459.183 300.269 449.791 299.884"
          stroke="currentColor"
          strokeOpacity="0.9"
          strokeWidth="32"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M148.256 288.537C132.777 324.881 108.579 301.223 100.645 290.296"
          stroke="currentColor"
          strokeOpacity="0.9"
          strokeWidth="32"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M39.8089 262.536C31.8177 263.639 23.6486 265.499 16.0035 267.26"
          stroke="currentColor"
          strokeOpacity="0.9"
          strokeWidth="32"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    </svg>
  );
}
