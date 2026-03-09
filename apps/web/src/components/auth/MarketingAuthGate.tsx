import { type FormEvent } from "react";
import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";

export type MarketingAuthGateProps = {
  onAuthenticated?: (payload: { email: string; password: string }) => void;
  error?: string | null;
  isLoading?: boolean;
  email: string;
  onEmailChange: (v: string) => void;
  password: string;
  onPasswordChange: (v: string) => void;
};

export function MarketingAuthGate({
  onAuthenticated,
  error,
  isLoading,
  email,
  onEmailChange,
  password,
  onPasswordChange,
}: MarketingAuthGateProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextEmail = email.trim();
    const nextPassword = password.trim();
    if (!nextEmail || !nextPassword) return;
    onAuthenticated?.({ email: nextEmail, password: nextPassword });
  }

  return (
    <div className="fixed inset-0 flex items-center justify-center overflow-hidden md:justify-end md:pr-[25.67%]">
      {/* Background — desktop only */}
      <div
        className="pointer-events-none absolute inset-0 hidden bg-cover bg-center bg-no-repeat xl:block"
        style={{ backgroundImage: "url('/composition-4.svg')" }}
      />

      <div className="relative w-full max-w-sm px-6 md:px-0">
        <div className="mb-4 flex justify-center md:mb-6">
          <img src="/logo.svg" alt="BookDB" className="h-6 w-auto md:h-8" />
        </div>

        {/* Marketing copy */}
        <div className="mb-6 text-center md:mb-8">
          <h1 className="font-serif text-xl font-medium tracking-tight md:text-2xl">
            Go slow. Only fools rush.
          </h1>
          <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground md:mt-2 md:text-sm">
            Discover books aligned with your interests. Record what you read.
            Develop your taste over time.
          </p>
        </div>

        {/* Sign in form */}
        <form onSubmit={handleSubmit} className="space-y-2.5 md:space-y-3">
          <div>
            <label
              htmlFor="auth-email"
              className="mb-1 block text-xs text-muted-foreground"
            >
              Email
            </label>
            <input
              id="auth-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => onEmailChange(e.target.value)}
              className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label
              htmlFor="auth-password"
              className="mb-1 block text-xs text-muted-foreground"
            >
              Password
            </label>
            <input
              id="auth-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => onPasswordChange(e.target.value)}
              className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              placeholder="••••••••"
            />
          </div>

          {error && <p className="text-xs text-destructive">{error}</p>}

          <Button
            type="submit"
            variant="outline"
            className="w-full"
            disabled={isLoading}
          >
            {isLoading ? "Please wait…" : "Continue"}
          </Button>
        </form>

        <p className="mt-3 text-center text-xs text-muted-foreground md:mt-4">
          Don't have an account?{" "}
          <Link
            to="/signup"
            className="underline underline-offset-2 hover:text-foreground"
          >
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
