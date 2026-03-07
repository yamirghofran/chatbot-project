import { useState, type FormEvent } from "react";
import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import * as api from "@/lib/api";
import { getToken, setToken } from "@/lib/auth";

export const Route = createFileRoute("/signup")({
  component: Signup,
});

function Signup() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [step, setStep] = useState<1 | 2>(1);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [username, setUsername] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Already signed in
  if (getToken()) {
    navigate({ to: "/" });
    return null;
  }

  function handleStep1(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email.trim() || !password.trim()) return;
    setStep(2);
  }

  async function handleStep2(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!fullName.trim() || !username.trim()) return;
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.register(
        email.trim(),
        password.trim(),
        fullName.trim(),
        username.trim(),
      );
      setToken(result.access_token);
      await queryClient.invalidateQueries();
      navigate({ to: "/" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex min-h-[80vh] flex-col items-center justify-center">
      <div className="w-full max-w-sm">
        {/* Copy */}
        <div className="mb-8 text-center">
          <h1 className="font-serif text-2xl font-medium tracking-tight">
            {step === 1 ? "Create your account." : "Tell us who you are."}
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            {step === 1
              ? "Start with your email and a password."
              : "Choose a name and username."}
          </p>
        </div>

        {/* Step 1 */}
        {step === 1 && (
          <form onSubmit={handleStep1} className="space-y-3">
            <div>
              <label htmlFor="signup-email" className="mb-1 block text-xs text-muted-foreground">
                Email
              </label>
              <input
                id="signup-email"
                type="email"
                autoComplete="email"
                autoFocus
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                placeholder="you@example.com"
              />
            </div>
            <div>
              <label htmlFor="signup-password" className="mb-1 block text-xs text-muted-foreground">
                Password
              </label>
              <input
                id="signup-password"
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                placeholder="••••••••"
              />
            </div>
            <Button type="submit" variant="outline" className="w-full">
              Continue
            </Button>
          </form>
        )}

        {/* Step 2 */}
        {step === 2 && (
          <form onSubmit={handleStep2} className="space-y-3">
            <div>
              <label htmlFor="signup-name" className="mb-1 block text-xs text-muted-foreground">
                Full name
              </label>
              <input
                id="signup-name"
                type="text"
                autoComplete="name"
                autoFocus
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                placeholder="Jane Doe"
              />
            </div>
            <div>
              <label htmlFor="signup-username" className="mb-1 block text-xs text-muted-foreground">
                Username
              </label>
              <input
                id="signup-username"
                type="text"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                placeholder="janedoe"
              />
            </div>
            {error && <p className="text-xs text-destructive">{error}</p>}
            <Button type="submit" variant="outline" className="w-full" disabled={isLoading}>
              {isLoading ? "Creating account…" : "Create account"}
            </Button>
            <button
              type="button"
              onClick={() => setStep(1)}
              className="w-full text-center text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
            >
              Back
            </button>
          </form>
        )}

        <p className="mt-6 text-center text-xs text-muted-foreground">
          Already have an account?{" "}
          <Link to="/" className="underline underline-offset-2 hover:text-foreground">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
