import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";

export type MarketingAuthGateProps = {
  onAuthenticated?: (payload: {
    email: string;
    password: string;
    mode: "signin" | "signup";
    name?: string;
    username?: string;
  }) => void;
  error?: string | null;
  isLoading?: boolean;
};

export function MarketingAuthGate({
  onAuthenticated,
  error,
  isLoading,
}: MarketingAuthGateProps) {
  const [fullName, setFullName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"signin" | "signup">("signin");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (mode === "signup" && (!fullName.trim() || !username.trim())) return;
    const nextEmail = email.trim();
    const nextPassword = password.trim();
    if (!nextEmail || !nextPassword) return;
    onAuthenticated?.({
      email: nextEmail,
      password: nextPassword,
      mode,
      name: fullName.trim() || undefined,
      username: username.trim() || undefined,
    });
  }

  function handleModeChange(next: "signin" | "signup") {
    setMode(next);
  }

  return (
    <section className="mx-auto w-full max-w-xs pt-[10rem]">
      <div className="space-y-4 text-center">
        <div className="space-y-4 pt-4">
          <img
            src="/logo.svg"
            alt="BookDB logo"
            className="mx-auto h-10 w-auto"
          />

          <div className="py-4 text-left sm:py-5">
            <form onSubmit={handleSubmit}>
              <div className="mb-4 inline-flex text-xs">
                <button
                  type="button"
                  onClick={() => handleModeChange("signin")}
                  className={`rounded-md supports-[corner-shape:squircle]:rounded-[110px] supports-[corner-shape:squircle]:[corner-shape:squircle] px-3 py-1 ${mode === "signin" ? "bg-muted text-foreground" : "text-muted-foreground"}`}
                >
                  Sign in
                </button>
                <button
                  type="button"
                  onClick={() => handleModeChange("signup")}
                  className={`rounded-md supports-[corner-shape:squircle]:rounded-[110px] supports-[corner-shape:squircle]:[corner-shape:squircle] px-3 py-1 ${mode === "signup" ? "bg-muted text-foreground" : "text-muted-foreground"}`}
                >
                  Sign up
                </button>
              </div>

              <p className="mb-3 text-xs text-muted-foreground">
                {mode === "signin"
                  ? "Welcome back. Sign in to continue."
                  : "Create your account to get started."}
              </p>

              <div className="space-y-3">
                {mode === "signup" && (
                  <>
                    <div>
                      <label
                        htmlFor="auth-full-name"
                        className="mb-1 block text-xs text-muted-foreground"
                      >
                        Full name
                      </label>
                      <input
                        id="auth-full-name"
                        type="text"
                        autoComplete="name"
                        value={fullName}
                        onChange={(event) => setFullName(event.target.value)}
                        className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
                        placeholder="Jane Doe"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="auth-username"
                        className="mb-1 block text-xs text-muted-foreground"
                      >
                        Username
                      </label>
                      <input
                        id="auth-username"
                        type="text"
                        autoComplete="username"
                        value={username}
                        onChange={(event) => setUsername(event.target.value)}
                        className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
                        placeholder="janedoe"
                      />
                    </div>
                  </>
                )}
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
                    onChange={(event) => setEmail(event.target.value)}
                    className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
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
                    autoComplete={
                      mode === "signin" ? "current-password" : "new-password"
                    }
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
                    placeholder="••••••••"
                  />
                </div>
              </div>

              {error && (
                <p className="mt-3 text-xs text-destructive">{error}</p>
              )}

              <Button
                type="submit"
                variant="outline"
                className="mt-4 w-full"
                disabled={isLoading}
              >
                {isLoading
                  ? "Please wait…"
                  : mode === "signin"
                    ? "Continue to BookDB"
                    : "Create account"}
              </Button>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
}
