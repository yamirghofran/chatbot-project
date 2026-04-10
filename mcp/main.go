package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/yamirghofran/bookdb-mcp/client"
	"github.com/yamirghofran/bookdb-mcp/transport"
)

func main() {
	// Check for subcommands first.
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "login":
			runLogin(os.Args[2:])
			return
		case "register":
			runRegister(os.Args[2:])
			return
		case "help", "--help", "-h":
			printUsage()
			return
		}
	}

	// Default: start MCP server.
	runServer()
}

func printUsage() {
	fmt.Print(`bookdb-mcp — MCP server for the BookDB book recommendation platform

Usage:
  bookdb-mcp [flags]             Start the MCP server (default: stdio)
  bookdb-mcp login [flags]       Log in and print a JWT token
  bookdb-mcp register [flags]    Create a new account and print a JWT token

Server flags:
  -transport string   Transport protocol: stdio (default), sse, http
  -addr string        Listen address for SSE/HTTP transports (default ":8080")
  -api-url string     BookDB API base URL (env: BOOKDB_API_URL, default: http://localhost:8000)
  -api-key string     BookDB API JWT token (env: BOOKDB_API_KEY)

Login flags:
  -email string       Your account email
  -password string    Your account password (omit to be prompted securely)
  -api-url string     BookDB API base URL

Register flags:
  -email string       Email for the new account
  -name string        Your display name
  -username string    Desired username
  -password string    Password (omit to be prompted securely)
  -api-url string     BookDB API base URL

Environment variables:
  BOOKDB_API_URL      API base URL (default: http://localhost:8000)
  BOOKDB_API_KEY      JWT token for authenticated endpoints
  MCP_TRANSPORT       Transport: stdio, sse, http (default: stdio)
  MCP_ADDR            Listen address for SSE/HTTP (default: :8080)

Examples:
  # Start MCP server for Claude Desktop (stdio)
  BOOKDB_API_KEY=eyJ... bookdb-mcp

  # Log in to get a token
  bookdb-mcp login -email user@example.com

  # Start HTTP server for remote access
  bookdb-mcp -transport http -addr :9090
`)
}

// ── Server subcommand ────────────────────────────────────────────────────────

func runServer() {
	// CLI flags (override env vars).
	transportFlag := flag.String("transport", "", "Transport protocol: stdio, sse, http")
	addrFlag := flag.String("addr", "", "Listen address for SSE/HTTP transports (e.g. :8080)")
	apiURLFlag := flag.String("api-url", "", "BookDB API base URL")
	apiKeyFlag := flag.String("api-key", "", "BookDB API authentication key (JWT)")
	flag.Parse()

	// Resolve configuration: CLI flags > env vars > defaults.
	transportMode := envOrFlag("MCP_TRANSPORT", *transportFlag, "stdio")
	addr := envOrFlag("MCP_ADDR", *addrFlag, ":8080")
	apiURL := envOrFlag("BOOKDB_API_URL", *apiURLFlag, "http://localhost:8000")
	apiKey := envOrFlag("BOOKDB_API_KEY", *apiKeyFlag, "")

	// Build the API client.
	apiClient := client.NewAPIClient(apiURL, apiKey)

	// Build the MCP server with all tools registered.
	mcpServer := NewBookDBServer(apiClient)

	// Start the selected transport.
	var err error
	switch transportMode {
	case "stdio":
		err = transport.ServeStdio(mcpServer)
	case "sse":
		err = transport.ServeSSE(mcpServer, addr)
	case "http":
		err = transport.ServeHTTP(mcpServer, addr)
	default:
		fmt.Fprintf(os.Stderr, "Unknown transport: %s (use: stdio, sse, http)\n", transportMode)
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}

// ── Login subcommand ─────────────────────────────────────────────────────────

func runLogin(args []string) {
	fs := flag.NewFlagSet("login", flag.ExitOnError)
	email := fs.String("email", "", "Your account email")
	password := fs.String("password", "", "Your password (omit to be prompted)")
	apiURL := fs.String("api-url", "", "BookDB API base URL")
	fs.Parse(args)

	baseURL := envOrFlag("BOOKDB_API_URL", *apiURL, "http://localhost:8000")

	if *email == "" {
		*email = prompt("Email: ")
	}
	if *password == "" {
		*password = promptSecret("Password: ")
	}

	if *email == "" || *password == "" {
		fmt.Fprintln(os.Stderr, "Error: email and password are required.")
		os.Exit(1)
	}

	apiClient := client.NewAPIClient(baseURL, "")
	token, err := apiClient.Login(context.Background(), *email, *password)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Login failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(token.AccessToken)
}

// ── Register subcommand ──────────────────────────────────────────────────────

func runRegister(args []string) {
	fs := flag.NewFlagSet("register", flag.ExitOnError)
	email := fs.String("email", "", "Email for the new account")
	name := fs.String("name", "", "Your display name")
	username := fs.String("username", "", "Desired username")
	password := fs.String("password", "", "Password (omit to be prompted)")
	apiURL := fs.String("api-url", "", "BookDB API base URL")
	fs.Parse(args)

	baseURL := envOrFlag("BOOKDB_API_URL", *apiURL, "http://localhost:8000")

	if *email == "" {
		*email = prompt("Email: ")
	}
	if *name == "" {
		*name = prompt("Display name: ")
	}
	if *username == "" {
		*username = prompt("Username: ")
	}
	if *password == "" {
		*password = promptSecret("Password: ")
	}

	if *email == "" || *name == "" || *username == "" || *password == "" {
		fmt.Fprintln(os.Stderr, "Error: all fields are required.")
		os.Exit(1)
	}

	apiClient := client.NewAPIClient(baseURL, "")
	token, err := apiClient.Register(context.Background(), *email, *name, *username, *password)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Registration failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(token.AccessToken)
}

// ── Prompt helpers ───────────────────────────────────────────────────────────

// prompt reads a line from stdin with the given prompt text.
func prompt(label string) string {
	fmt.Print(label)
	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		return strings.TrimSpace(scanner.Text())
	}
	return ""
}

// promptSecret reads a password from stdin without echo (falls back to normal
// prompt if terminal echo cannot be disabled).
func promptSecret(label string) string {
	fmt.Print(label)
	// Try disabling echo via /dev/tty for Unix systems.
	// Fall back to normal input if unavailable.
	line, err := readLineNoEcho()
	if err != nil {
		// Fallback: just read normally (visible input).
		scanner := bufio.NewScanner(os.Stdin)
		if scanner.Scan() {
			return strings.TrimSpace(scanner.Text())
		}
		return ""
	}
	fmt.Println() // newline after the hidden input
	return strings.TrimSpace(line)
}

// envOrFlag returns the CLI flag value if non-empty, else the env var, else the default.
func envOrFlag(envVar, flagVal, defaultVal string) string {
	if flagVal != "" {
		return flagVal
	}
	if v := os.Getenv(envVar); v != "" {
		return v
	}
	return defaultVal
}
