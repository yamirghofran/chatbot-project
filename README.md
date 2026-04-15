# Machine Learning project template

The objective of this repository is to serve as a template for machine learning projects.

## Structure

- .github: CI/CD with GitHub Actions. It runs the tests every time there is a pull request to the repository.
- docs: Documentation of the project.
- examples: Jupyter notebooks with machine learning experiments. Here is where you would do data exploration, try different machine learning models, etc.
- bookdb: Libraries with common functions that you use in the project.
- tests: Python tests of the libraries.

## Setup

    pip install -e .
    python -c "import bookdb; print(bookdb.__version__)"

## Coding Principles

Next there are a few coding principles that I follow when working on machine learning projects.

### Start from something that works

Here is one of the most practical tips I know about working on machine learning. **Instead of starting from scratch, start with something that works and adapt it to your problem.**

For example, let's say you want to build a recommendation system with data from your company. What I would do is something as simple as this:

1. Go to [Recommenders](https://github.com/recommenders-team/recommenders) and look at an example that a similar dataset structure and compute. For example, if your data is text-based and you want to use GPU, explore the examples of LSTUR or NPA.
2. Install the dependencies and run the example. Make sure that it works.
3. Change the data of the example to your data. If your data is different or more extensive, just forget about it and use the part of your data that is similar to the example. Make sure that it works.
4. Change the code to adapt it to your specific data and problem.

### Notebooks that call a library

One of the main differences between a professional and an amateur machine learning project is this. Don't put your functions and classes in the notebooks, instead, create libraries and call them from the notebooks. This is the only way to reuse your code and make it scalable.

Most of the time, notebooks are not deployed, they are used for experimentation and visualization. You deploy the libraries. In addition, if you create libraries, you can test them.

### Why tests are important?

Tests solve one of the most expensive problems in development: maintenance. The way I see testing is like the immune system of your project. It protects your project from bugs and errors and makes sure your project is healthy.

A strong test pipeline minimizes maintenance. It is one of the best investments you can do in your project, because it will avoid new buggy code in the project, and it will detect breaking changes when using dependencies.

## MCP Server

A Go-based [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server lives under `mcp/`. It exposes BookDB's book search, recommendations, and discovery tools to LLM clients like Claude Desktop, Cursor, or any MCP-compatible application.

### Tools

| Tool | Description |
|------|-------------|
| `search_books` | Search books by title, author, or keyword (with AI fallback) |
| `get_book` | Get detailed book info, stats, and description |
| `get_related_books` | Find semantically similar books via vector embeddings |
| `get_book_reviews` | Read reviews and ratings for a book |
| `get_recommendations` | Personalized recommendations (BPR + vector clustering) |
| `get_staff_picks` | Curated set of well-rated books |
| `get_user_profile` | Look up a user's profile by username |
| `get_user_ratings` | View a user's rated books and scores |

### Install

One-line install (macOS, Linux, Windows):

```bash
curl -fsSL https://raw.githubusercontent.com/yamirghofran/BookDB/main/mcp/install.sh | bash
```

This downloads the latest binary for your platform, installs it to `~/.local/bin`, and prints setup instructions for any AI coding assistants it detects (Claude Desktop, Claude Code, Cursor, OpenCode, Codex).

### Build from Source

Requires [Go 1.24+](https://go.dev/dl/).

```bash
cd mcp
go build -o bookdb-mcp .
```

### Authentication

Most tools work without authentication (search, staff picks, public profiles). Personalized recommendations require a JWT token. Get one using the built-in login command:

```bash
# Interactive — prompts for email and password (password is hidden)
./bookdb-mcp login -email you@example.com

# Or specify everything on the command line (for scripting)
./bookdb-mcp login -email you@example.com -password yourpassword

# New account? Register first:
./bookdb-mcp register -email you@example.com -name "Your Name" -username yourname
```

The command prints the JWT token to stdout. Capture it in an environment variable:

```bash
export BOOKDB_API_KEY=$(./bookdb-mcp login -email you@example.com)
```

### Usage

#### CLI Flags & Environment Variables

| Flag | Env Variable | Default | Description |
|------|-------------|---------|-------------|
| `-api-url` | `BOOKDB_API_URL` | `http://localhost:8000` | BookDB API base URL |
| `-api-key` | `BOOKDB_API_KEY` | _(empty)_ | JWT token for auth |
| `-transport` | `MCP_TRANSPORT` | `stdio` | Transport: `stdio`, `sse`, `http` |
| `-addr` | `MCP_ADDR` | `:8080` | Listen address (SSE/HTTP) |

#### Claude Desktop (stdio)

Add to your Claude Desktop config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "bookdb": {
      "command": "/path/to/bookdb-mcp",
      "env": {
        "BOOKDB_API_URL": "http://localhost:8000",
        "BOOKDB_API_KEY": "your-jwt-token"
      }
    }
  }
}
```

#### Remote / HTTP

```bash
./bookdb-mcp -transport http -addr :9090 -api-url http://localhost:8000 -api-key eyJ...
```

#### SSE (Server-Sent Events)

```bash
./bookdb-mcp -transport sse -addr :8080
```

### Project Structure

```
mcp/
├── main.go           # CLI entry point, subcommands (login, register, server)
├── server.go         # MCP server setup and tool registration
├── install.sh        # curl | bash one-line installer
├── client/
│   └── api.go        # HTTP client for the BookDB FastAPI backend
├── tools/
│   ├── books.go      # search_books, get_book, get_related_books, get_book_reviews
│   ├── discovery.go  # get_recommendations, get_staff_picks
│   └── users.go      # get_user_profile, get_user_ratings
└── transport/
    ├── stdio.go      # Stdio transport (Claude Desktop, local LLMs)
    ├── sse.go        # SSE transport (browser integrations)
    └── http.go       # Streamable HTTP transport (remote deployment)
```

### Releasing

Push a tag to trigger the release workflow:

```bash
git tag mcp/v0.1.0
git push origin mcp/v0.1.0
```

This runs [`.github/workflows/release-mcp.yml`](.github/workflows/release-mcp.yml) which cross-compiles for 5 platforms (linux amd64/arm64, macos amd64/arm64, windows amd64), packages tarballs with SHA-256 checksums, and creates a GitHub release. The install script picks up the latest release automatically.

## Checklist

- [ ] Create a recommender system / chatbot
- [ ] Perform Expolratory Data Analysis (EDA)
- [ ] Deploy the system
- [ ] Have MLOps (versioning, testing, monitoring, etc)
- [ ] Follow good development practices
  - [ ] Work on branches
  - [ ] Add code via Pull Requests
  - [ ] Comment on issues
  - [ ] Libraries that are called by notebooks
  - [ ] Tests
- [ ] Evidence of exceptional ability as a group
