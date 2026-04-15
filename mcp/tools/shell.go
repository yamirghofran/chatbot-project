package tools

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/yamirghofran/bookdb-mcp/client"
)

// RegisterShellTools registers shell (reading list) MCP tools.
func RegisterShellTools(register func(tool mcp.Tool, handler ToolHandler), api *client.APIClient) {
	// get_my_shell
	register(
		mcp.NewTool("get_my_shell",
			mcp.WithDescription("Get the books in the authenticated user's shell (reading list). Shows all books the user has saved to read later."),
		),
		makeGetMyShell(api),
	)

	// add_book_to_shell
	register(
		mcp.NewTool("add_book_to_shell",
			mcp.WithDescription("Add a book to the authenticated user's shell (reading list) by searching for it. If multiple books match, returns the top candidates and asks for clarification."),
			mcp.WithString("book_query",
				mcp.Required(),
				mcp.Description("Book title, author, keyword, or numeric book ID. IMPORTANT: when a book ID is known (e.g. from search results showing [ID: 123]), always pass the ID as a number string like '123' — it is faster and avoids ambiguity."),
			),
		),
		makeAddBookToShell(api),
	)
}

// ── get_my_shell ────────────────────────────────────────────────────────────

func makeGetMyShell(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		books, err := api.GetShell(ctx)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get shell: %v", err)), nil
		}

		var sb strings.Builder
		if len(books) == 0 {
			sb.WriteString("Your shell is empty. Use add_book_to_shell to start adding books!")
			return mcp.NewToolResultText(sb.String()), nil
		}

		sb.WriteString(fmt.Sprintf("Your Shell (%d books):\n\n", len(books)))
		for i, b := range books {
			sb.WriteString(fmt.Sprintf("%d. %s", i+1, formatShellBook(b)))
			sb.WriteString("\n---\n")
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}

// ── add_book_to_shell ───────────────────────────────────────────────────────

func makeAddBookToShell(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := request.RequireString("book_query")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		resolved, result := resolveBookQuery(ctx, api, query)
		if result != nil {
			return result, nil
		}

		bookID, err := strconv.Atoi(resolved.ID)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Invalid book ID: %s", resolved.ID)), nil
		}

		if err := api.AddToShell(ctx, bookID); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to add book to shell: %v", err)), nil
		}

		return mcp.NewToolResultText(
			fmt.Sprintf("✅ Added **%s** by %s to your shell!", resolved.Title, resolved.Author),
		), nil
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

// formatShellBook formats a compact Book for shell listing.
func formatShellBook(b client.Book) string {
	text := fmt.Sprintf("**%s** by %s", b.Title, b.Author)
	if b.ID != "" {
		text += fmt.Sprintf(" [ID: %s]", b.ID)
	}
	return text
}
