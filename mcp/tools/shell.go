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
				mcp.Description("Book title, author, or keyword to search for (e.g. 'The Hobbit' or 'Dune by Frank Herbert')"),
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

		// Search for the book (top 3 candidates for disambiguation)
		result, err := api.SearchBooks(ctx, query, 3)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Search failed: %v", err)), nil
		}

		// Collect all candidates from the search result
		var candidates []client.BookDetail
		if result.DirectHit != nil {
			candidates = append(candidates, *result.DirectHit)
		}
		candidates = append(candidates, result.KeywordResults...)
		candidates = append(candidates, result.AIBooks...)

		// Deduplicate by ID
		candidates = dedupByID(candidates)

		if len(candidates) == 0 {
			return mcp.NewToolResultError(
				fmt.Sprintf("No books found matching \"%s\". Try a different title, author, or keyword.", query),
			), nil
		}

		// Single unambiguous result — add it
		if len(candidates) == 1 {
			bookID, err := strconv.Atoi(candidates[0].ID)
			if err != nil {
				return mcp.NewToolResultError(fmt.Sprintf("Invalid book ID: %s", candidates[0].ID)), nil
			}

			if err := api.AddToShell(ctx, bookID); err != nil {
				return mcp.NewToolResultError(fmt.Sprintf("Failed to add book to shell: %v", err)), nil
			}

			book := candidates[0]
			return mcp.NewToolResultText(
				fmt.Sprintf("✅ Added **%s** by %s to your shell!", book.Title, book.Author),
			), nil
		}

		// Multiple matches — ask for clarification
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Multiple books found matching \"%s\":\n\n", query))
		for i, b := range candidates {
			sb.WriteString(fmt.Sprintf("%d. %s\n---\n", i+1, client.FormatBookDetail(b)))
		}
		sb.WriteString("\nPlease specify which book by providing the exact title, author, or book ID.")

		return mcp.NewToolResultText(sb.String()), nil
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

// dedupByID removes duplicate BookDetail entries by ID, preserving order.
func dedupByID(books []client.BookDetail) []client.BookDetail {
	seen := make(map[string]struct{}, len(books))
	result := make([]client.BookDetail, 0, len(books))
	for _, b := range books {
		if b.ID == "" {
			continue
		}
		if _, ok := seen[b.ID]; ok {
			continue
		}
		seen[b.ID] = struct{}{}
		result = append(result, b)
	}
	return result
}
