package tools

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/yamirghofran/bookdb-mcp/client"
)

// resolveBookQuery resolves a user's book query (title, author, keyword, or numeric ID)
// to a single BookDetail. Returns:
//   - (book, nil)       — single match resolved, caller should proceed
//   - (nil, result)     — error or disambiguation message, caller should return it
func resolveBookQuery(ctx context.Context, api *client.APIClient, query string) (*client.BookDetail, *mcp.CallToolResult) {
	// If the query is a pure number, treat it as a book ID directly.
	if bookID, err := strconv.Atoi(query); err == nil {
		book, err := api.GetBook(ctx, bookID)
		if err != nil {
			return nil, mcp.NewToolResultError(fmt.Sprintf("Book with ID %d not found: %v", bookID, err))
		}
		return book, nil
	}

	// Text search — top 3 candidates for disambiguation
	result, err := api.SearchBooks(ctx, query, 3)
	if err != nil {
		return nil, mcp.NewToolResultError(fmt.Sprintf("Search failed: %v", err))
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
		return nil, mcp.NewToolResultError(
			fmt.Sprintf("No books found matching \"%s\". Try a different title, author, or keyword.", query),
		)
	}

	// Single unambiguous result
	if len(candidates) == 1 {
		return &candidates[0], nil
	}

	// Multiple matches — ask for clarification
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Multiple books found matching \"%s\":\n\n", query))
	for i, b := range candidates {
		sb.WriteString(fmt.Sprintf("%d. %s\n---\n", i+1, client.FormatBookDetail(b)))
	}
	sb.WriteString("\nTo pick a book, reply with its numeric ID (e.g. '42'). Using the ID is the most reliable way to specify which book you mean.")

	return nil, mcp.NewToolResultText(sb.String())
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
