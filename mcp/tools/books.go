// Package tools defines MCP tool definitions and handlers for book operations.
package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/yamirghofran/bookdb-mcp/client"
)

// ToolHandler is the function signature for MCP tool handlers.
type ToolHandler func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error)

// RegisterBookTools registers all book-related MCP tools using the provided
// registration function. This decouples tools from the MCP server implementation.
func RegisterBookTools(register func(tool mcp.Tool, handler ToolHandler), api *client.APIClient) {
	// search_books
	register(
		mcp.NewTool("search_books",
			mcp.WithDescription("Search for books by title, author, or keyword. Returns matching books ranked by relevance and popularity."),
			mcp.WithString("query",
				mcp.Required(),
				mcp.Description("Search query — book title, author name, genre, or keyword"),
			),
			mcp.WithNumber("limit",
				mcp.Description("Maximum number of results to return (1-100)"),
			),
		),
		makeSearchBooks(api),
	)

	// get_book
	register(
		mcp.NewTool("get_book",
			mcp.WithDescription("Get detailed information about a specific book including description, stats (average rating, review count), and tags."),
			mcp.WithNumber("book_id",
				mcp.Required(),
				mcp.Description("The internal BookDB book ID"),
			),
		),
		makeGetBook(api),
	)

	// get_related_books
	register(
		mcp.NewTool("get_related_books",
			mcp.WithDescription("Find books that are semantically similar to a given book using vector embeddings. Great for discovering 'if you liked X, try Y' recommendations."),
			mcp.WithNumber("book_id",
				mcp.Required(),
				mcp.Description("The book ID to find similar books for"),
			),
			mcp.WithNumber("limit",
				mcp.Description("Maximum number of related books to return (1-20)"),
			),
		),
		makeGetRelatedBooks(api),
	)

	// get_book_reviews
	register(
		mcp.NewTool("get_book_reviews",
			mcp.WithDescription("Get reader reviews and ratings for a specific book. Returns review text, like counts, and reviewer info."),
			mcp.WithNumber("book_id",
				mcp.Required(),
				mcp.Description("The book ID to get reviews for"),
			),
			mcp.WithNumber("limit",
				mcp.Description("Maximum number of reviews to return (1-100)"),
			),
			mcp.WithNumber("offset",
				mcp.Description("Number of reviews to skip (for pagination)"),
			),
		),
		makeGetBookReviews(api),
	)
}

// ── search_books ────────────────────────────────────────────────────────────

func makeSearchBooks(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := request.RequireString("query")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		limit := int(request.GetFloat("limit", 20))
		if limit < 1 {
			limit = 1
		}
		if limit > 100 {
			limit = 100
		}

		result, err := api.SearchBooks(ctx, query, limit)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Search failed: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Search results for \"%s\":\n\n", query))

		if result.DirectHit != nil {
			sb.WriteString("**Best Match:**\n")
			sb.WriteString(client.FormatBookDetail(*result.DirectHit))
			sb.WriteString("\n\n")
		}

		if len(result.KeywordResults) > 0 {
			sb.WriteString(fmt.Sprintf("**Other Results (%d):**\n", len(result.KeywordResults)))
			for _, b := range result.KeywordResults {
				sb.WriteString(client.FormatBookDetail(b))
				sb.WriteString("\n---\n")
			}
		}

		if result.AINarrative != nil && *result.AINarrative != "" {
			sb.WriteString(fmt.Sprintf("\n**AI Recommendation:**\n%s\n", *result.AINarrative))
			if len(result.AIBooks) > 0 {
				sb.WriteString("\n**AI-Recommended Books:**\n")
				for _, b := range result.AIBooks {
					sb.WriteString(client.FormatBookDetail(b))
					sb.WriteString("\n---\n")
				}
			}
		}

		if result.DirectHit == nil && len(result.KeywordResults) == 0 && len(result.AIBooks) == 0 {
			sb.WriteString("No books found matching your query.")
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}

// ── get_book ────────────────────────────────────────────────────────────────

func makeGetBook(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		bookID, err := request.RequireFloat("book_id")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		book, err := api.GetBook(ctx, int(bookID))
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get book: %v", err)), nil
		}

		return mcp.NewToolResultText(client.FormatBookDetail(*book)), nil
	}
}

// ── get_related_books ───────────────────────────────────────────────────────

func makeGetRelatedBooks(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		bookID, err := request.RequireFloat("book_id")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		limit := int(request.GetFloat("limit", 6))
		if limit < 1 {
			limit = 1
		}
		if limit > 20 {
			limit = 20
		}

		books, err := api.GetRelatedBooks(ctx, int(bookID), limit)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get related books: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Books related to book #%d:\n\n", int(bookID)))
		for i, b := range books {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, client.FormatBookDetail(b)))
			if i < len(books)-1 {
				sb.WriteString("---\n")
			}
		}

		if len(books) == 0 {
			sb.WriteString("No related books found.")
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}

// ── get_book_reviews ────────────────────────────────────────────────────────

func makeGetBookReviews(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		bookID, err := request.RequireFloat("book_id")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		limit := int(request.GetFloat("limit", 20))
		offset := int(request.GetFloat("offset", 0))

		result, err := api.GetBookReviews(ctx, int(bookID), limit, offset)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get reviews: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Reviews for book #%d (showing %d of %d):\n\n", int(bookID), len(result.Items), result.Total))

		for i, r := range result.Items {
			sb.WriteString(fmt.Sprintf("**%d. %s** (@%s)\n", offset+i+1, r.User.Name, r.User.Username))
			sb.WriteString(fmt.Sprintf("%s\n", r.Text))
			sb.WriteString(fmt.Sprintf("❤️ %d likes · %s\n\n", r.Likes, r.Timestamp))
		}

		if len(result.Items) == 0 {
			sb.WriteString("No reviews yet.")
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}
