package tools

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/yamirghofran/bookdb-mcp/client"
)

// RegisterInteractionTools registers rating and review MCP tools.
func RegisterInteractionTools(register func(tool mcp.Tool, handler ToolHandler), api *client.APIClient) {
	// rate_book
	register(
		mcp.NewTool("rate_book",
			mcp.WithDescription("Rate a book on a 1-5 star scale. If multiple books match, returns the top candidates and asks for clarification."),
			mcp.WithString("book_query",
				mcp.Required(),
				mcp.Description("Book title, author, keyword, or numeric book ID. IMPORTANT: when a book ID is known (e.g. from search results showing [ID: 123]), always pass the ID as a number string like '123' — it is faster and avoids ambiguity."),
			),
			mcp.WithNumber("rating",
				mcp.Required(),
				mcp.Description("Rating from 1 to 5 stars"),
			),
		),
		makeRateBook(api),
	)

	// review_book
	register(
		mcp.NewTool("review_book",
			mcp.WithDescription("Write a text review for a book. Each user can only review a book once. If multiple books match, returns the top candidates and asks for clarification."),
			mcp.WithString("book_query",
				mcp.Required(),
				mcp.Description("Book title, author, keyword, or numeric book ID. IMPORTANT: when a book ID is known (e.g. from search results showing [ID: 123]), always pass the ID as a number string like '123' — it is faster and avoids ambiguity."),
			),
			mcp.WithString("text",
				mcp.Required(),
				mcp.Description("Your review text"),
			),
		),
		makeReviewBook(api),
	)
}

// ── rate_book ────────────────────────────────────────────────────────────────

func makeRateBook(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := request.RequireString("book_query")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		rating := int(request.GetFloat("rating", 0))

		if rating < 1 || rating > 5 {
			return mcp.NewToolResultError("Rating must be between 1 and 5 stars."), nil
		}

		resolved, result := resolveBookQuery(ctx, api, query)
		if result != nil {
			return result, nil
		}

		bookID, err := strconv.Atoi(resolved.ID)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Invalid book ID: %s", resolved.ID)), nil
		}

		if err := api.RateBook(ctx, bookID, rating); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to rate book: %v", err)), nil
		}

		stars := strings.Repeat("⭐", rating)
		return mcp.NewToolResultText(
			fmt.Sprintf("✅ Rated **%s** by %s %s", resolved.Title, resolved.Author, stars),
		), nil
	}
}

// ── review_book ──────────────────────────────────────────────────────────────

func makeReviewBook(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := request.RequireString("book_query")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		text, err := request.RequireString("text")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		if strings.TrimSpace(text) == "" {
			return mcp.NewToolResultError("Review text cannot be empty."), nil
		}

		resolved, result := resolveBookQuery(ctx, api, query)
		if result != nil {
			return result, nil
		}

		bookID, err := strconv.Atoi(resolved.ID)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Invalid book ID: %s", resolved.ID)), nil
		}

		review, err := api.ReviewBook(ctx, bookID, text)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to post review: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("✅ Review posted for **%s** by %s!\n\n", resolved.Title, resolved.Author))
		sb.WriteString(fmt.Sprintf("> %s\n", text))
		if review.ID != "" {
			sb.WriteString(fmt.Sprintf("\nReview ID: %s", review.ID))
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}
