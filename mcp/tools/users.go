package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/yamirghofran/bookdb-mcp/client"
)

// RegisterUserTools registers user-related MCP tools.
func RegisterUserTools(register func(tool mcp.Tool, handler ToolHandler), api *client.APIClient) {
	// get_user_profile
	register(
		mcp.NewTool("get_user_profile",
			mcp.WithDescription("Get a BookDB user's profile information by username."),
			mcp.WithString("username",
				mcp.Required(),
				mcp.Description("The username to look up"),
			),
		),
		makeGetUserProfile(api),
	)

	// get_user_ratings
	register(
		mcp.NewTool("get_user_ratings",
			mcp.WithDescription("Get a user's book ratings. Shows which books a user has rated and their scores, useful for understanding reading preferences."),
			mcp.WithString("username",
				mcp.Required(),
				mcp.Description("The username whose ratings to fetch"),
			),
			mcp.WithNumber("limit",
				mcp.Description("Maximum number of ratings to return (1-200)"),
			),
			mcp.WithString("sort",
				mcp.Description("Sort order: 'recent' for newest first, 'rating' for highest rated first"),
				mcp.Enum("recent", "rating"),
			),
		),
		makeGetUserRatings(api),
	)
}

// ── get_user_profile ────────────────────────────────────────────────────────

func makeGetUserProfile(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		username, err := request.RequireString("username")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		user, err := api.GetUserProfile(ctx, username)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get user profile: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("**%s** (@%s)\n", user.Name, user.Username))
		sb.WriteString(fmt.Sprintf("User ID: %s\n", user.ID))
		if user.AvatarURL != nil {
			sb.WriteString(fmt.Sprintf("Avatar: %s\n", *user.AvatarURL))
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}

// ── get_user_ratings ────────────────────────────────────────────────────────

func makeGetUserRatings(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		username, err := request.RequireString("username")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		limit := int(request.GetFloat("limit", 50))
		if limit < 1 {
			limit = 1
		}
		if limit > 200 {
			limit = 200
		}
		sortOrder := request.GetString("sort", "recent")

		ratings, err := api.GetUserRatings(ctx, username, limit, sortOrder)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get user ratings: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Ratings by @%s (showing %d, sorted by %s):\n\n", username, len(ratings), sortOrder))

		for _, r := range ratings {
			stars := strings.Repeat("⭐", r.Rating)
			sb.WriteString(fmt.Sprintf("- %s **%s**", stars, r.Book.Title))
			if r.Book.ID != "" {
				sb.WriteString(fmt.Sprintf(" [ID: %s]", r.Book.ID))
			}
			if r.Book.Author != "" {
				sb.WriteString(fmt.Sprintf(" by %s", r.Book.Author))
			}
			if r.RatedAt != nil {
				sb.WriteString(fmt.Sprintf(" (%s)", *r.RatedAt))
			}
			sb.WriteString("\n")
		}

		if len(ratings) == 0 {
			sb.WriteString("No ratings found for this user.")
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}
