package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/yamirghofran/bookdb-mcp/client"
)

// RegisterDiscoveryTools registers discovery and recommendation MCP tools.
func RegisterDiscoveryTools(register func(tool mcp.Tool, handler ToolHandler), api *client.APIClient) {
	// get_recommendations
	register(
		mcp.NewTool("get_recommendations",
			mcp.WithDescription("Get personalized book recommendations combining collaborative filtering (BPR model), semantic vector similarity, and clustering. Requires authentication for personalized results; falls back to popular books otherwise."),
			mcp.WithNumber("limit",
				mcp.Description("Maximum number of recommendations to return (1-100)"),
			),
		),
		makeGetRecommendations(api),
	)

	// get_staff_picks
	register(
		mcp.NewTool("get_staff_picks",
			mcp.WithDescription("Get a curated set of well-rated books as staff picks. Based on community ratings and reviews."),
			mcp.WithNumber("limit",
				mcp.Description("Maximum number of picks to return (1-50)"),
			),
		),
		makeGetStaffPicks(api),
	)
}

// ── get_recommendations ─────────────────────────────────────────────────────

func makeGetRecommendations(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		limit := int(request.GetFloat("limit", 20))
		if limit < 1 {
			limit = 1
		}
		if limit > 100 {
			limit = 100
		}

		books, err := api.GetRecommendations(ctx, limit)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get recommendations: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Book Recommendations (%d):\n\n", len(books)))
		for i, b := range books {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, client.FormatBookDetail(b)))
			if i < len(books)-1 {
				sb.WriteString("---\n")
			}
		}

		if len(books) == 0 {
			sb.WriteString("No recommendations available. Try adjusting your reading history or ratings.")
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}

// ── get_staff_picks ─────────────────────────────────────────────────────────

func makeGetStaffPicks(api *client.APIClient) ToolHandler {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		limit := int(request.GetFloat("limit", 6))
		if limit < 1 {
			limit = 1
		}
		if limit > 50 {
			limit = 50
		}

		books, err := api.GetStaffPicks(ctx, limit)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get staff picks: %v", err)), nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Staff Picks:\n\n"))
		for i, b := range books {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, client.FormatBookDetail(b)))
			if i < len(books)-1 {
				sb.WriteString("---\n")
			}
		}

		if len(books) == 0 {
			sb.WriteString("No staff picks available at the moment.")
		}

		return mcp.NewToolResultText(sb.String()), nil
	}
}
