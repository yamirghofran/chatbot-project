package main

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/yamirghofran/bookdb-mcp/client"
	"github.com/yamirghofran/bookdb-mcp/tools"
)

const (
	serverName    = "BookDB"
	serverVersion = "0.1.0"
)

// serverInstructions tells connecting LLM clients what this server provides.
const serverInstructions = `BookDB is a book discovery and recommendation platform.

This server provides tools to:
- Search books by title, author, or keyword
- Get detailed book information and stats
- Find semantically similar books using vector embeddings
- Read book reviews and ratings
- Get personalized recommendations (collaborative filtering + semantic)
- Browse curated staff picks
- Look up user profiles and their reading history

All data comes from the BookDB API which combines PostgreSQL, Qdrant vector search,
and a BPR recommendation model to provide rich book discovery.`

// NewBookDBServer creates a fully-configured MCP server backed by the BookDB API.
func NewBookDBServer(apiClient *client.APIClient) *server.MCPServer {
	s := server.NewMCPServer(
		serverName,
		serverVersion,
		server.WithToolCapabilities(false),
		server.WithRecovery(),
		server.WithInstructions(serverInstructions),
	)

	// Create an adapter that bridges tools.ToolHandler -> server.ToolHandlerFunc.
	register := func(tool mcp.Tool, handler tools.ToolHandler) {
		s.AddTool(tool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return handler(ctx, request)
		})
	}

	// Register all tool groups.
	tools.RegisterBookTools(register, apiClient)
	tools.RegisterDiscoveryTools(register, apiClient)
	tools.RegisterUserTools(register, apiClient)

	return s
}
