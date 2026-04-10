package transport

import (
	"fmt"

	"github.com/mark3labs/mcp-go/server"
)

// ServeStdio starts the MCP server using the stdio transport.
// This is the default and recommended transport for local LLM integrations
// like Claude Desktop.
func ServeStdio(s *server.MCPServer) error {
	fmt.Println("Starting BookDB MCP server (stdio transport)...")
	if err := server.ServeStdio(s); err != nil {
		return fmt.Errorf("stdio server error: %w", err)
	}
	return nil
}
