package transport

import (
	"fmt"

	"github.com/mark3labs/mcp-go/server"
)

// ServeSSE starts the MCP server using the SSE (Server-Sent Events) transport.
// Suitable for browser-based integrations and web clients.
func ServeSSE(s *server.MCPServer, addr string) error {
	sseServer := server.NewSSEServer(s)
	fmt.Printf("Starting BookDB MCP server (SSE transport) on %s\n", addr)
	if err := sseServer.Start(addr); err != nil {
		return fmt.Errorf("SSE server error: %w", err)
	}
	return nil
}
