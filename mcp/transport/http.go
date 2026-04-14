package transport

import (
	"fmt"

	"github.com/mark3labs/mcp-go/server"
)

// ServeHTTP starts the MCP server using the Streamable HTTP transport.
// Suitable for remote deployment and modern MCP clients.
func ServeHTTP(s *server.MCPServer, addr string) error {
	httpServer := server.NewStreamableHTTPServer(s)
	fmt.Printf("Starting BookDB MCP server (HTTP transport) on %s\n", addr)
	if err := httpServer.Start(addr); err != nil {
		return fmt.Errorf("HTTP server error: %w", err)
	}
	return nil
}
