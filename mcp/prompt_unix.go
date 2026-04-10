//go:build darwin || linux || freebsd || openbsd || netbsd

package main

import (
	"fmt"
	"os"
	"os/exec"
)

// readLineNoEcho reads a line from the terminal without echoing input
// by stty -echo on Unix systems.
func readLineNoEcho() (string, error) {
	// Disable terminal echo
	_ = exec.Command("stty", "-echo").Run()
	defer func() {
		// Re-enable echo
		_ = exec.Command("stty", "echo").Run()
	}()

	var line []byte
	buf := make([]byte, 1)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil || n == 0 {
			break
		}
		if buf[0] == '\n' || buf[0] == '\r' {
			break
		}
		line = append(line, buf[0])
	}

	if len(line) == 0 {
		return "", fmt.Errorf("empty input")
	}
	return string(line), nil
}
