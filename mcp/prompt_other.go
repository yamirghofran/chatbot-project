//go:build !darwin && !linux && !freebsd && !openbsd && !netbsd

package main

import "fmt"

// readLineNoEcho on non-Unix systems falls back to normal input (visible).
func readLineNoEcho() (string, error) {
	return "", fmt.Errorf("secure input not supported on this platform")
}
