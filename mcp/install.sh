#!/usr/bin/env bash
set -eo pipefail

# BookDB MCP Server installer
# Usage: curl -fsSL https://raw.githubusercontent.com/yamirghofran/BookDB/main/mcp/install.sh | bash

REPO="yamirghofran/BookDB"
BINARY_NAME="bookdb-mcp"
INSTALL_DIR="${BOOKDB_MCP_INSTALL_DIR:-$HOME/.local/bin}"

info()    { printf '\033[1;34m%s\033[0m\n' "$*"; }
success() { printf '\033[1;32m%s\033[0m\n' "$*"; }
warn()    { printf '\033[1;33m%s\033[0m\n' "$*"; }
error()   { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

print_json() {
    if command -v jq &>/dev/null; then
        echo "$1" | jq .
    else
        echo "$1"
    fi
}

# ── Platform detection ───────────────────────────────────────────────────────

detect_platform() {
    local os arch

    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)
            case "$arch" in
                x86_64)         echo "x86_64-unknown-linux-musl" ;;
                aarch64|arm64)  echo "aarch64-unknown-linux-musl" ;;
                *)              error "Unsupported Linux architecture: $arch" ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64)         echo "x86_64-apple-darwin" ;;
                aarch64|arm64)  echo "aarch64-apple-darwin" ;;
                *)              error "Unsupported macOS architecture: $arch" ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            case "$arch" in
                x86_64)         echo "x86_64-pc-windows-msvc" ;;
                aarch64|arm64)  echo "aarch64-pc-windows-msvc" ;;
                *)              error "Unsupported Windows architecture: $arch" ;;
            esac
            ;;
        *)
            error "Unsupported OS: $os"
            ;;
    esac
}

# ── Release lookup ───────────────────────────────────────────────────────────

get_latest_release_tag() {
    local target="$1"

    local releases_json
    releases_json=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases") \
        || error "Failed to fetch releases from https://github.com/${REPO}/releases"

    # Find the first release that contains a binary for our platform
    local tag
    tag=$(echo "$releases_json" \
        | grep -oE '"(tag_name|name)": *"[^"]*"' \
        | awk -v target="bookdb-mcp-${target}" '
            /"tag_name":/ { gsub(/.*": *"|"/, ""); current_tag = $0; next }
            /"name":/ && index($0, target) { print current_tag; exit }
        ')

    if [ -z "$tag" ]; then
        error "No release found containing ${BINARY_NAME} for ${target}.

Check available releases at: https://github.com/${REPO}/releases"
    fi
    echo "$tag"
}

# ── Download & install ───────────────────────────────────────────────────────

download_binary() {
    local target="$1"
    local tag="$2"
    local ext=""

    case "$target" in
        *windows*) ext=".exe" ;;
    esac

    local filename="${BINARY_NAME}-${target}.tar.gz"
    local url="https://github.com/${REPO}/releases/download/${tag}/${filename}"
    local checksum_url="${url}.sha256"

    info "Downloading ${filename} from release ${tag}..."

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT

    if ! curl -fsSL -o "${tmp_dir}/${filename}" "$url" 2>/dev/null; then
        echo "" >&2
        printf '\033[1;31mError: Failed to download binary for your platform.\033[0m\n' >&2
        echo "" >&2
        echo "  URL: ${url}" >&2
        echo "  Release: ${tag}" >&2
        echo "  Platform: ${target}" >&2
        echo "" >&2
        echo "This likely means the MCP binary hasn't been built for this platform yet." >&2
        echo "Check available releases at: https://github.com/${REPO}/releases" >&2
        exit 1
    fi

    # Verify checksum
    if command -v sha256sum &>/dev/null; then
        if curl -fsSL -o "${tmp_dir}/${filename}.sha256" "$checksum_url" 2>/dev/null; then
            info "Verifying checksum..."
            (cd "$tmp_dir" && sha256sum -c "${filename}.sha256") \
                || error "Checksum verification failed!"
        else
            warn "Checksum file not available, skipping verification."
        fi
    else
        warn "sha256sum not found, skipping checksum verification."
    fi

    # Extract
    tar xzf "${tmp_dir}/${filename}" -C "${tmp_dir}"

    # Install
    mkdir -p "$INSTALL_DIR"
    mv "${tmp_dir}/${BINARY_NAME}${ext}" "${INSTALL_DIR}/${BINARY_NAME}${ext}"
    chmod +x "${INSTALL_DIR}/${BINARY_NAME}${ext}"

    if [ "$IS_UPDATE" != true ]; then
        success "Installed ${BINARY_NAME} to ${INSTALL_DIR}/${BINARY_NAME}${ext}"
    fi
}

# ── PATH check ───────────────────────────────────────────────────────────────

check_path() {
    case ":$PATH:" in
        *":${INSTALL_DIR}:"*) return 0 ;;
    esac

    warn "${INSTALL_DIR} is not in your PATH."
    echo ""
    echo "Add it to your shell profile:"
    echo ""

    local shell_name
    shell_name="$(basename "${SHELL:-bash}")"
    case "$shell_name" in
        zsh)
            echo "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.zshrc"
            echo "  source ~/.zshrc"
            ;;
        fish)
            echo "  fish_add_path ${INSTALL_DIR}"
            ;;
        *)
            echo "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc"
            echo "  source ~/.bashrc"
            ;;
    esac
    echo ""
}

# ── Setup instructions per detected tool ─────────────────────────────────────

print_setup_instructions() {
    local binary_path="${INSTALL_DIR}/${BINARY_NAME}"
    local found_any=false

    echo ""
    success "BookDB MCP Server installed successfully!"
    echo ""
    info "Setup with your AI coding assistant:"
    echo ""

    # ── Claude Desktop ───────────────────────────────────────────────────
    local claude_config_dir claude_config_file
    case "$(uname -s)" in
        Darwin) claude_config_dir="$HOME/Library/Application Support/Claude" ;;
        Linux)  claude_config_dir="$HOME/.config/Claude" ;;
        *)      claude_config_dir="" ;;
    esac
    claude_config_file="${claude_config_dir}/claude_desktop_config.json"

    if [ -n "$claude_config_dir" ] && ([ -d "$claude_config_dir" ] || [ -f "$claude_config_file" ]); then
        found_any=true
        success "[Claude Desktop] config detected at ${claude_config_file}"
        echo ""
        echo "Add to ${claude_config_file}:"
        echo ""
        print_json "{
  \"mcpServers\": {
    \"bookdb\": {
      \"command\": \"${binary_path}\",
      \"env\": {
        \"BOOKDB_API_URL\": \"https://bookdb.up.railway.app\",
        \"BOOKDB_API_KEY\": \"\"
      }
    }
  }
}"
        echo ""
    fi

    # ── Claude Code (CLI) ────────────────────────────────────────────────
    if command -v claude &>/dev/null; then
        found_any=true
        success "[Claude Code] detected"
        echo ""
        echo "  claude mcp add -s user bookdb -- ${binary_path}"
        echo ""
        echo "To set the API URL and key (persisted in server config):"
        echo "  claude mcp add -s user \\"
        echo "    -e BOOKDB_API_URL=https://bookdb.up.railway.app \\"
        echo "    -e BOOKDB_API_KEY=eyJ... \\"
        echo "    bookdb -- ${binary_path}"
        echo ""
    fi

    # ── Cursor ───────────────────────────────────────────────────────────
    local cursor_config_dir
    case "$(uname -s)" in
        Darwin) cursor_config_dir="$HOME/.cursor" ;;
        *)      cursor_config_dir="$HOME/.cursor" ;;
    esac
    if [ -d "$cursor_config_dir" ]; then
        found_any=true
        success "[Cursor] detected"
        echo ""
        echo "Add to your project's .cursor/mcp.json:"
        echo ""
        print_json "{
  \"mcpServers\": {
    \"bookdb\": {
      \"command\": \"${binary_path}\",
      \"env\": {
        \"BOOKDB_API_URL\": \"https://bookdb.up.railway.app\",
        \"BOOKDB_API_KEY\": \"\"
      }
    }
  }
}"
        echo ""
    fi

    # ── OpenCode ─────────────────────────────────────────────────────────
    if command -v opencode &>/dev/null; then
        found_any=true
        success "[OpenCode] detected"
        echo ""
        echo "Add to ~/.config/opencode/opencode.json:"
        echo ""
        print_json "{
  \"mcp\": {
    \"bookdb\": {
      \"type\": \"local\",
      \"command\": [\"${binary_path}\"],
      \"env\": {
        \"BOOKDB_API_URL\": \"https://bookdb.up.railway.app\",
        \"BOOKDB_API_KEY\": \"\"
      },
      \"enabled\": true
    }
  }
}"
        echo ""
    fi

    # ── Codex ────────────────────────────────────────────────────────────
    if command -v codex &>/dev/null; then
        found_any=true
        success "[Codex] detected"
        echo ""
        echo "  codex mcp add bookdb -- ${binary_path}"
        echo ""
        echo "To set the API URL and key (persisted in server config):"
        echo "  codex mcp add bookdb \\"
        echo "    -e BOOKDB_API_URL=https://bookdb.up.railway.app \\"
        echo "    -e BOOKDB_API_KEY=eyJ... \\"
        echo "    -- ${binary_path}"
        echo ""
    fi

    # ── No assistant found ───────────────────────────────────────────────
    if [ "$found_any" = false ]; then
        echo "No AI coding assistants detected on this machine."
        echo ""
        echo "Binary: ${binary_path}"
        echo ""
        info "Manual configuration — add this JSON to your MCP client config:"
        echo ""
        print_json "{
  \"mcpServers\": {
    \"bookdb\": {
      \"command\": \"${binary_path}\",
      \"env\": {
        \"BOOKDB_API_URL\": \"https://bookdb.up.railway.app\",
        \"BOOKDB_API_KEY\": \"\"
      }
    }
  }
}"
        echo ""
    fi

    # ── Auth reminder ────────────────────────────────────────────────────
    info "Authentication"
    echo ""
    echo "  # Get a JWT token (personalized recommendations require this):"
    echo "  ${binary_path} login -email you@example.com"
    echo ""
    echo "  # Then set BOOKDB_API_KEY in the config above with the printed token."
    echo ""

    echo "Binary: ${binary_path}"
    echo "Docs:   https://github.com/${REPO}#mcp-server"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

IS_UPDATE=false

main() {
    local target
    target="$(detect_platform)"

    local existing_binary="${INSTALL_DIR}/${BINARY_NAME}"
    if [ -x "$existing_binary" ]; then
        IS_UPDATE=true
        info "Updating BookDB MCP Server..."
    else
        info "Installing BookDB MCP Server..."
    fi
    echo ""

    info "Detected platform: ${target}"

    local tag
    tag="$(get_latest_release_tag "$target")"
    info "Latest release: ${tag}"

    download_binary "$target" "$tag"

    if [ "$IS_UPDATE" = true ]; then
        echo ""
        success "BookDB MCP Server updated to ${tag}!"
        echo ""
    else
        check_path
    fi
    print_setup_instructions
}

main
