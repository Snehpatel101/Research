# SSE Bridge Setup for Claude Code + Serena

## Overview

This project uses Serena MCP server via an SSE (Server-Sent Events) bridge for Claude Code integration. This enables semantic code analysis, symbol-based editing, and intelligent code navigation.

## Architecture

```
Claude Code ←→ MCP Proxy (SSE) ←→ Serena MCP Server ←→ pylsp (Python LSP)
                (port 8808)         (semantic tools)    (code intelligence)
```

## Prerequisites

1. **mcp-proxy** installed:
   ```bash
   uv tool install mcp-proxy
   ```

2. **Serena** installed:
   ```bash
   uv tool install serena
   # OR
   uvx --from git+https://github.com/oraios/serena serena --version
   ```

3. **MCP proxy config** at `~/.config/mcp-proxy/servers.json`:
   ```json
   {
     "mcpServers": {
       "serena": {
         "command": "uvx",
         "args": [
           "--from", "git+https://github.com/oraios/serena",
           "serena", "start-mcp-server",
           "--context", "claude-code",
           "--mode", "editing",
           "--mode", "interactive",
           "--project", "/Users/sneh/research"
         ]
       }
     }
   }
   ```

## Starting the SSE Bridge

### Option 1: Using Make commands

```bash
# Start MCP proxy (background)
make mcp-start

# Check status
make mcp-status

# Stop
make mcp-stop

# Debug mode (foreground with logs)
make mcp-logs
```

### Option 2: Using dev script directly

```bash
# Start
./dev/start-mcp-proxy.sh start

# Stop
./dev/start-mcp-proxy.sh stop

# Status
./dev/start-mcp-proxy.sh status
```

## Claude Code Configuration

### Automatic Setup

```bash
make mcp-setup
# OR
python3 dev/configure-mcp-sse.py
```

This updates `~/.claude/settings.local.json` with SSE endpoints.

### Manual Configuration

Add to `~/.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "serena": {
      "url": "http://127.0.0.1:8808/servers/serena/sse",
      "transport": "sse"
    }
  }
}
```

## Serena Modes for Claude Code

### Editing Mode (Default)
- Full access to symbolic editing tools
- `replace_symbol_body`, `insert_after_symbol`, etc.
- Recommended for implementation tasks

### Planning Mode
- Focused on analysis and strategy
- Read-only by default
- Use for architecture decisions

### Interactive Mode
- Engages with user throughout task
- Asks for clarification when unclear
- Default behavior

## Available Serena Tools (via SSE)

| Tool | Purpose |
|------|---------|
| `find_symbol` | Search for symbols by name |
| `get_symbols_overview` | Get file structure |
| `replace_symbol_body` | Replace entire symbol |
| `insert_after_symbol` | Insert code after symbol |
| `insert_before_symbol` | Insert code before symbol |
| `find_referencing_symbols` | Find symbol references |
| `search_for_pattern` | Regex search in codebase |
| `list_dir` | List directory contents |
| `read_memory` / `write_memory` | Persistent knowledge |
| `think_about_*` | Reflection tools |

## Troubleshooting

### MCP Proxy Not Starting
```bash
# Check if port is in use
lsof -i :8808

# Verify mcp-proxy installation
which mcp-proxy
mcp-proxy --version

# Check config file
cat ~/.config/mcp-proxy/servers.json
```

### Serena Not Responding
```bash
# Test Serena directly
uvx --from git+https://github.com/oraios/serena \
  serena start-mcp-server \
  --context claude-code \
  --project /Users/sneh/research

# Check for errors in logs
make mcp-logs
```

### Symbol Index Stale
```bash
# Restart language server (via Serena tool)
# Use: restart_language_server tool

# Or manually clear cache
rm -rf .serena/cache/
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MCP_PORT` | 8808 | SSE server port |
| `MCP_HOST` | 127.0.0.1 | SSE server host |
| `MCP_LOG_LEVEL` | INFO | Logging verbosity |
| `MCP_PROXY_BIN` | ~/.local/bin/mcp-proxy | Proxy binary path |
| `MCP_CONFIG` | ~/.config/mcp-proxy/servers.json | Server config |

## Performance Tips

1. **Pre-index project** before starting work:
   ```bash
   uvx --from git+https://github.com/oraios/serena serena project index
   ```

2. **Use ignored_paths** in project.yml to exclude large directories

3. **Restart proxy** if switching between projects

4. **Check memory usage** if LSP becomes slow:
   ```bash
   ps aux | grep pylsp
   ```
