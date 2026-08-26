"""FastMCP tool server - the execution surface J.A.R.V.I.S. agents call into.

Runs as a stdio MCP server (for Ollama/CrewAI tool bindings) and doubles as
the contract n8n workflows invoke over HTTP via the backend /api/command and
/api/speak endpoints.

Security model: commands are matched against an allow-list of prefixes and
always run with a timeout; every invocation is appended to an audit log.

Run:  python tools/mcp_server.py            (stdio transport, default)
      JARVIS_MCP_HTTP=1 python tools/mcp_server.py   (streamable-http :9110)
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "FastMCP is not installed. Run: pip install -r requirements-full.txt"
    ) from exc

mcp = FastMCP("jarvis-tools")

ALLOWED_PREFIXES = (
    "say ", "uptime", "df ", "df", "whoami", "date", "uname",
    "sw_vers", "sysctl ", "pmset -g", "networksetup -listallhardwareports",
    "ls ", "cat ", "head ", "tail ", "grep ", "python --version",
    "python3 --version", "node --version", "git status", "git log",
    "ollama list", "ollama ps",
)
AUDIT_LOG = Path(os.environ.get("JARVIS_MCP_AUDIT", "/tmp/jarvis_mcp_audit.log"))
COMMAND_TIMEOUT_S = float(os.environ.get("JARVIS_MCP_TIMEOUT", "10"))


def _audit(line: str) -> None:
    with AUDIT_LOG.open("a") as fh:
        fh.write(f"{time.time():.0f} {line}\n")


@mcp.tool()
def exec_command(command: str) -> str:
    """Run an allow-listed shell command and return its output (read-only ops)."""
    command = command.strip()
    if not any(command == p.strip() or command.startswith(p) for p in ALLOWED_PREFIXES):
        _audit(f"DENIED {command!r}")
        return f"DENIED: command not in allow-list. Allowed prefixes: {ALLOWED_PREFIXES}"
    _audit(f"ALLOWED {command!r}")
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {COMMAND_TIMEOUT_S}s"
    output = (proc.stdout or proc.stderr or "").strip()
    return f"exit={proc.returncode}\n{output[:4000]}"


@mcp.tool()
def run_script(path: str) -> str:
    """Execute a script stored in the JARVIS_SCRIPTS_DIR sandbox."""
    root = Path(os.environ.get("JARVIS_SCRIPTS_DIR", "scripts")).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
        return f"DENIED: {path!r} is not a file inside {root}"
    _audit(f"SCRIPT {path!r}")
    proc = subprocess.run(
        ["bash", str(target)], capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S
    )
    return f"exit={proc.returncode}\n{(proc.stdout or proc.stderr or '')[:4000]}"


@mcp.tool()
def speak(text: str) -> str:
    """Make J.A.R.V.I.S. say something on the holographic interface."""
    import urllib.request

    base = os.environ.get("JARVIS_API", "http://127.0.0.1:8000")
    req = urllib.request.Request(
        f"{base}/api/speak",
        data=f'{{"text": {text!r}}}'.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode()


if __name__ == "__main__":
    if os.environ.get("JARVIS_MCP_HTTP"):
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
