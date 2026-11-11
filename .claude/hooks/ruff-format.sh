#!/usr/bin/env bash
# PostToolUse hook: format + autofix a just-edited Python file with ruff.
set -u
file=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)
[[ "$file" == *.py ]] || exit 0
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
ruff="$root/.venv/bin/ruff"
[[ -x "$ruff" && -f "$file" ]] || exit 0
"$ruff" format --quiet "$file" >/dev/null 2>&1
"$ruff" check --fix --quiet "$file" >/dev/null 2>&1
exit 0
