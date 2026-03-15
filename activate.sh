#!/usr/bin/env bash
# Activate the project virtualenv and load .env variables into the current shell.
# Usage: source activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtualenv
source "$SCRIPT_DIR/.venv/bin/activate"

# Export variables from .env (skip comments and blank lines)
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        export "$line"
    done < "$SCRIPT_DIR/.env"
    echo "Loaded .env"
fi

echo "Activated .venv (Python: $(python --version))"
