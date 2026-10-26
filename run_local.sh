
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# Install requirements.txt into an activated Python 3.12 environment first.
# Keep local state outside the checkout by default; preserve explicit overrides.
state_dir="${XDG_STATE_HOME:-${HOME}/.local/state}/raphael"
mkdir -p "$state_dir"
export DATABASE_URL="${DATABASE_URL:-sqlite:///$state_dir/raphael.db}"
exec python -m uvicorn main:app --reload --host 127.0.0.1 --port "${PORT:-8000}"
