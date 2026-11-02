#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DATABASE_URL:?Set a dedicated synthetic SQLite database URL}"
if [ "${RAPHAEL_DEV_PORTAL:-0}" != 1 ] || [ "${RAPHAEL_DEV_SYNTHETIC_ONLY:-0}" != 1 ]; then
  echo 'Staging requires explicit synthetic portal flags.' >&2
  exit 2
fi
python -m dev_runtime migrate --database-url "$DATABASE_URL"
exec python -m uvicorn main:app --host "${RAPHAEL_BIND_HOST:-127.0.0.1}" --port "${PORT:-8000}"
