
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../backend"
python -m pip install -U pip
pip install -e .
uvicorn app.main:app --reload --port 8000
