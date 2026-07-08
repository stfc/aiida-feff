#!/usr/bin/env bash
# post-create.sh — run once after the devcontainer is created
set -euo pipefail

cd /workspace

# ── 0. Configure Git and Workspace Permissions ──────────────────────────────
# Configure git to trust the workspace and ignore file mode changes.
# This prevents Podman Desktop on MacOS from showing thousands of mode changes
# (100644 => 100755) due to virtualization mount differences.
git config --global --add safe.directory /workspace
git config --global core.filemode false
git config core.filemode false || true

# Fix workspace ownership ONLY if the workspace is not currently writable.
# Frequently, Podman Desktop on MacOS automatically maps the host user, and
# running chown recursively modifies host file metadata/permissions unnecessarily.
if [ ! -w /workspace ]; then
  echo "Workspace of /workspace is not writable. Attempting to fix ownership..."
  sudo chown "$(id -u):$(id -g)" /workspace
  if [ -d /workspace/.git ]; then
    sudo find /workspace -mindepth 1 -path /workspace/.git -prune -o -exec chown "$(id -u):$(id -g)" {} +
  else
    sudo chown -R "$(id -u):$(id -g)" /workspace
  fi
fi

# ── 0b. Remove any stale .venv left by a different UID (e.g. host bind-mount) ─
# uv cannot modify a venv it doesn't own; safer to recreate it.
if [ -d /workspace/.venv ]; then
  VENV_OWNER="$(stat -c '%u' /workspace/.venv)"
  if [ "$VENV_OWNER" != "$(id -u)" ]; then
    sudo rm -rf /workspace/.venv
  fi
fi

# ── 0c. Add x86_64 glibc for QEMU-emulated FEFF binaries (ARM64 hosts) ──────
# larch ships only x86_64 FEFF binaries. On aarch64 they run via
# qemu-x86_64-static, which is present in the base image, but requires
# the x86_64 dynamic linker and glibc to be installed as a foreign arch.
if [ "$(uname -m)" = "aarch64" ] && ! dpkg -l libc6:amd64 &>/dev/null; then
  sudo dpkg --add-architecture amd64
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends libc6:amd64
fi

# ── 1. Install uv ────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  curl -Lsf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Avoid hardlink warnings on container/bind-mount filesystems.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# ── 2. Install the package and all optional deps ─────────────────────────────
uv sync --extra testing

# Refresh command hash so newly installed console scripts (e.g. verdi)
# are resolvable in this shell session.
hash -r 2>/dev/null || true

# ── 2b. Prepare persistent AiiDA repository volume ──────────────────────────
mkdir -p /tmp/aiida-feff-repository
sudo chown -R "$(id -u):$(id -g)" /tmp/aiida-feff-repository

# ── 3. Set up the AiiDA profile ──────────────────────────────────────────────
if uv run verdi profile show default &>/dev/null; then
  uv run verdi profile set-default default
else
  # A failed first-time setup can leave the development database partially
  # initialised even though no profile was written to config. Reset the local
  # dev state before retrying profile creation.
  if uv run python - <<'PY'
import json
from pathlib import Path

config_path = Path('/home/vscode/.aiida/config.json')
if not config_path.exists():
    raise SystemExit(1)

data = json.loads(config_path.read_text())
raise SystemExit(0 if not data.get('profiles') else 1)
PY
  then
    rm -rf /tmp/aiida-feff-repository/*
    uv run python - <<'PY'
import os
import psycopg

conn = psycopg.connect(
    host=os.environ.get('AIIDA_DB_HOST', 'localhost'),
    port=int(os.environ.get('AIIDA_DB_PORT', '5432')),
    dbname=os.environ.get('AIIDA_DB_NAME', 'aiida'),
    user=os.environ.get('AIIDA_DB_USER', 'aiida'),
    password=os.environ.get('AIIDA_DB_PASS', 'aiida'),
    autocommit=True,
)
with conn, conn.cursor() as cur:
    cur.execute('DROP SCHEMA IF EXISTS public CASCADE')
    cur.execute('CREATE SCHEMA public')
PY
  fi

  uv run verdi profile setup core.psql_dos \
    --profile-name default \
    --non-interactive \
    --set-as-default \
    --database-hostname "${AIIDA_DB_HOST:-localhost}" \
    --database-port "${AIIDA_DB_PORT:-5432}" \
    --database-name "${AIIDA_DB_NAME:-aiida}" \
    --database-username "${AIIDA_DB_USER:-aiida}" \
    --database-password "${AIIDA_DB_PASS:-aiida}" \
    --use-rabbitmq \
    --email "dev@local" \
    --first-name Dev \
    --last-name User \
    --institution Local \
    --repository-uri "file:///tmp/aiida-feff-repository"
fi

# ── 4. Set up a localhost computer for running calculations ──────────────────
if ! uv run verdi computer show localhost &>/dev/null; then
  uv run verdi computer setup \
    --label localhost \
    --hostname localhost \
    --transport core.local \
    --scheduler core.direct \
    --work-dir /tmp/aiida-feff-runs \
    --mpirun-command "" \
    --non-interactive
  uv run verdi computer configure core.local localhost --non-interactive --safe-interval 0
fi

# ── 5. Locate the FEFF8L binary that xraylarch ships ────────────────────────
FEFF_EXE=$(uv run python -c "
import sys, os
sp = next(p for p in sys.path if 'site-packages' in p)
print(os.path.join(sp, 'larch', 'bin', 'linux64', 'feff8l.sh'))
")

if [ ! -f "$FEFF_EXE" ]; then
  echo "ERROR: could not find feff8l.sh inside the xraylarch package." >&2
  echo "       Make sure 'uv sync' succeeded." >&2
  exit 1
fi

# Fix shebang: feff8l.sh uses ${BASH_SOURCE[0]} but ships with #!/bin/sh
sed -i 's|^#!/bin/sh|#!/bin/bash|' "$FEFF_EXE"

# Make every binary in that directory executable.
# Must run AFTER sed -i because sed -i rewrites the file and can strip the +x bit.
chmod +x "$(dirname "$FEFF_EXE")"/feff8l*

# ── 6. Register feff8l as the 'feff' code in AiiDA ───────────────────────────
if ! uv run verdi code show feff@localhost &>/dev/null 2>&1; then
  uv run verdi code create core.code.installed \
    --non-interactive \
    --label feff \
    --computer localhost \
    --filepath-executable "$FEFF_EXE" \
    --description "FEFF8L from xraylarch"
fi

# ── 7. Register venv python3 for path aggregation ───────────────────────────
PYTHON3_EXE="/workspace/.venv/bin/python3"
if [ ! -x "$PYTHON3_EXE" ]; then
  echo "ERROR: expected venv python at $PYTHON3_EXE" >&2
  exit 1
fi

if ! uv run verdi code show python3@localhost &>/dev/null 2>&1; then
  uv run verdi code create core.code.installed \
    --non-interactive \
    --label python3 \
    --computer localhost \
    --filepath-executable "$PYTHON3_EXE" \
    --description "Python 3 (venv) for FEFF path aggregation"
fi

# ── 8. Start the AiiDA daemon ────────────────────────────────────────────────
uv run verdi daemon start 2

echo ""
echo "✓ aiida-feff devcontainer ready."
echo "  FEFF binary : $FEFF_EXE"
echo "  Python code : $PYTHON3_EXE (python3@localhost)"
echo "  Run tests   : uv run pytest tests/ -v"
echo "  verdi shell : uv run verdi shell"
echo "  Daemon      : uv run verdi daemon status"
