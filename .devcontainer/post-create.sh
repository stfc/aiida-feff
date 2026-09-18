#!/usr/bin/env bash
# post-create.sh — run once after the devcontainer is created.
#
# Idempotent: every step checks before acting, so re-running is safe and a
# rebuild never destroys stored data. An earlier version dropped the Postgres
# schema and emptied the repository directory whenever ~/.aiida/config.json
# was missing -- which was exactly the state a rebuild produced, because the
# config was not on a named volume while the data volumes were. That reset
# step is gone along with Postgres itself.
set -euo pipefail

cd /workspace

# ── 0. Git configuration ────────────────────────────────────────────────────
# Trust the workspace and ignore file-mode changes: rootless Podman on macOS
# otherwise reports thousands of spurious 100644 => 100755 diffs.
git config --global --add safe.directory /workspace
git config --global core.filemode false
git config core.filemode false || true

# NB: there is deliberately no chown of /workspace here. The previous version
# chowned everything *except* .git, so in the one situation it triggered --
# the workspace being unwritable due to a UID mismatch -- .git stayed
# unwritable and every subsequent git operation failed. Ownership is now
# handled by keeping the venv out of the bind mount (see docker-compose.yml)
# and by userns_mode for Podman users (docker-compose.podman.yml).

# ── 0b. x86_64 glibc for QEMU-emulated FEFF binaries on ARM64 hosts ─────────
# larch ships only x86_64 FEFF binaries. On aarch64 they run through
# qemu-x86_64-static. Note that the emulation is provided by the container
# runtime's binfmt_misc registration on the host, not by this image, so it can
# be absent even when the packages below are installed; step 5 verifies that
# the binary actually executes rather than assuming it.
if [ "$(uname -m)" = "aarch64" ] && ! dpkg -l libc6:amd64 &>/dev/null; then
  sudo dpkg --add-architecture amd64
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends libc6:amd64
fi

# ── 1. Install uv (version-pinned) ──────────────────────────────────────────
# Pinned rather than `curl https://astral.sh/uv/install.sh | sh`, so that two
# developers building the same commit months apart get the same uv, and so the
# toolchain is not whatever the vendor is serving at container-creation time.
UV_VERSION="0.11.21"
if ! command -v uv &>/dev/null || [ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── 2. Install the package and the extras CI uses ───────────────────────────
# --extra plots: without it tests/test_visualise.py silently *skips* here via
#   importorskip while running in CI -- green locally, red on push.
# --extra pre-commit: AGENTS.md makes a clean local `pre-commit` the contract
#   for a clean CI lint job, so the tool has to exist in the container.
# --locked: fail rather than silently re-resolving and rewriting uv.lock.
uv sync --locked --extra testing --extra plots --extra pre-commit

# Install the git hooks so the contract above is actually enforced locally.
uv run pre-commit install

# Refresh the command hash so newly installed console scripts resolve.
hash -r 2>/dev/null || true

# ── 3. Set up the AiiDA profile ─────────────────────────────────────────────
# core.sqlite_dos, matching the test suite. With --use-rabbitmq the daemon
# still works, so examples/example_ensemble.py (which calls submit) runs.
if uv run verdi profile show default &>/dev/null; then
  uv run verdi profile set-default default
else
  uv run verdi profile setup core.sqlite_dos \
    --profile-name default \
    --non-interactive \
    --set-as-default \
    --use-rabbitmq \
    --email "dev@local" \
    --first-name Dev \
    --last-name User \
    --institution Local
fi

# ── 4. Localhost computer for running calculations ──────────────────────────
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
# Ask larch where it lives rather than guessing at a site-packages entry: the
# previous version took the first sys.path entry containing "site-packages",
# which need not be the one holding larch.
FEFF_SRC=$(uv run python -c "
import pathlib, sys
import larch
platform_dir = {'linux': 'linux64', 'darwin': 'darwin64', 'win32': 'win64'}[sys.platform]
print(pathlib.Path(larch.__file__).parent / 'bin' / platform_dir / 'feff8l.sh')
")

if [ ! -f "$FEFF_SRC" ]; then
  echo "ERROR: could not find feff8l.sh inside the xraylarch package." >&2
  echo "       Make sure 'uv sync' succeeded." >&2
  exit 1
fi

# feff8l.sh ships with #!/bin/sh but uses ${BASH_SOURCE[0]}, which is a bash
# builtin. Rather than patching the file inside .venv -- which any later
# `uv sync` silently reverts, leaving an obscure runtime failure -- register a
# wrapper this repository owns.
FEFF_EXE=/usr/local/bin/feff8l-wrapper
sudo tee "$FEFF_EXE" >/dev/null <<WRAPPER
#!/bin/bash
# Runs the xraylarch-bundled FEFF8L under bash. Generated by post-create.sh.
exec bash "$FEFF_SRC" "\$@"
WRAPPER
sudo chmod +x "$FEFF_EXE"
chmod +x "$(dirname "$FEFF_SRC")"/feff8l* || true

# Verify FEFF actually runs. On ARM64 this depends on the host runtime having
# registered qemu-x86_64 binfmt handlers, which the image cannot guarantee, so
# fail here with a clear message rather than inside a queued calculation.
if ! "$(dirname "$FEFF_SRC")/feff8l_rdinp" --version &>/dev/null \
   && ! "$(dirname "$FEFF_SRC")/feff8l_rdinp" &>/dev/null; then
  echo "WARNING: the bundled FEFF8L binaries did not execute." >&2
  echo "         On Apple Silicon / ARM64 this usually means the container" >&2
  echo "         runtime has not registered x86_64 emulation." >&2
fi

# ── 6. Register feff8l as the 'feff' code in AiiDA ──────────────────────────
if ! uv run verdi code show feff@localhost &>/dev/null 2>&1; then
  uv run verdi code create core.code.installed \
    --non-interactive \
    --label feff \
    --computer localhost \
    --filepath-executable "$FEFF_EXE" \
    --description "FEFF8L from xraylarch"
fi

# ── 7. Register the venv python3 for path aggregation ───────────────────────
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

# ── 8. Start the AiiDA daemon ───────────────────────────────────────────────
# Also started by postStartCommand in devcontainer.json, because the daemon
# does not survive a container stop/start and submitted processes would
# otherwise sit in 'created' with no obvious cause.
uv run verdi daemon start 2

echo ""
echo "✓ aiida-feff devcontainer ready."
echo "  Storage     : core.sqlite_dos (no PostgreSQL needed)"
echo "  FEFF binary : $FEFF_EXE -> $FEFF_SRC"
echo "  Python code : $PYTHON3_EXE (python3@localhost)"
echo "  Run tests   : uv run pytest tests/"
echo "  Lint as CI  : uv run pre-commit run --all-files"
echo "  verdi shell : uv run verdi shell"
echo "  Daemon      : uv run verdi daemon status"
