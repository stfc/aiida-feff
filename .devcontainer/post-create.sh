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

# ── 0a. Take ownership of the named volumes ─────────────────────────────────
# A named volume whose mount point does not already exist in the image is
# created owned by root, so the vscode user cannot write to it. /home/vscode
# exists but /home/vscode/.aiida does not, so both AiiDA volumes land
# root-owned and the first `verdi profile setup` fails with EACCES.
#
# Empty-directory chown only; this never touches existing data.
for volume_path in /home/vscode/.aiida /home/vscode/.aiida/repository; do
  if [ ! -w "$volume_path" ]; then
    sudo chown "$(id -u):$(id -g)" "$volume_path"
  fi
done

# ── 0b. x86_64 loader for the FEFF binaries on arm64 hosts ──────────────────
# larch ships FEFF8L only as x86_64. On Apple Silicon these run through the
# qemu-x86_64 binfmt handler that the container runtime registers on the host.
# That handler is registered with the F ("fix binary") flag, so it is *not*
# visible as an entry in the container's own /proc/sys/fs/binfmt_misc -- do
# not conclude from an empty listing there that emulation is unavailable.
#
# What the container does have to supply is the x86_64 dynamic loader and
# glibc. Without them the handler fires and then fails with
#   qemu-x86_64-static: Could not open '/lib64/ld-linux-x86-64.so.2'
# which reads like a missing binary rather than a missing dependency.
if [ "$(uname -m)" = "aarch64" ] && ! dpkg -l libc6:amd64 &>/dev/null; then
  echo "arm64 host: installing the x86_64 loader so FEFF8L can run …"
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

# Verify FEFF actually executes, rather than discovering it does not inside a
# queued calculation. With the amd64 pin this should always succeed; if it
# does not, the binaries or their glibc dependencies are the problem.
if ! "$(dirname "$FEFF_SRC")/feff8l_rdinp" &>/dev/null; then
  echo "WARNING: the bundled FEFF8L binaries did not execute." >&2
  echo "         Check 'ldd $(dirname "$FEFF_SRC")/feff8l_rdinp'." >&2
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
# Matches UV_PROJECT_ENVIRONMENT in docker-compose.yml.
PYTHON3_EXE="${UV_PROJECT_ENVIRONMENT:-/home/vscode/.venv}/bin/python3"
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

# ── 7b. Silence the RabbitMQ version warning ────────────────────────────────
# aiida-core warns about any broker >= 3.8.15 because of the 30-minute default
# consumer timeout. rabbitmq.conf raises that timeout to ~115 days, so the
# condition the warning describes does not apply here. Suppressed only
# because it has actually been addressed; if that mount is ever removed, this
# line has to go with it.
uv run verdi config set warnings.rabbitmq_version False >/dev/null

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
