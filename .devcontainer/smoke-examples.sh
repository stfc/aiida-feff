#!/usr/bin/env bash
# Run every example against the devcontainer's registered codes.
#
# examples/ is not imported by the test suite and not type-checked, so nothing
# else notices when it drifts from the API. Three separate breakages reached
# main that way: an import of the deleted calcfunctions.debye_waller, a call to
# the removed store_msrd, and a "title" key that FeffParameters rejects. Each
# would have failed here in seconds.
#
# The checks below assert on output rather than only on exit status. An
# example that runs to completion while silently skipping the feature it is
# meant to demonstrate is the failure mode worth catching -- the serial route
# producing no archive, say, or batch mode quietly falling back to serial.
#
# Requires: feff@localhost and python3@localhost, i.e. post-create.sh has run.
# Memory: the ensemble runs need roughly 4 GB. Below that the daemon is killed
# mid-run and the symptom is exit 137 with no message.
set -euo pipefail

REPO=${REPO:-/workspace}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"   # FEFF writes log.dat and friends into the working directory

run() { echo; echo "── $* ──"; }
fail() { echo "SMOKE FAILURE: $*" >&2; exit 1; }

# Every example must at least import and expose its CLI. This catches a broken
# import without waiting for a FEFF run.
run "--help on every example"
for example in "$REPO"/examples/*.py; do
  uv run --project "$REPO" python "$example" --help >/dev/null \
    || fail "$(basename "$example") --help failed"
  echo "  ok: $(basename "$example")"
done

run "example_single_site.py"
uv run --project "$REPO" python "$REPO/examples/example_single_site.py" \
  --code feff@localhost 2>&1 | tee single.log
grep -q "exit_status=0" single.log || fail "single-site calculation did not finish cleanly"

run "example_ensemble_synthetic.py — serial, with path contributions"
uv run --project "$REPO" python "$REPO/examples/example_ensemble_synthetic.py" \
  --code feff@localhost --python-code python3@localhost \
  --n-snapshots 2 --sigma 0.06 --store-paths \
  --plot-file "$WORK/serial.png" 2>&1 | tee serial.log
grep -q "path_contributions pk=" serial.log || fail "--store-paths produced no path_contributions"
[ -s "$WORK/serial.png" ] || fail "no plot was written"
# The serial route writes no batch shards, so it must report no archive rather
# than leaving the output silently absent.
grep -q "archive pk=" serial.log && fail "serial route reported an archive"

run "example_ensemble_synthetic.py — batch mode"
uv run --project "$REPO" python "$REPO/examples/example_ensemble_synthetic.py" \
  --code feff@localhost --python-code python3@localhost \
  --n-snapshots 3 --sigma 0.06 --batch-size 2 \
  --plot-file "$WORK/batch.png" 2>&1 | tee batch.log
grep -q "batch mode: chunks of 2" batch.log || fail "batch mode was not engaged"
grep -q "archive pk=" batch.log || fail "batch mode produced no consolidated archive"
grep -q "ensemble=True" batch.log || fail "the archive is not an ensemble archive"
# 3 snapshots in chunks of 2 is 2 scheduler jobs, which is the point of batching.
grep -q "merged from 2 batch shard(s)" batch.log || fail "expected 2 batch shards"

run "example_ensemble_synthetic.py — precomputed potentials"
uv run --project "$REPO" python "$REPO/examples/example_ensemble_synthetic.py" \
  --code feff@localhost --n-snapshots 2 --sigma 0.06 \
  --precompute-potentials --plot-file "$WORK/pot.png" 2>&1 | tee pot.log
grep -q "precomputing scattering potentials" pot.log || fail "potentials step was not engaged"

run "example_ensemble.py — against a stored trajectory"
TRAJ_PK=$(uv run --project "$REPO" python - <<'PY'
import sys

from aiida import load_profile

sys.path.insert(0, "/workspace/examples")
load_profile()
from example_ensemble_synthetic import make_trajectory

traj = make_trajectory(2, 0.06)
traj.store()
print(traj.pk)
PY
)
uv run --project "$REPO" python "$REPO/examples/example_ensemble.py" \
  --code feff@localhost --trajectory-pk "$TRAJ_PK" 2>&1 | tee ensemble.log
grep -qi "finished" ensemble.log || fail "example_ensemble.py did not report a finished workchain"

echo
echo "✓ all examples ran and demonstrated what they claim to."
