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
# Both routes end at merge_exafs_shards, so the archive exists either way.
grep -q "archive pk=" serial.log || fail "serial route produced no archive"
grep -q "ensemble=True" serial.log || fail "serial archive is not an ensemble archive"
grep -q "merged from 1 shard(s)" serial.log || fail "serial route wrote no shard"
# The Debye-Waller step reads md-exafs' grouped-MSRD dicts by key, and a
# renamed key there surfaces as a KeyError three quarters of the way through
# a run that has already spent minutes in FEFF. Assert the line it prints.
grep -qE "Fe-Fe +reff=2\.[0-9]+ " serial.log || fail "no Fe-Fe Debye-Waller group"
# A cutoff past the cell's inscribed sphere aliases neighbours and biases
# sigma^2 low. md-exafs warns rather than raising, so the warning has to fail
# the build or the example silently reports wrong physics.
! grep -q "exceeds the maximum safe MIC cutoff" serial.log \
  || fail "Debye-Waller cutoff exceeds the minimum-image limit"
grep -q "2 snapshot spectra for the overlay" serial.log || fail "serial overlay lost snapshots"

run "example_ensemble_synthetic.py — batch mode"
uv run --project "$REPO" python "$REPO/examples/example_ensemble_synthetic.py" \
  --code feff@localhost --python-code python3@localhost \
  --n-snapshots 3 --sigma 0.06 --batch-size 2 \
  --plot-file "$WORK/batch.png" 2>&1 | tee batch.log
grep -q "batch mode: chunks of 2" batch.log || fail "batch mode was not engaged"
grep -q "archive pk=" batch.log || fail "batch mode produced no consolidated archive"
grep -q "ensemble=True" batch.log || fail "the archive is not an ensemble archive"
# 3 snapshots in chunks of 2 is 2 scheduler jobs, which is the point of batching.
grep -q "merged from 2 shard(s)" batch.log || fail "expected 2 batch shards"
# Every k must be backed by all 3 snapshots: a resampling regression that drops
# the top of the grid shows up here as a min below the max.
grep -q "contributors per k: min=3 max=3" batch.log || fail "ragged ensemble coverage"
# FeffBatchCalculation exposes xas_data as a namespace rather than one node,
# so the overlay has to flatten it. Getting that wrong drops every snapshot
# curve from the plot, or crashes in the plotting helper.
grep -q "3 snapshot spectra for the overlay" batch.log || fail "batch overlay lost snapshots"
[ -s "$WORK/batch.png" ] || fail "batch mode wrote no plot"

run "example_ensemble_synthetic.py — precomputed potentials"
uv run --project "$REPO" python "$REPO/examples/example_ensemble_synthetic.py" \
  --code feff@localhost --n-snapshots 2 --sigma 0.06 \
  --precompute-potentials --plot-file "$WORK/pot.png" 2>&1 | tee pot.log
grep -q "precomputing scattering potentials" pot.log || fail "potentials step was not engaged"

run "example_ensemble.py — against a stored trajectory"
TRAJ_PK=$(REPO="$REPO" uv run --project "$REPO" python - <<'PY'
import os
import sys

from aiida import load_profile

# Through the environment, not interpolated: the heredoc is quoted so that the
# Python below cannot be mangled by the shell, and $REPO is overridable.
sys.path.insert(0, os.path.join(os.environ["REPO"], "examples"))
load_profile()
from example_ensemble_synthetic import make_trajectory

traj = make_trajectory(2, 0.06)
traj.store()
print(traj.pk)
PY
)
uv run --project "$REPO" python "$REPO/examples/example_ensemble.py" \
  --code feff@localhost --trajectory-pk "$TRAJ_PK" 2>&1 | tee ensemble.log
WC_PK=$(sed -n 's/.*Submitted EnsembleExafsWorkChain pk=\([0-9]*\).*/\1/p' ensemble.log)
[ -n "$WC_PK" ] || fail "example_ensemble.py submitted nothing"

# This example calls submit() rather than run(), so it returns as soon as the
# workchain is handed to the daemon and never reports an outcome itself. That
# is the point of it, and the only example that exercises the daemon and the
# broker at all, so the waiting belongs here rather than in the example.
uv run --project "$REPO" python - "$WC_PK" <<'PY' || fail "submitted workchain did not finish cleanly"
import sys
import time

from aiida import load_profile, orm

load_profile()
pk = int(sys.argv[1])
node = orm.load_node(pk)
deadline = time.time() + 900
while not node.is_terminated and time.time() < deadline:
    time.sleep(5)
    node = orm.load_node(pk)

if not node.is_terminated:
    print(f"workchain {pk} still {node.process_state} after 15 minutes")
    sys.exit(1)
if not node.is_finished_ok:
    print(f"workchain {pk} finished with exit status {node.exit_status}: {node.exit_message}")
    sys.exit(1)
# An archive proves the run got all the way through the merge, not merely that
# the daemon picked it up and stopped somewhere.
if "archive" not in node.outputs:
    print(f"workchain {pk} finished without an archive output")
    sys.exit(1)
print(f"workchain {pk} finished OK with an archive")
PY

echo
echo "✓ all examples ran and demonstrated what they claim to."
