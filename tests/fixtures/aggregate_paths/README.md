# Golden FEFF fixtures

Real FEFF output, used by `tests/test_aggregate_paths.py` and `tests/test_exafs.py`
to pin the output-format assumptions and the EXAFS equation against data no test
helper produced.

## Provenance

| Field | Value |
|---|---|
| Code | **Feff8L (EXAFS) 0.1**, as bundled by xraylarch |
| System | SrTiO₃ |
| Absorber | Ti, K edge |
| `RPATH` | 8.0 Å |
| `PWCRIT` | 3.00 % |

Read from the file headers; there is no run script and no input deck alongside
them.

## Contents

- `files.dat` — the full amplitude ranking table, ~700 paths.
- `feff0001.dat` … `feff0011.dat` — 9 of those paths. The rest were not kept,
  so `files.dat` lists many paths with no corresponding `.dat` file. That
  asymmetry is deliberate: it exercises the "listed but absent" branch of
  `_aggregate_paths.py`.

## Regenerating

There is no regeneration script, and the original SrTiO₃ run is not recorded
beyond the headers above. **Treat these files as irreplaceable.** Adding
fixtures for another FEFF version is fine; replacing these is not, because the
FEFF8L column layout they encode is what `_parse_files_dat` and
`_parse_with_text` are written against.

If you do add fixtures from a different FEFF, put them in a sibling directory
named for that version and say so in the test that consumes them.
