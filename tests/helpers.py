"""Shared parsing helpers for assertions about generated ``feff.inp`` files.

Kept out of ``conftest.py`` because these are plain functions rather than
fixtures, and both ``test_calculations.py`` and ``test_batch.py`` need them.

Parsing the blocks properly matters: substring checks such as ``"  0 " in
text`` match whitespace anywhere in the file, including the ATOMS coordinate
columns, so they cannot tell a present POTENTIALS entry from an absent one.
"""

from __future__ import annotations


def is_float(token: str) -> bool:
    """True if ``token`` parses as a float."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def atoms_rows(text: str) -> list[dict]:
    """Parse the ATOMS block: ``x y z ipot tag distance number``.

    Returns an empty list when there is no ATOMS block, so callers must
    assert on the contents rather than iterating and hoping -- a loop over
    nothing silently asserts nothing.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith("ATOMS"))
    except StopIteration:
        return []

    rows = []
    for line in lines[start + 1 :]:
        if line.strip().startswith("END"):
            break
        parts = line.split()
        if len(parts) < 7 or not is_float(parts[0]):
            continue  # header or separator row
        rows.append(
            {
                "x": float(parts[0]),
                "y": float(parts[1]),
                "z": float(parts[2]),
                "ipot": int(parts[3]),
                "tag": parts[4],
                "distance": float(parts[5]),
            }
        )
    return rows


def potentials_rows(text: str) -> list[dict]:
    """Parse the POTENTIALS block: ``ipot Z tag lmax1 lmax2 xnatph spinph``."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith("POTENTIALS"))
    except StopIteration:
        return []

    rows = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if not parts:
            break
        if len(parts) < 3 or not parts[0].lstrip("-").isdigit():
            continue  # header or separator row
        rows.append({"ipot": int(parts[0]), "z": int(parts[1]), "tag": parts[2]})
    return rows


def origin_atoms(text: str) -> list[dict]:
    """ATOMS rows sitting at the coordinate origin, i.e. candidate absorbers."""
    return [
        row
        for row in atoms_rows(text)
        if abs(row["x"]) < 1e-6 and abs(row["y"]) < 1e-6 and abs(row["z"]) < 1e-6
    ]
