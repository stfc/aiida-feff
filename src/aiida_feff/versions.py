"""Version provenance helpers.

Every spectrum this plugin produces depends on three external codes that are
not pinned by the AiiDA graph: FEFF itself, larch (background subtraction and
``feff????.dat`` parsing) and pymatgen (``feff.inp`` generation).  A change in
any of them changes the physics silently.

The helpers here collect those versions so calcjobs and parsers can stamp them
into node **attributes**, making "which codes produced this spectrum?" a
question the provenance graph can answer.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Attribute key used on every node that records code versions.
VERSIONS_ATTR = "code_versions"

# Attribute key used on XasData / calcjob nodes for the FEFF banner.
FEFF_VERSION_ATTR = "feff_version"

_PACKAGES = ("aiida-feff", "xraylarch", "pymatgen", "numpy", "ase", "h5py")

# FEFF writes its banner into log.dat / files.dat, e.g.
#   " Feff 8.50L "        (FEFF8L via xraylarch)
#   " Feff8L (EXAFS)  0.1"
#   " FEFF 9.6.4 "
_FEFF_BANNER = re.compile(r"\bfeff\s*([0-9]+(?:\.[0-9]+)*[a-zA-Z]*)", re.IGNORECASE)


def dependency_versions() -> dict[str, str]:
    """Return installed versions of the packages that shape the physics.

    Missing packages are omitted rather than recorded as ``"unknown"``, so an
    absent key means "not installed in the environment that produced this
    node".
    """
    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            continue
    return out


def parse_feff_version(text: str) -> str | None:
    """Extract the FEFF version banner from ``log.dat`` / ``files.dat`` text.

    FEFF prints its identity in the first few lines of every output file.  Only
    the header is scanned so a stray "feff0001.dat" further down cannot be
    mistaken for a version string.

    The returned string is normalised to a leading ``Feff`` regardless of
    how the banner capitalises it, so it can be compared across versions.

    Returns ``None`` when no banner is recognised.
    """
    header_lines = 20
    for line in text.splitlines()[:header_lines]:
        match = _FEFF_BANNER.search(line)
        if match:
            return f"Feff{match.group(1)}"
    return None
