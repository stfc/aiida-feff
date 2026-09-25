"""AiiDA plugin for FEFF: provenance-tracked EXAFS calculations."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the installed distribution metadata, which is
    # what ``versions.dependency_versions`` stamps into node attributes.
    # A hardcoded literal here drifted from pyproject.toml (0.1.0 vs 0.1.0a1)
    # and the two disagreed in stored provenance.
    __version__ = version("aiida-feff")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "unknown"

__all__ = ["__version__"]
