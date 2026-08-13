"""Larch-backed import of experimental XAS spectra.

The reader delegates format recognition and Athena project handling to Larch.
It stores the original upload separately as :class:`~aiida.orm.SinglefileData`
and materialises a selected spectrum as :class:`~aiida_feff.data.xasdata.XasData`.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Dict, SinglefileData
from scipy.constants import eV, hbar, m_e

from aiida_feff.data.xasdata import XasData

_HBAR2_OVER_2M_ELECTRON_EV_ANGSTROM2 = (hbar**2 / (2 * m_e)) * (1e20 / eV)


def scaled_chi_arrays(
    k: np.ndarray, chi: np.ndarray, s02: float = 1.0, e0_shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""Apply comparison-only $S_0^2$ and ΔE₀ adjustments to χ(k).

    ΔE₀ shifts the photoelectron energy using
    $E = \frac{\hbar^2}{2m_e} k^2$ in eV, with the conversion factor
    represented by ``_HBAR2_OVER_2M_ELECTRON_EV_ANGSTROM2``. Points below the
    shifted threshold are omitted so the returned k-grid remains monotonic and
    suitable for Larch Fourier transforms.
    """
    k = np.asarray(k, dtype=float)
    chi = np.asarray(chi, dtype=float)
    shifted_k_squared = k**2 - float(e0_shift) / _HBAR2_OVER_2M_ELECTRON_EV_ANGSTROM2
    # Keep k=0 for an unshifted spectrum. Apart from preserving the original
    # data grid, this keeps ensemble χ(k) uncertainty arrays aligned.
    valid = shifted_k_squared >= 0.0
    return np.sqrt(shifted_k_squared[valid]), float(s02) * chi[valid]


def list_experimental_groups(source: SinglefileData) -> list[str]:
    """Return selectable Larch group labels from an Athena project upload.

    Non-Athena files represent one spectrum and return an empty list. The
    function intentionally does not create provenance: it is only used to
    populate a user-interface selector before the import calcfunction runs.
    """
    with _source_path(source) as path:
        from larch.io import is_athena_project, read_athena

        if not is_athena_project(path):
            return []
        project = read_athena(path, do_preedge=False, do_bkg=False)
        return sorted(name for name in project.__dict__ if not name.startswith("_"))


@contextmanager
def _source_path(source: SinglefileData):
    """Yield a temporary local path for a stored uploaded file."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / source.filename
        with source.open(mode="rb") as handle, path.open("wb") as destination:
            shutil.copyfileobj(handle, destination)
        yield str(path)


def _import_experimental_spectrum_impl(path: str, parameters: dict[str, Any]) -> XasData:
    """Read one experimental spectrum using Larch and return an unstored node."""
    from larch.io import guess_filereader, is_athena_project, read_ascii, read_athena

    selected_group = parameters.get("group") or None
    labels = parameters.get("labels") or None
    if isinstance(labels, str):
        labels = [label.strip() for label in labels.split(",") if label.strip()]

    if is_athena_project(path):
        project = read_athena(
            path,
            match=selected_group,
            do_preedge=True,
            do_bkg=bool(parameters.get("autobk", True)),
        )
        groups = [
            value
            for name, value in project.__dict__.items()
            if not name.startswith("_") and hasattr(value, "__dict__")
        ]
        if not groups:
            raise ValueError("No spectrum groups found in Athena project.")
        if len(groups) > 1:
            available = ", ".join(name for name in project.__dict__ if not name.startswith("_"))
            raise ValueError(
                "Select one Athena group before importing. " f"Available groups: {available}"
            )
        group = groups[0]
        reader_name = "read_athena"
    else:
        reader_name = guess_filereader(path)
        if reader_name == "read_ascii":
            group = read_ascii(path, labels=labels)
        else:
            from larch import io

            group = getattr(io, reader_name)(path)

    return _xas_from_larch_group(group, parameters, reader_name)


def _xas_from_larch_group(group: Any, parameters: dict[str, Any], reader_name: str) -> XasData:
    """Map Larch's energy/mu and k/chi attributes into ``XasData``."""
    energy = _first_array(group, "energy", "omega")
    mu = _first_array(group, "mu", "xmu")
    k = _first_array(group, "k")
    chi = _first_array(group, "chi", "chi_k")

    if (
        energy is not None
        and mu is not None
        and (k is None or chi is None)
        and parameters.get("autobk", True)
    ):
        from larch.xafs import autobk

        autobk(energy, mu, group=group)
        k = _first_array(group, "k")
        chi = _first_array(group, "chi", "chi_k")

    if (energy is None or mu is None) and (k is None or chi is None):
        available = ", ".join(getattr(group, "array_labels", [])) or "none"
        raise ValueError(
            "Larch could not identify energy/mu or k/chi columns. "
            f"Detected columns: {available}."
        )

    out = XasData()
    if energy is not None and mu is not None:
        out.set_spectrum(
            energy,
            mu,
            _first_array(group, "mu0"),
            e0=float(getattr(group, "e0", 0.0)),
        )
    if k is not None and chi is not None:
        out.set_chi(k, chi)
    out.base.extras.set("source_kind", "experimental")
    out.base.extras.set("larch_reader", reader_name)
    out.base.extras.set("larch_import_options", dict(parameters))
    return out


def _first_array(group: Any, *names: str) -> np.ndarray | None:
    """Return the first finite one-dimensional Larch array among ``names``."""
    for name in names:
        value = getattr(group, name, None)
        if value is not None:
            array = np.asarray(value, dtype=float)
            if array.ndim == 1 and len(array):
                return array
    return None


@calcfunction
def scale_simulated_spectrum(simulated: XasData, parameters: Dict) -> XasData:
    """Store a provenance-linked, $S_0^2$/ΔE₀-scaled simulated spectrum."""
    options = parameters.get_dict()
    out = XasData()
    for name in simulated.get_arraynames():
        out.set_array(name, simulated.get_array(name))
    for key, value in simulated.base.extras.all.items():
        out.base.extras.set(key, value)
    if "k" not in simulated.get_arraynames() or "chi_k" not in simulated.get_arraynames():
        raise ValueError("Simulated spectrum has no χ(k) arrays to scale.")
    k, chi = scaled_chi_arrays(
        simulated.get_array("k"),
        simulated.get_array("chi_k"),
        float(options.get("s02", 1.0)),
        float(options.get("e0_shift", 0.0)),
    )
    out.set_chi(k, chi)
    out.base.extras.set("comparison_s02", float(options.get("s02", 1.0)))
    out.base.extras.set("comparison_e0_shift", float(options.get("e0_shift", 0.0)))
    return out


@calcfunction
def import_experimental_spectrum(source: SinglefileData, parameters: Dict) -> XasData:
    """Import a Larch-readable experimental spectrum into ``XasData``.

    ``parameters`` may set ``labels`` for plain text columns, ``group`` for an
    Athena project group, and ``autobk`` to derive χ(k) from μ(E). The original
    upload is retained as the input provenance node.
    """
    options = parameters.get_dict()
    with _source_path(source) as path:
        result = _import_experimental_spectrum_impl(path, options)
    result.base.extras.set("source_file", source.filename)
    result.base.extras.set("source_kind", "experimental")
    return result
