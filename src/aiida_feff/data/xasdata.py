"""XasData: output data node storing parsed FEFF spectra as numpy arrays."""

from __future__ import annotations

import numpy as np
from aiida.orm import ArrayData


class XasData(ArrayData):
    """Data node that carries XAS spectra as numpy arrays.

    Which arrays are present depends on where the node came from.  A node
    parsed from a FEFF run holds χ(k) alone, read straight from ``chi.dat``;
    the plugin no longer retrieves ``xmu.dat`` or runs larch's background
    subtraction over simulated data, so ``energy``, ``mu``, ``mu0`` and a
    meaningful ``e0`` are absent.  A node imported from experiment by
    :mod:`aiida_feff.calcfunctions.experimental` has μ(E), and χ(k) too once
    a background has been subtracted.  Reach for :meth:`get_arraynames` before
    assuming either.

    χ(k) arrays
    -----------
    k : (M,) float
        Photoelectron wavenumber grid in Å⁻¹.
    chi_k : (M,) float
        EXAFS χ(k).  May be NaN where an ensemble average had no contributing
        member; ``chi_k_count`` says where.
    chi_k_std : (M,) float
        Sample standard deviation across ensemble members (averaged nodes).
    chi_k_count : (M,) float
        Members contributing at each k (averaged nodes).

    μ(E) arrays (experimental imports)
    ----------------------------------
    energy : (N,) float
        Energy grid in eV **relative to E0**.  Add :attr:`e0` for absolute.
    mu : (N,) float
        Total absorption μ(E).
    mu0 : (N,) float
        Atomic background μ₀(E).

    Fourier transform arrays, from :func:`~aiida_feff.calcfunctions.larch.chi_k_to_r`
    ---------------------------------------------------------------------------------
    ``r``, ``chir_mag``, ``chir_re``, ``chir_im``, each (P,) float.  χ(R)
    carries units Å^-(kweight+1), so read ``kweight`` back out of
    ``fourier_params`` before labelling an axis.

    Metadata, stored in node **attributes**
    ---------------------------------------
    Attributes rather than extras: extras stay mutable after storage and are
    excluded from the node hash, so scientific metadata kept there can be
    rewritten on a stored node and makes caching treat physically different
    nodes as identical.

    chi_source : str  — where χ(k) came from, e.g. ``feff.chi.dat``
    absorber_element : str  — element FEFF put the core hole on
    frame_index, site_index : int  — position in the ensemble
    e0 : float  — threshold energy in eV (absolute); 0.0 for FEFF nodes
    source_file : str  — original filename tag
    fourier_params : dict  — FT parameters used (kmin, kmax, kweight, window, …)
    n_snapshots : int  — number of ensemble members (averaged nodes only)
    code_versions : dict  — versions of larch / pymatgen / … that produced this
    feff_version : str  — FEFF banner parsed from ``log.dat``

    Usage::

        xas = XasData()
        xas.set_chi(k, chi_k)
        xas.store()

        chi = xas.get_array("chi_k")
    """

    # ------------------------------------------------------------------
    # μ(E) — experimental imports only; FEFF runs go through set_chi
    # ------------------------------------------------------------------

    def set_spectrum(
        self,
        energy: np.ndarray,
        mu: np.ndarray,
        mu0: np.ndarray | None = None,
        e0: float = 0.0,
    ) -> None:
        """Store μ(E), with ``energy`` relative to ``e0``."""
        self.set_array("energy", np.asarray(energy, dtype=float))
        self.set_array("mu", np.asarray(mu, dtype=float))
        if mu0 is not None:
            self.set_array("mu0", np.asarray(mu0, dtype=float))
        self.base.attributes.set("e0", float(e0))

    def set_chi(self, k: np.ndarray, chi_k: np.ndarray) -> None:
        """Store χ(k) on the wavenumber grid ``k``."""
        self.set_array("k", np.asarray(k, dtype=float))
        self.set_array("chi_k", np.asarray(chi_k, dtype=float))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def energy(self) -> np.ndarray:
        """Energy grid in eV, relative to :attr:`e0`."""
        return self.get_array("energy")

    @property
    def absolute_energy(self) -> np.ndarray:
        """Energy grid in eV on an absolute scale (``energy + e0``)."""
        return self.get_array("energy") + self.e0

    @property
    def mu(self) -> np.ndarray:
        """Absorption μ(E)."""
        return self.get_array("mu")

    @property
    def chi_k(self) -> np.ndarray:
        """EXAFS χ(k)."""
        return self.get_array("chi_k")

    @property
    def k(self) -> np.ndarray:
        """Photoelectron wavenumber grid in Å⁻¹."""
        return self.get_array("k")

    @property
    def e0(self) -> float:
        """Edge threshold energy in eV (absolute)."""
        return float(self.base.attributes.get("e0", 0.0))
