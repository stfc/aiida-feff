"""XasData: output data node storing parsed FEFF spectra as numpy arrays."""

from __future__ import annotations

import numpy as np
from aiida.orm import ArrayData


class XasData(ArrayData):
    """Data node that carries parsed XAS spectra from a FEFF calculation.

    Arrays stored
    -------------
    energy : (N,) float
        Absolute energy grid in eV.
    mu : (N,) float
        Total absorption μ(E) (from ``xmu.dat``).
    mu0 : (N,) float
        Atomic background μ₀(E).
    chi_k : (M,) float
        EXAFS χ(k) on a uniform k-grid.
    k : (M,) float
        Photoelectron wavenumber grid in Å⁻¹.

    Optional FT result arrays (populated by :func:`~aiida_feff.calcfunctions.larch.chi_k_to_r`)
    --------------------------------------------------------------------------------------------
    r : (P,) float
    chir_mag : (P,) float
    chir_re  : (P,) float
    chir_im  : (P,) float

    Metadata stored in :attr:`extras`
    -----------------------------------
    edge : str
    e0 : float  — threshold energy in eV
    source_file : str  — original filename tag
    fourier_params : dict  — FT parameters used (kmin, kmax, kweight, …)
    n_snapshots : int  — number of ensemble members (averaged nodes only)

    Usage::

        xas = XasData()
        xas.set_spectrum(energy, mu, mu0)
        xas.set_chi(k, chi_k)
        xas.store()

        energy = xas.get_array("energy")
        chi    = xas.get_array("chi_k")
    """

    # ------------------------------------------------------------------
    # Spectrum (xmu.dat)
    # ------------------------------------------------------------------

    def set_spectrum(
        self,
        energy: np.ndarray,
        mu: np.ndarray,
        mu0: np.ndarray | None = None,
        e0: float = 0.0,
    ) -> None:
        """Store μ(E) data from ``xmu.dat``."""
        self.set_array("energy", np.asarray(energy, dtype=float))
        self.set_array("mu", np.asarray(mu, dtype=float))
        if mu0 is not None:
            self.set_array("mu0", np.asarray(mu0, dtype=float))
        self.base.extras.set("e0", float(e0))

    def set_chi(self, k: np.ndarray, chi_k: np.ndarray) -> None:
        """Store χ(k) data from ``chi.dat``."""
        self.set_array("k", np.asarray(k, dtype=float))
        self.set_array("chi_k", np.asarray(chi_k, dtype=float))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def energy(self) -> np.ndarray:
        """Energy grid in eV."""
        return self.get_array("energy")

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
        """Edge threshold energy in eV."""
        return float(self.base.extras.get("e0", 0.0))
