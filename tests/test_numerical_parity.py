"""End-to-end numerical parity validation between AiiDA-FEFF and md-exafs."""

import numpy as np
from ase.build import bulk

# Tier 1 (Core)
from md_exafs.experimental import (
    scaled_chi_arrays as core_scaled_chi,
)
from md_exafs.experimental import (
    shifted_k_mask as core_shifted_mask,
)
from md_exafs.paths import path_chi as core_path_chi
from md_exafs.selection import resolve_frame_absorbers as core_resolve_absorbers
from md_exafs.spectra import xftf_arrays as core_xftf

# Tier 2 (AiiDA Plugin)
from aiida_feff.calcfunctions.exafs import path_chi as aiida_path_chi
from aiida_feff.calcfunctions.experimental import (
    scaled_chi_arrays as aiida_scaled_chi,
)
from aiida_feff.calcfunctions.experimental import (
    shifted_k_mask as aiida_shifted_mask,
)
from aiida_feff.calcfunctions.larch import xftf_arrays as aiida_xftf
from aiida_feff.workflows.ensemble import _resolve_absorber_sites


def test_path_chi_parity():
    """Assert path_chi outputs are bit-for-bit identical."""
    k_native = np.linspace(0.0, 20.0, 50)
    feff_data = np.ones((50, 6))
    feff_data[:, 5] = k_native  # rep

    k_out = np.linspace(2.0, 14.0, 100)
    chi_core = core_path_chi(
        k_native=k_native,
        feff_data=feff_data,
        r_eff=2.5,
        degeneracy=12.0,
        k_out=k_out,
        sigma2=0.003,
        s02=0.85,
        e0_shift=2.0,
    )
    chi_aiida = aiida_path_chi(
        k_native=k_native,
        feff_data=feff_data,
        r_eff=2.5,
        degeneracy=12.0,
        k_out=k_out,
        sigma2=0.003,
        s02=0.85,
        e0_shift=2.0,
    )

    diff = np.max(np.abs(chi_core - chi_aiida))
    assert diff < 1e-12, f"path_chi discrepancy: {diff}"


def test_fourier_transform_parity():
    """Assert xftf_arrays outputs are bit-for-bit identical."""
    k = np.linspace(2.0, 14.0, 120)
    chi = np.sin(2.0 * k) * np.exp(-0.05 * k**2)

    ft_params = {"kmin": 2.5, "kmax": 13.0, "kweight": 2, "dk": 1.0, "window": "kaiser"}
    res_core = core_xftf(k, chi, ft_params)
    res_aiida = aiida_xftf(k, chi, ft_params)

    assert np.allclose(res_core["r"], res_aiida["r"])
    diff = np.max(np.abs(res_core["chir_mag"] - res_aiida["chir_mag"]))
    assert diff < 1e-12, f"xftf chir_mag discrepancy: {diff}"


def test_experimental_scaling_parity():
    """Assert experimental scaling outputs are bit-for-bit identical."""
    k = np.linspace(0.5, 12.0, 80)
    chi = np.cos(k)

    mask_core = core_shifted_mask(k, 5.0)
    mask_aiida = aiida_shifted_mask(k, 5.0)
    assert np.array_equal(mask_core, mask_aiida)

    k_c, chi_c = core_scaled_chi(k, chi, s02=0.9, e0_shift=5.0)
    k_a, chi_a = aiida_scaled_chi(k, chi, s02=0.9, e0_shift=5.0)
    assert np.allclose(k_c, k_a)
    assert np.allclose(chi_c, chi_a)


def test_absorber_resolution_parity():
    """Assert absorber resolution is identical."""
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = bulk("Cu", "fcc", a=3.61)

    class DummyStructure:
        def get_pymatgen_structure(self):
            return AseAtomsAdaptor().get_structure(atoms)

    indices_core = core_resolve_absorbers(["Cu"] * len(atoms), "Cu")
    indices_aiida = _resolve_absorber_sites(DummyStructure(), "Cu")
    assert indices_core == indices_aiida
