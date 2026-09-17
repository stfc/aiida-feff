"""FeffParameters: AiiDA Dict node for FEFF calculation inputs.

This node contains only the parameters that directly determine what FEFF
computes — i.e., the inputs to one FEFF run.  It is intentionally *not*
a mirror of ``larch_cli_wrapper.feff_utils.FeffConfig``.

``FeffConfig`` mixes three unrelated concerns:

* FEFF calculation parameters  ← these belong here
* Fourier-transform parameters ← these belong in a separate ``orm.Dict``
                                  passed to ``chi_k_to_r``
* Runtime/execution parameters ← these belong in ``metadata.options``
                                  (AiiDA scheduler) or have no AiiDA
                                  equivalent (e.g. ``cleanup_feff_files``)

Keeping those three concerns separate means:
* This node is small, focused, and trivially re-used across different
  analysis protocols without dragging in unrelated FT settings.
* The provenance graph accurately reflects which parameters influenced
  which outputs (FEFF cards → chi.dat; FT params → chi(R)).
* There is nothing non-standard to bridge when migrating alc-dls-exafs
  callers to the aiida workflow.
"""

from __future__ import annotations

from aiida.orm import Dict
from md_exafs.feff_input import FeffConfig

VALID_EDGE_LABELS = frozenset({"K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"})
VALID_SPECTRUM_TYPES = frozenset({"EXAFS"})

#: Every key this node understands.  Anything else is rejected by
#: :meth:`FeffParameters.validate`: a silently-ignored key produces a FEFF run
#: at default settings while the user believes their value took effect.
VALID_KEYS = frozenset(
    {
        # calculation control
        "edge",
        "spectrum_type",
        "radius",
        "absorbing_atom",
        "absorbing_atoms",
        "exclude_hydrogen",
        # FEFF cards
        "s02",
        "nleg",
        "scf",
        "exchange",
        "control",
        "print",
        "exafs",
        "criteria",
        "delete_tags",
    }
)

#: Keys people reach for that this node does not implement, mapped to the name
#: that actually works.  Used to turn a silent no-op into a pointed error.
_KEY_ALIASES = {
    "calc_mode": "spectrum_type",
    "rpath": "radius",
    "cluster_radius": "radius",
    "absorber": "absorbing_atom",
    "amp_reduction": "s02",
}


class FeffParameters(Dict):
    """Typed :class:`~aiida.orm.Dict` for FEFF calculation parameters (ADR 0004).

    Contains only parameters that control the FEFF code itself — i.e. the
    contents of ``feff.inp``.  Fourier-transform and fitting parameters are
    *not* stored here; pass them as a plain ``orm.Dict`` to the relevant
    calcfunctions.

    Tag normalisation and ``feff.inp`` generation are delegated to
    :class:`md_exafs.FeffConfig` (ADR 0004); this class owns only the
    AiiDA-facing schema and its validation.

    Keys
    ----
    ``edge`` : str  *(required)*
        Absorption edge — ``"K"``, ``"L1"`` … ``"M5"``.
    ``spectrum_type`` : str, default ``"EXAFS"``
        Currently only ``"EXAFS"`` is supported.
    ``radius`` : float, default ``5.5``
        Cluster / path radius in Å (FEFF ``RPATH`` card).
    ``absorbing_atom`` : int, default ``0``
        0-based index of the absorbing site in the ``StructureData``.
    ``absorbing_atoms`` : int | str | list[int]
        Multi-site absorber specification; see
        :func:`~aiida_feff.workflows.ensemble._resolve_absorber_sites`.
    ``exclude_hydrogen`` : bool, default ``False``
        Remove H atoms before generating ``feff.inp``.

    FEFF card fields (all optional; ``None`` / absent means omit the card):

    ``s02``      : float,      default ``1.0``
    ``nleg``     : int,        default ``6``
    ``scf``      : str,        e.g. ``"4.0 0 30 0.2 1"``
    ``exchange`` : int | str,  default ``"0 0 0"``
    ``control``  : str,        e.g. ``"1 1 1 1 1 1"``
    ``print``    : str,        default ``"1 0 0 0 0 3"``
    ``exafs``    : int,        FEFF EXAFS k_max card
    ``criteria`` : str,        e.g. ``"4.0 2.5"``
    ``delete_tags`` : list[str]  Card names to strip from the generated file.

    Example::

        params = FeffParameters(dict={
            "edge": "K",
            "radius": 6.0,
            "s02": 0.9,
            "scf": "4.0 0 30 0.2 1",
            "nleg": 6,
        })

    Fourier-transform parameters for downstream larch analysis::

        ft_params = orm.Dict({"kmin": 2.0, "kmax": 14.0, "kweight": 2, "dk": 1.0})
        chir = chi_k_to_r(xas_data, ft_params)
    """

    _storable = True

    def __init__(self, dict: dict | None = None, **kwargs):
        """Create a FeffParameters node."""
        super().__init__(dict=dict or {}, **kwargs)
        if dict:
            self.validate()

    def validate(self) -> None:
        """Raise :exc:`ValueError` if the stored dict is invalid."""
        d = self.get_dict()
        self._validate_keys(d)

        edge = d.get("edge")
        if edge is None:
            raise ValueError("'edge' is required")
        if edge not in VALID_EDGE_LABELS:
            raise ValueError(f"edge must be one of {sorted(VALID_EDGE_LABELS)}, got {edge!r}")

        st = d.get("spectrum_type", "EXAFS")
        if st not in VALID_SPECTRUM_TYPES:
            raise ValueError(f"spectrum_type must be 'EXAFS', got {st!r}")

        radius = d.get("radius")
        if radius is not None and float(radius) <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        s02 = d.get("s02")
        if s02 is not None and float(s02) < 0:
            raise ValueError(f"s02 must be >= 0, got {s02}")

    @staticmethod
    def _validate_keys(d: dict) -> None:
        """Reject keys this node will not act on.

        An unknown key that is accepted but never used produces a FEFF run at
        default settings while the caller believes their value took effect, so
        unknown keys are an error rather than a warning.
        """
        unknown = sorted(set(d) - VALID_KEYS)
        if not unknown:
            return
        hints = [f"{k!r} (did you mean {_KEY_ALIASES[k]!r}?)" for k in unknown if k in _KEY_ALIASES]
        plain = [repr(k) for k in unknown if k not in _KEY_ALIASES]
        raise ValueError(
            f"Unknown FeffParameters key(s): {', '.join(hints + plain)}. "
            f"Recognised keys: {sorted(VALID_KEYS)}"
        )

    @property
    def edge(self) -> str:
        """Absorption edge label."""
        return str(self["edge"])

    @property
    def spectrum_type(self) -> str:
        """Spectrum type."""
        return str(self.get("spectrum_type", "EXAFS"))

    @property
    def radius(self) -> float:
        """Cluster radius in Å."""
        return float(self.get("radius", 5.5))

    def to_feff_config(self) -> FeffConfig:
        """Convert stored dictionary to an md_exafs.FeffConfig instance."""
        d = self.get_dict()
        # Filter keys relevant to FeffConfig
        config_kwargs = {}
        for key in (
            "spectrum_type",
            "edge",
            "radius",
            "exclude_hydrogen",
            "control",
            "print",
            "s02",
            "scf",
            "exchange",
            "nleg",
            "exafs",
            "criteria",
            "delete_tags",
        ):
            if key in d:
                config_kwargs[key] = d[key]
        return FeffConfig(**config_kwargs)

    def to_pymatgen_user_tags(self) -> dict:
        """Build user_tag_settings dict delegating to FeffConfig."""
        d = self.get_dict()
        cfg = self.to_feff_config()
        tags = cfg.to_pymatgen_user_tags()
        del_list = list(tags.get("_del", []))
        if "scf" in d and d["scf"] is None and "SCF" not in del_list:
            del_list.append("SCF")
        if del_list:
            tags["_del"] = list(dict.fromkeys(del_list))
        return tags

    def to_feff_cards(self) -> list[str]:
        """Render the FEFF cards preview."""
        tags = self.to_pymatgen_user_tags()
        deleted = set(tags.pop("_del", []))

        cards = [f"EDGE  {self.edge}", f"RPATH {self.radius}"]
        cards += [f"{name:<6s}{value}" for name, value in tags.items() if name not in deleted]
        cards += [f"* deleted: {name}" for name in sorted(deleted)]
        return cards


__all__ = [
    "FeffParameters",
    "VALID_EDGE_LABELS",
    "VALID_SPECTRUM_TYPES",
    "VALID_KEYS",
]
