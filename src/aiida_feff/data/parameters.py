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

VALID_EDGE_LABELS = frozenset({"K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"})
VALID_SPECTRUM_TYPES = frozenset({"EXAFS"})


class FeffParameters(Dict):
    """Typed :class:`~aiida.orm.Dict` for FEFF calculation parameters.

    Contains only parameters that control the FEFF code itself — i.e. the
    contents of ``feff.inp``.  Fourier-transform and fitting parameters are
    *not* stored here; pass them as a plain ``orm.Dict`` to the relevant
    calcfunctions.

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :exc:`ValueError` if the stored dict is invalid."""
        d = self.get_dict()

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

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def edge(self) -> str:
        """Absorption edge label."""
        return str(self["edge"])

    @property
    def spectrum_type(self) -> str:
        """Spectrum type (always ``'EXAFS'`` for now)."""
        return str(self.get("spectrum_type", "EXAFS"))

    @property
    def radius(self) -> float:
        """Cluster / path radius in Å."""
        return float(self.get("radius", 5.5))

    # ------------------------------------------------------------------
    # Pymatgen input generation
    # ------------------------------------------------------------------

    def to_pymatgen_user_tags(self) -> dict:
        """Build the ``user_tag_settings`` dict for ``pymatgen MPEXAFSSet``.

        Returns the same structure as
        ``larch_cli_wrapper.feff_utils.FeffConfig.to_pymatgen_user_tags()``,
        so both packages drive the same pymatgen input writer.
        """
        d = self.get_dict()
        tags: dict = {}

        field_map = {
            "CONTROL": d.get("control"),
            "PRINT": d.get("print", "1 0 0 0 0 3"),
            "S02": d.get("s02", 1.0),
            "SCF": d.get("scf"),
            "EXCHANGE": d.get("exchange", "0 0 0"),
            "NLEG": d.get("nleg", 6),
            "EXAFS": d.get("exafs"),
            "CRITERIA": d.get("criteria"),
        }

        for key, val in field_map.items():
            if val is not None:
                tags[key] = _normalize_tag(key, val)

        raw_del = d.get("delete_tags")
        del_list: list[str] = [raw_del] if isinstance(raw_del, str) else list(raw_del or [])

        # scf=None means "explicitly delete pymatgen's default SCF card"
        if "scf" in d and d["scf"] is None and "SCF" not in del_list:
            del_list.append("SCF")

        if del_list:
            tags["_del"] = list(dict.fromkeys(del_list))  # dedup, preserve order

        return tags


# ---------------------------------------------------------------------------
# FEFF card normalisation  (same rules as FeffConfig._normalize_tag)
# ---------------------------------------------------------------------------


def _to_str_tokens(value) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(x) for x in value]
    return value.split() if isinstance(value, str) else [str(value)]


def _normalize_tag(name: str, value) -> str:  # noqa: PLR0911
    """Normalise a FEFF card value to a string.

    Rules are identical to ``FeffConfig._normalize_tag`` so that both
    packages always produce the same feff.inp for the same inputs.
    """
    key = name.strip().upper()
    tokens = _to_str_tokens(value)

    if key == "S02":
        s02 = float(tokens[0])
        if s02 < 0:
            raise ValueError(f"S02 must be >= 0, got {s02}")
        return str(s02)

    if key == "EXAFS":
        exafs = int(float(tokens[0]))
        if exafs <= 0:
            raise ValueError(f"EXAFS must be > 0, got {exafs}")
        return str(exafs)

    if key in ("PRINT", "CONTROL"):
        try:
            return " ".join(str(int(float(t))) for t in tokens)
        except Exception as exc:
            raise ValueError(f"{key} expects integer tokens, got {value!r}") from exc

    if key == "SCF":
        scf_max_tokens = 5
        if not (1 <= len(tokens) <= scf_max_tokens):
            raise ValueError(f"SCF expects 1–{scf_max_tokens} tokens, got {len(tokens)}: {value!r}")

        def _fmt(x: float) -> str:
            s = str(float(x))
            return s.rstrip("0").rstrip(".") if "." in s else str(int(float(x)))

        rfms1 = float(tokens[0])
        if rfms1 <= 0:
            raise ValueError(f"SCF rfms1 must be > 0, got {rfms1}")
        lfms1 = int(float(tokens[1])) if len(tokens) >= 2 else 0
        nscmt = int(float(tokens[2])) if len(tokens) >= 3 else 30
        ca = float(tokens[3]) if len(tokens) >= 4 else 0.2
        nmix = int(float(tokens[4])) if len(tokens) >= 5 else 1
        return " ".join([_fmt(rfms1), str(lfms1), str(nscmt), _fmt(ca), str(nmix)])  # noqa: PLR2004

    if key == "NLEG":
        nleg = int(float(tokens[0]))
        if nleg <= 0:
            raise ValueError(f"NLEG must be > 0, got {nleg}")
        return str(nleg)

    if key == "EXCHANGE":
        import warnings

        if len(tokens) == 1 and tokens[0] == "0":
            warnings.warn(
                "EXCHANGE '0' is ambiguous — FEFF needs 'ixc Vr Vi'. "
                "Using '0 0 0'. Pass exchange='0 0 0' to silence this.",
                UserWarning,
                stacklevel=2,
            )
            return "0 0 0"
        return " ".join(tokens)

    return " ".join(tokens)
