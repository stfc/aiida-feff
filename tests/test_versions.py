"""Tests for version provenance capture."""

from __future__ import annotations

import pytest

from aiida_feff.versions import dependency_versions, parse_feff_version


class TestDependencyVersions:
    def test_reports_the_packages_that_shape_the_physics(self):
        versions = dependency_versions()
        # pymatgen writes feff.inp; larch does the background subtraction.
        # Without these two a spectrum cannot be reproduced.
        assert "pymatgen" in versions
        assert "xraylarch" in versions
        assert "aiida-feff" in versions

    def test_absent_package_is_omitted_not_guessed(self, monkeypatch):
        import aiida_feff.versions as module

        monkeypatch.setattr(module, "_PACKAGES", ("definitely-not-installed",))
        assert dependency_versions() == {}


class TestParseFeffVersion:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Feff8L (EXAFS)  0.1\n", "Feff8L"),
            ("  Feff 8.50L\n", "Feff8.50L"),
            (" FEFF 9.6.4 \n", "Feff9.6.4"),
            ("feff  10.0\n", "Feff10.0"),
        ],
    )
    def test_recognises_banners(self, text, expected):
        assert parse_feff_version(text) == expected

    def test_returns_none_without_a_banner(self):
        assert parse_feff_version("no version here\njust output\n") is None

    def test_ignores_matches_past_the_header(self):
        # "feff0001.dat" appears throughout files.dat and must not be mistaken
        # for a version string.
        body = "\n".join(["padding"] * 40) + "\nfeff0001.dat\n"
        assert parse_feff_version(body) is None
