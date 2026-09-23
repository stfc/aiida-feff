"""Tests for the ``verdi data feff`` commands."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from aiida_feff.cli import cmd_export, cmd_list, cmd_show
from aiida_feff.data.parameters import FeffParameters


@pytest.fixture()
def runner():
    return CliRunner()


class TestExport:
    def test_renders_feff_cards(self, runner, aiida_profile_clean):
        node = FeffParameters(dict={"edge": "L3", "radius": 6.5, "s02": 0.9}).store()
        result = runner.invoke(cmd_export, [str(node.pk)])
        assert result.exit_code == 0, result.output
        assert "EDGE" in result.output
        assert "L3" in result.output
        assert "RPATH" in result.output

    def test_writes_to_a_file(self, runner, tmp_path, aiida_profile_clean):
        node = FeffParameters(dict={"edge": "K"}).store()
        target = tmp_path / "feff.inp"
        result = runner.invoke(cmd_export, [str(node.pk), "-o", str(target)])
        assert result.exit_code == 0, result.output
        assert "EDGE" in target.read_text()

    def test_rejects_a_non_parameters_node(self, runner, generate_xas_data, aiida_profile_clean):
        node = generate_xas_data().store()
        result = runner.invoke(cmd_export, [str(node.pk)])
        assert result.exit_code != 0


class TestShow:
    def test_lists_arrays_and_metadata(self, runner, generate_xas_data, aiida_profile_clean):
        node = generate_xas_data().store()
        result = runner.invoke(cmd_show, [str(node.pk)])
        assert result.exit_code == 0, result.output
        assert "chi_k" in result.output
        assert "k" in result.output


class TestList:
    def test_lists_both_types(self, runner, generate_xas_data, aiida_profile_clean):
        FeffParameters(dict={"edge": "K"}).store()
        generate_xas_data().store()
        result = runner.invoke(cmd_list, [])
        assert result.exit_code == 0, result.output
        assert "FeffParameters" in result.output
        assert "XasData" in result.output

    def test_type_filter(self, runner, generate_xas_data, aiida_profile_clean):
        FeffParameters(dict={"edge": "K"}).store()
        generate_xas_data().store()
        result = runner.invoke(cmd_list, ["--type", "parameters"])
        assert "FeffParameters" in result.output
        assert "XasData" not in result.output

    def test_limit_is_respected(self, runner, aiida_profile_clean):
        for _ in range(5):
            FeffParameters(dict={"edge": "K"}).store()
        result = runner.invoke(cmd_list, ["--type", "parameters", "--limit", "2"])
        assert result.output.count("FeffParameters") == 2

    def test_empty_database_says_so(self, runner, aiida_profile_clean):
        result = runner.invoke(cmd_list, [])
        assert "No nodes found" in result.output
