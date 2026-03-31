"""training/prepare_dataset.py uchun unit testlar."""

import pytest
from prepare_dataset import remap_label


class TestRemapLabel:
    """remap_label funksiyasi testlari."""

    def test_single_line_no_change(self):
        content = "0 0.5 0.5 0.1 0.1"
        result = remap_label(content, {0: 0})
        assert result == "0 0.5 0.5 0.1 0.1\n"

    def test_single_line_remap(self):
        content = "0 0.5 0.5 0.1 0.1"
        result = remap_label(content, {0: 3})
        assert result == "3 0.5 0.5 0.1 0.1\n"

    def test_multiple_lines(self):
        content = "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8"
        result = remap_label(content, {0: 4, 1: 5})
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "4 0.1 0.2 0.3 0.4"
        assert lines[1] == "5 0.5 0.6 0.7 0.8"

    def test_unknown_class_skipped(self):
        content = "0 0.1 0.2 0.3 0.4\n9 0.5 0.6 0.7 0.8"
        result = remap_label(content, {0: 3})
        assert result == "3 0.1 0.2 0.3 0.4\n"

    def test_empty_content(self):
        result = remap_label("", {0: 0})
        assert result == ""

    def test_whitespace_only(self):
        result = remap_label("   \n\n  ", {0: 0})
        assert result == ""

    def test_coordinates_preserved(self):
        content = "0 0.8559027777777778 0.3680555555555556 0.050694444444444445 0.109375"
        result = remap_label(content, {0: 5})
        assert "0.8559027777777778" in result
        assert result.startswith("5 ")

    def test_multiple_remaps_handrise(self):
        content = "0 0.1 0.2 0.3 0.4\n1 0.5 0.5 0.1 0.1\n2 0.9 0.9 0.05 0.05"
        result = remap_label(content, {0: 0, 1: 1, 2: 2})
        lines = result.strip().split("\n")
        assert lines[0].startswith("0 ")
        assert lines[1].startswith("1 ")
        assert lines[2].startswith("2 ")

    def test_discuss_remap(self):
        content = "0 0.5 0.5 0.1 0.1"
        result = remap_label(content, {0: 3})
        assert result.startswith("3 ")

    def test_bowturnhead_remap(self):
        content = "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8"
        result = remap_label(content, {0: 4, 1: 5})
        lines = result.strip().split("\n")
        assert lines[0].startswith("4 ")
        assert lines[1].startswith("5 ")
