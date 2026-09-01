"""Tests for trim_top_layers.py.

The key correctness criterion: when both the original and the trimmed file
are parsed by the simulator's own G-code parser (app.instructions.gcode), the
trimmed file must reproduce the original file's top layers exactly (modulo the
Z shift), and nothing else.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trim_top_layers import trim_text  # noqa: E402

HEADER = "; synthetic test file\nM104 S200\nG90\nM82\nG92 E0\n"

FOOTER = "G1 E-0.8 F1800 ; final retract\nG1 Z50 F3000\nM104 S0\n"

TRICKY_GCODE = HEADER + (
    "G1 Z0.2 F3000\n"
    "G1 X0 Y0\n"
    "G1 X10 E1.0      ; layer 1 extrusion\n"
    "G1 Y5 E2.0       ; still layer 1\n"
    "G1 Z0.4\n"
    "G1 X0 E3.0       ; layer 2\n"
    "G91\n"
    "G1 X1            ; relative move, no Z change\n"
    "G90\n"
    "G92 E0           ; reset absolute E\n"
    "G1 Z0.6\n"
    "G1 X5 E0.5       ; layer 3 (first kept with -n 2)\n"
    "M83\n"
    "G1 Z0.8\n"
    "G1 Y10 E0.4      ; layer 4, relative mode\n"
    "G2 X5 Y5 I-2.5 J-2.5 E0.2 ; arc back, relative E\n"
) + FOOTER


def filaments_of(text):
    """Parse gcode text with the simulator's own parser, return filaments."""
    from app.configs.printer import Printer

    printer = Printer(
        config_dict={
            "nozzle_diameter": 0.4,
            "feedstock_filament_diameter": 1.75,
            "nozzle_jerk_speed": 8.0,
            "extruder_jerk_speed": 5.0,
            "nozzle_acceleration": 500.0,
            "extruder_acceleration": 1000.0,
        }
    )
    if printer.feedstock_filament_diameter <= 0:
        raise AssertionError("bad printer config")
    # Lazy import to keep collection cheap if app is not importable
    from app.instructions.gcode import Gcode

    instruction = Gcode(gcode_content=text, printer=printer)
    instruction.read()
    return instruction


def filament_tuples(instruction):
    return [
        (
            round(f[0][0], 6),
            round(f[0][1], 6),
            round(f[0][2], 6),
            round(f[1][0], 6),
            round(f[1][1], 6),
            round(f[1][2], 6),
            round(f[2], 6),
        )
        for f in instruction.filaments_coordinates
    ]


def test_keeps_top_layers_with_exact_geometry():
    trimmed, stats = trim_text(TRICKY_GCODE, n_layers=2)
    assert stats["layers_total"] == 4
    assert stats["layers_kept"] == 2
    assert stats["z_cut"] == pytest.approx(0.6)
    assert stats["z_top"] == pytest.approx(0.8)

    original = filaments_of(TRICKY_GCODE)
    expected = [
        f
        for f in filament_tuples(original)
        if f[2] >= 0.6 - 1e-9 and f[5] >= 0.6 - 1e-9
    ]
    kept = filament_tuples(filaments_of(trimmed))

    # Same segments, with the Z shift applied (x/y/volume identical).
    assert len(kept) == len(expected) > 0
    for kept_f, expected_f in zip(kept, expected):
        assert kept_f[0:2] == expected_f[0:2]
        assert kept_f[3:4] == expected_f[3:4]
        assert kept_f[2] == pytest.approx(expected_f[2] - stats["z_shift"])
        assert kept_f[5] == pytest.approx(expected_f[5] - stats["z_shift"])
        assert kept_f[6] == expected_f[6]


def test_shift_puts_first_kept_layer_one_layer_above_bed():
    trimmed, stats = trim_text(TRICKY_GCODE, n_layers=2)
    assert stats["z_shift"] == pytest.approx(0.4)
    assert stats["first_layer_height"] == pytest.approx(0.2)
    for filament in filament_tuples(filaments_of(trimmed)):
        assert min(filament[2], filament[5]) > 0.0  # nothing at/below bed


def test_mode_commands_are_normalized():
    trimmed, _ = trim_text(TRICKY_GCODE, n_layers=2)
    kept_lines = trimmed.split("; travel to trim point", 1)[1]
    # No leftover mode commands in the kept section could fight the rewrite.
    for line in kept_lines.splitlines():
        code = line.split(";", 1)[0].strip()
        if code and not code.startswith(";"):
            assert code.split()[0] not in (
                "G90", "G91", "M82", "M83", "G92", "G20", "G21",
            ), line
    # Absolute E with a G92 E0 reset must become the correct delta.
    assert "G1 X5 E0.5" in trimmed


def test_relative_xy_section_is_absolutized():
    trimmed, _ = trim_text(TRICKY_GCODE, n_layers=2)
    # The G91 move G1 X1 (from X0) is folded into the intro travel.
    assert "G1 X1 Y5 Z0 ; travel to trim point" in trimmed
    for filament in filament_tuples(filaments_of(trimmed)):
        assert filament[0] >= -1e-9
        assert filament[3] >= -1e-9


def test_footer_and_z_lift_dropped():
    trimmed, _ = trim_text(TRICKY_GCODE, n_layers=2)
    assert "Z50" not in trimmed
    assert trimmed.rstrip().endswith("; final retract")
    instruction = filaments_of(trimmed)
    assert instruction.coordinate_limits["z"][1] <= 0.4 + 1e-9


def test_height_selection():
    trimmed, stats = trim_text(TRICKY_GCODE, height=0.2)
    assert stats["layers_kept"] == 2
    assert stats["z_cut"] == pytest.approx(0.6)


def test_n_layers_larger_than_available_keeps_everything():
    trimmed, stats = trim_text(TRICKY_GCODE, n_layers=99)
    assert stats["layers_kept"] == 4
    assert stats["z_shift"] == 0.0
    original = filament_tuples(filaments_of(TRICKY_GCODE))
    kept = filament_tuples(filaments_of(trimmed))
    assert kept == original


def test_no_shift_keeps_original_heights():
    trimmed, stats = trim_text(TRICKY_GCODE, n_layers=2, shift_z=False)
    assert stats["z_shift"] == 0.0
    instruction = filaments_of(trimmed)
    assert instruction.coordinate_limits["z"][1] == pytest.approx(0.8)


def test_no_extrusion_raises():
    with pytest.raises(ValueError, match="no extrusion"):
        trim_text("G1 X10\nG1 Y20\n", n_layers=2)


def test_inch_units_raise():
    with pytest.raises(ValueError, match="G20"):
        trim_text("G20\nG1 X1 E1\n", n_layers=1)


def test_checksums_and_line_numbers_are_stripped():
    content = (
        HEADER.replace("G92 E0\n", "")
        + "N1 G1 Z0.2 F3000*12\n"
        + "N2 G1 X10 E1.0*34\n"
        + "N3 G1 Z0.4*56\n"
        + "N4 G1 X0 E2.0*78\n"
    )
    trimmed, _ = trim_text(content, n_layers=1)
    assert "*" not in trimmed
    kept_section = trimmed.split("; travel to trim point", 1)[1]
    for line in kept_section.splitlines():
        code = line.split(";", 1)[0].strip()
        assert not code.startswith(("N2", "N3", "N4"))
    kept = filament_tuples(filaments_of(trimmed))
    assert len(kept) == 1
    assert kept[0][5] == pytest.approx(0.2)  # Z 0.4 shifted down to 0.2


def test_example_gcode_equivalence():
    """Trim the shipped example file with every layer kept: no change at all."""
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples",
        "gcode_example.gcode",
    )
    with open(example_path) as handle:
        content = handle.read()
    trimmed, stats = trim_text(content, n_layers=99)
    assert stats["layers_total"] == 3  # Z 0.3, 0.5, 0.7 (last G1 F7200 is travel)
    assert math.isclose(stats["z_shift"], 0.0)
    assert filament_tuples(filaments_of(trimmed)) == filament_tuples(
        filaments_of(content)
    )
