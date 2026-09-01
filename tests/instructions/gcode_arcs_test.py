import math

import numpy as np
import pytest

from app.instructions.gcode import Gcode


class MockPrinter:
    def __init__(self, feedstock_filament_diameter=1.75):
        self.feedstock_filament_diameter = feedstock_filament_diameter


def parse(lines, **kwargs):
    gcode = Gcode(gcode_content="\n".join(lines), printer=MockPrinter(), **kwargs)
    gcode.read()
    return gcode


class TestArcTessellation:
    def test_clockwise_quarter_arc_ij_form(self):
        # CW quarter circle of radius 10 from (0,0) to (10,10), centre (10,0)
        gcode = parse(
            [
                "G90", "M82",
                "G1 X0 Y0 F1200",
                "G2 X10 Y10 I10 J0 E1.0 F1200",
            ]
        )

        arc_moves = gcode.movements[2:]  # skip origin and positioning move
        assert len(arc_moves) == math.ceil(
            (math.pi * 0.5 * 10.0) / gcode.arc_segment_length
        )

        # exact endpoint
        assert arc_moves[-1][0] == pytest.approx(10.0, abs=1e-9)
        assert arc_moves[-1][1] == pytest.approx(10.0, abs=1e-9)

        # all tessellation points lie on the circle around (10, 0)
        for move in arc_moves:
            radius = math.hypot(move[0] - 10.0, move[1] - 0.0)
            assert radius == pytest.approx(10.0, abs=1e-9)

        # arc stays in the quadrant x <= 10, y >= 0 (CW sweep, not long way round)
        assert max(move[0] for move in arc_moves) <= 10.0 + 1e-9
        assert min(move[1] for move in arc_moves) >= -1e-9

        # extrusion sums to the commanded delta
        assert sum(move[3] for move in arc_moves) == pytest.approx(1.0)

    def test_counterclockwise_arc_takes_the_short_way(self):
        # Same endpoints as above but CCW (G3) must take the long way round
        # (3/4 circle): more segments and it visits negative y
        gcode = parse(
            [
                "G90", "M82",
                "G1 X0 Y0 F1200",
                "G3 X10 Y10 I10 J0 E1.0 F1200",
            ]
        )

        arc_moves = gcode.movements[2:]
        y_values = [move[1] for move in arc_moves]
        assert min(y_values) < -5.0  # dips below: the long way round
        assert arc_moves[-1][0] == pytest.approx(10.0, abs=1e-9)
        assert arc_moves[-1][1] == pytest.approx(10.0, abs=1e-9)

    def test_full_circle_ij_form(self):
        gcode = parse(
            [
                "G90", "M82",
                "G1 X0 Y0 F1200",
                "G2 I2 J0 E0.6 F1200",  # full circle, radius 2, back to start
            ]
        )

        arc_moves = gcode.movements[2:]
        assert len(arc_moves) == math.ceil(
            2 * math.pi * 2.0 / gcode.arc_segment_length
        )
        assert arc_moves[-1][0] == pytest.approx(0.0, abs=1e-9)
        assert arc_moves[-1][1] == pytest.approx(0.0, abs=1e-9)
        # reaches the far side of the circle (centre (2, 0), radius 2)
        assert max(move[0] for move in arc_moves) > 3.9

    def test_radius_form_minor_and_major_arc(self):
        minor = parse(
            ["G90", "M82", "G1 X0 Y0 F1200", "G2 X10 Y0 R10 E1.0"]
        )
        major = parse(
            ["G90", "M82", "G1 X0 Y0 F1200", "G2 X10 Y0 R-10 E1.0"]
        )

        minor_moves = minor.movements[2:]
        major_moves = major.movements[2:]

        # both end exactly at the endpoint
        assert minor_moves[-1][0] == pytest.approx(10.0, abs=1e-9)
        assert major_moves[-1][0] == pytest.approx(10.0, abs=1e-9)

        # major arc (sweep > pi) has more segments than the minor one
        assert len(major_moves) > len(minor_moves)

        # analytic centres: chord midpoint (5, 0) +/- height h = sqrt(10^2-5^2)
        height = math.sqrt(100.0 - 25.0)
        # minor CW arc dips below the chord (centre below); major bulges up
        for move in minor_moves:
            assert math.hypot(move[0] - 5.0, move[1] + height) == pytest.approx(
                10.0, abs=1e-9
            )
        for move in major_moves:
            assert math.hypot(move[0] - 5.0, move[1] - height) == pytest.approx(
                10.0, abs=1e-9
            )

    def test_extrusion_reference_after_arc_in_absolute_mode(self):
        # Regression: the arc's extrusion must not leak into the next move
        gcode = parse(
            [
                "G90", "M82",
                "G1 X0 Y0 E0.5 F1200",
                "G2 X10 Y10 I10 J0 E1.0",   # delta 0.5 over the arc
                "G1 X20 Y10 E1.5",           # delta 0.5 on this line
            ]
        )

        assert gcode.movements[-1][3] == pytest.approx(0.5)

    def test_extrusion_reference_relative_mode(self):
        gcode = parse(
            [
                "G90", "M83",
                "G1 X0 Y0 E0.5 F1200",
                "G3 X10 Y10 I10 J0 E0.25",   # relative delta 0.25
                "G1 X20 Y10 E0.5",
            ]
        )

        arc_moves = gcode.movements[2:-1]
        assert sum(move[3] for move in arc_moves) == pytest.approx(0.25)
        assert gcode.movements[-1][3] == pytest.approx(0.5)

    def test_helical_arc_interpolates_z(self):
        gcode = parse(
            [
                "G90", "M82",
                "G1 X0 Y0 Z1.0 F1200",
                "G2 X10 Y10 Z2.0 I10 J0 E1.0",
            ]
        )

        arc_moves = gcode.movements[2:]
        assert arc_moves[0][2] > 1.0
        assert arc_moves[-1][2] == pytest.approx(2.0)

        z_deltas = np.diff([move[2] for move in arc_moves])
        assert np.allclose(z_deltas, z_deltas[0])  # linear in arc length

    def test_arc_resolution_is_configurable(self):
        base = dict(gcode_content="\n".join(
            ["G90", "M82", "G1 X0 Y0 F1200", "G2 X10 Y10 I10 J0 E1.0"]
        ), printer=MockPrinter())

        fine = Gcode(arc_segment_length=0.05, **base)
        fine.read()
        coarse = Gcode(arc_segment_length=0.5, **base)
        coarse.read()

        assert len(fine.movements) > len(coarse.movements)

    def test_malformed_radius_is_skipped_with_warning_and_e_consumed(self, caplog):
        with caplog.at_level("WARNING", logger="app.instructions.gcode"):
            gcode = parse(
                [
                    "G90", "M82",
                    "G1 X0 Y0 F1200",
                    "G2 X10 Y0 R2 E1.0",   # chord 10 > 2*R: impossible
                    "G1 X20 Y0 E1.5",
                ]
            )

        assert "malformed arc" in caplog.text
        # no movement added for the skipped arc
        assert gcode.movements[1][0:3] == [0.0, 0.0, 0.0]
        # E reference still consumed: delta on the next line is 0.5, not 1.5
        assert gcode.movements[2][0:2] == [20.0, 0.0]
        assert gcode.movements[2][3] == pytest.approx(0.5)

    def test_arc_without_centre_or_radius_is_skipped_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="app.instructions.gcode"):
            gcode = parse(
                [
                    "G90", "M82",
                    "G1 X0 Y0 F1200",
                    "G2 X10 Y10 E1.0",      # no I/J/R -> skipped
                    "G1 X20 Y0 E1.5",       # extruding move -> filament exists
                ]
            )

        assert "malformed arc" in caplog.text
        # no movement recorded for the skipped arc
        assert len(gcode.movements) == 3
        # E reference still consumed
        assert gcode.movements[-1][3] == pytest.approx(0.5)

    def test_file_without_any_extrusion_move_raises_a_clear_error(self):
        with pytest.raises(ValueError, match="no extrusion moves"):
            parse(["G90", "G1 X10 Y10 F1200"])

    def test_arcs_in_non_xy_planes_are_skipped_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="app.instructions.gcode"):
            gcode = parse(
                [
                    "G90", "M82", "G18",
                    "G1 X0 Y0 E0.5 F1200",
                    "G2 X10 Z2 I5 E1.0",
                    "G1 X20 Y0 E1.5",
                ]
            )

        assert "unsupported plane" in caplog.text
        assert gcode.movements[-1][3] == pytest.approx(0.5)

    def test_tessellated_movement_chain_is_continuous(self):
        # The reported symptom: the G1 after an arc used to resume from the
        # pre-arc position (a discontinuous jump skipping the whole arc).
        gcode = parse(
            [
                "G90", "M82",
                "G1 X10 Y10 F1200",
                "G2 X10 Y20 I0 J5 E0.5",     # arc ends at (10, 20)
                "G1 X30 Y20 E1.0 F1200",     # must start from (10, 20)
            ]
        )

        # distance between consecutive recorded positions never jumps:
        # bounded by the tessellation resolution or the move's own length
        positions = np.array([move[0:2] for move in gcode.movements])
        gaps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        # last gap is the final G1 (length 20); all arc-internal gaps <= resolution
        assert np.all(gaps[1:-1] <= gcode.arc_segment_length * (1 + 1e-9))
        assert gcode.movements[-2][0:2] == pytest.approx([10.0, 20.0], abs=1e-6)
