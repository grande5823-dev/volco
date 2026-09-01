import pytest

from app.instructions.gcode import Gcode


class MockPrinter:
    def __init__(self, feedstock_filament_diameter=1.75):
        self.feedstock_filament_diameter = feedstock_filament_diameter


class TestGcode:
    def test_should_read_the_gcode(self):
        gcode = Gcode(
            gcode_path="tests/fixtures/gcode_example.gcode",
            default_nozzle_speed=40.0,
            printer=MockPrinter(),
        )

        gcode.read()

        assert gcode.number_printed_filaments == 3

        assert len(gcode.movements) == 8

        assert len(gcode.filaments_coordinates) == 3

        assert gcode.coordinate_limits["x"] == [10.0, 14.0]
        assert gcode.coordinate_limits["y"] == [8.0, 12.0]
        assert gcode.coordinate_limits["z"] == [0.0, 0.7]

    def test_should_tolerate_vendor_commands_with_non_numeric_arguments(self):
        gcode_content = "\n".join(
            [
                "G90",
                "M82",
                "M73 P0 R15",
                "M204 S10000 ; init ACC",
                "M1002 set_gcode_claim_speed_level : 5",
                "M1002 gcode_claim_action : 29",
                "M221 X0 Y0 Z0 ; turn off soft endstop",
                "G1 X10 Y10 F1200",
                "G1 E2.0",
                "M73.2   R1.0 ;Reset left time magnitude",
            ]
        )
        gcode = Gcode(gcode_content=gcode_content, printer=MockPrinter())

        gcode.read()

        assert gcode.movements[-1][:4] == [10.0, 10.0, 0.0, 2.0]

    def test_should_still_reject_malformed_motion_parameters(self):
        gcode_content = "G1 XBAD Y10"

        gcode = Gcode(gcode_content=gcode_content, printer=MockPrinter())

        with pytest.raises(ValueError, match="correct gcode format"):
            gcode.read()
