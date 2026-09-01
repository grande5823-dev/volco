import math

import pytest

from app.physics.acceleration.extruder_speed import ExtruderSpeed


class MockPrinter:
    def __init__(self, feedstock_filament_diameter=1.75):
        self.feedstock_filament_diameter = feedstock_filament_diameter


class TestExtruderSpeed:
    def test_should_calculate_the_speed_profile(self):
        threshold_speed = 50
        acceleration = 100.0

        extrusion_length = 100

        filament_diameter = 1.75
        filament_area = math.pi * (filament_diameter / 2) ** 2
        volume = extrusion_length * filament_area

        extruder_speed = ExtruderSpeed(
            volume=volume,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
            total_time=1.25,
            printer=MockPrinter(feedstock_filament_diameter=filament_diameter),
        )
        extruder_speed.calculate_displacements()

        assert extruder_speed.target_speed == pytest.approx(100.0)

        assert extruder_speed.speed_profile.displacements[-1] == pytest.approx(
            100.0, abs=1e-3
        )
