import math

import pytest

from app.physics.acceleration.extruder_speed import ExtruderSpeed
from app.physics.acceleration.nozzle_speed import NozzleSpeed
from app.physics.acceleration.volume import AccelerationVolume


class MockPrinter:
    def __init__(self, feedstock_filament_diameter=1.75):
        self.feedstock_filament_diameter = feedstock_filament_diameter


class TestVolume:
    def test_should_calculate_volumes_with_acceleration(self):
        target_speed = 10
        threshold_speed = 40
        acceleration = 1200.0

        filament_length = 100.0

        extrusion_length = 10.0

        filament_diameter = 1.75
        filament_area = math.pi * (filament_diameter / 2) ** 2
        volume = extrusion_length * filament_area

        nozzle_speed = NozzleSpeed(
            filament_length=filament_length,
            target_speed=target_speed,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
        )
        nozzle_speed.calculate_displacements()

        extruder_speed = ExtruderSpeed(
            volume=volume,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
            total_time=nozzle_speed.total_time,
            printer=MockPrinter(feedstock_filament_diameter=filament_diameter),
        )
        extruder_speed.calculate_displacements()

        volumes = AccelerationVolume.calculate_volumes_with_acceleration(
            number_simulation_steps=5,
            step_size=20,
            feedstock_filament_diameter=filament_diameter,
            nozzle_profile=nozzle_speed,
            extruder_profile=extruder_speed,
        )

        expected_volume = extrusion_length * filament_area

        assert volumes[-1] == pytest.approx(expected_volume)
