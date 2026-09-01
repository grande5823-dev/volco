import math

import numpy as np
import pytest

from app.physics.acceleration.acceleration_profiles import AccelerationManager
from app.physics.acceleration.extruder_speed import ExtruderSpeed
from app.physics.acceleration.filtered_speed_profile import FilteredSpeedProfile
from app.physics.acceleration.flat_speed_profile import FlatSpeedProfile
from app.physics.acceleration.nozzle_speed import NozzleSpeed
from app.physics.acceleration.transfer_function import TransferFunction
from app.physics.acceleration.trapezoidal_speed_profile import TrapezoidalSpeedProfile
from app.physics.acceleration.volume import AccelerationVolume


class MockPrinter:
    def __init__(self, nozzle_motion_filter=None, extruder_motion_filter=None):
        self.nozzle_jerk_speed = 8.0
        self.extruder_jerk_speed = 5.0
        self.nozzle_acceleration = 500.0
        self.extruder_acceleration = 1000.0
        self.feedstock_filament_diameter = 1.75
        self.nozzle_diameter = 0.4
        self.nozzle_motion_filter = nozzle_motion_filter
        self.extruder_motion_filter = extruder_motion_filter


FILAMENT_LENGTH = 50.0
PRINTING_SPEED = 40.0
EXTRUSION_LENGTH = 3.0
STEPS = 10


def compute_step_volumes(printer):
    filament_area = math.pi * (printer.feedstock_filament_diameter / 2) ** 2
    volume = EXTRUSION_LENGTH * filament_area

    nozzle_speed, extruder_speed = AccelerationManager.calculate_speed_profiles(
        filament_length=FILAMENT_LENGTH,
        volume=volume,
        printing_speed=PRINTING_SPEED,
        printer=printer,
    )

    volumes = AccelerationVolume.calculate_volumes_with_acceleration(
        number_simulation_steps=STEPS,
        step_size=FILAMENT_LENGTH / STEPS,
        feedstock_filament_diameter=printer.feedstock_filament_diameter,
        nozzle_profile=nozzle_speed,
        extruder_profile=extruder_speed,
    )

    return volume, nozzle_speed, extruder_speed, volumes


class TestDecoupledExtruderProfile:
    def test_ideal_extrusion_should_be_scaled_ideal_motion(self):
        extrusion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.02}
        )
        printer = MockPrinter(extruder_motion_filter=extrusion_filter)

        _, _, extruder_speed, _ = compute_step_volumes(printer)

        scale = EXTRUSION_LENGTH / FILAMENT_LENGTH
        assert extruder_speed.target_speed == pytest.approx(scale * PRINTING_SPEED)
        assert extruder_speed.threshold_speed == pytest.approx(
            scale * printer.nozzle_jerk_speed
        )
        assert extruder_speed.acceleration == pytest.approx(
            scale * printer.nozzle_acceleration
        )

    def test_ideal_extrusion_should_integrate_to_gcode_volume(self):
        extrusion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.02}
        )
        printer = MockPrinter(extruder_motion_filter=extrusion_filter)

        volume, nozzle_speed, extruder_speed, _ = compute_step_volumes(printer)

        # Integrate the unfiltered ideal trajectory of the decoupled extruder
        ideal = TrapezoidalSpeedProfile()
        discrete_time = np.linspace(0, nozzle_speed.total_time, num=int(1e5))
        ideal.calculate_displacements_in_time(
            discrete_time=discrete_time,
            final_time=nozzle_speed.total_time,
            target_speed=extruder_speed.target_speed,
            threshold_speed=extruder_speed.threshold_speed,
            acceleration=extruder_speed.acceleration,
        )

        assert ideal.simulation_length == pytest.approx(
            EXTRUSION_LENGTH, rel=1e-9
        )

    def test_profile_should_be_filtered_trapezoid(self):
        extrusion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.02}
        )
        printer = MockPrinter(extruder_motion_filter=extrusion_filter)

        _, _, extruder_speed, _ = compute_step_volumes(printer)

        assert isinstance(extruder_speed.speed_profile, FilteredSpeedProfile)
        assert (
            extruder_speed.speed_profile.transfer_function is extrusion_filter
        )

    def test_flat_ideal_motion_should_yield_flat_extrusion(self):
        extrusion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.02}
        )
        printer = MockPrinter(extruder_motion_filter=extrusion_filter)

        # Printing speed below jerk speed -> flat ideal motion
        filament_area = math.pi * (printer.feedstock_filament_diameter / 2) ** 2
        volume = EXTRUSION_LENGTH * filament_area

        _, extruder_speed = AccelerationManager.calculate_speed_profiles(
            filament_length=FILAMENT_LENGTH,
            volume=volume,
            printing_speed=1.0,
            printer=printer,
        )

        assert isinstance(extruder_speed.speed_profile, FlatSpeedProfile)

    def test_decoupled_mode_requires_nozzle_reference(self):
        printer = MockPrinter()
        extrusion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.02}
        )

        filament_area = math.pi * (printer.feedstock_filament_diameter / 2) ** 2
        extruder_speed = ExtruderSpeed(
            volume=EXTRUSION_LENGTH * filament_area,
            threshold_speed=printer.extruder_jerk_speed,
            acceleration=printer.extruder_acceleration,
            total_time=1.0,
            printer=printer,
            extrusion_filter=extrusion_filter,
        )

        with pytest.raises(ValueError, match="nozzle_speed"):
            extruder_speed.calculate_displacements()


class TestDecoupledExtrusionPhysics:
    def test_lag_should_redistribute_volume_towards_segment_end(self):
        extrusion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.02}
        )
        printer = MockPrinter(extruder_motion_filter=extrusion_filter)

        volume, _, _, volumes = compute_step_volumes(printer)

        uniform = volume / STEPS

        # Under-extrusion at the start, over-extrusion at the end
        assert volumes[0] < uniform
        assert volumes[-1] - volumes[-2] > uniform

        # Cumulative volume at segment midpoint below the uniform share
        assert volumes[STEPS // 2 - 1] < 0.5 * volume

    def test_ooze_tail_should_not_be_deposited_when_nozzle_is_ideal(self):
        # With an ideal (unfiltered) nozzle, the nozzle reaches the segment end
        # exactly at final_time; the extruder is still ringing down afterwards,
        # and that ooze is extruded beyond the segment and not deposited.
        extrusion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.02}
        )
        printer = MockPrinter(extruder_motion_filter=extrusion_filter)

        volume, _, _, volumes = compute_step_volumes(printer)

        assert volumes[-1] < volume
        assert volumes[-1] > 0.99 * volume

    def test_volume_should_accumulate_when_nozzle_lags_more_than_extruder(self):
        # When the nozzle lags its ideal trajectory more than the extruder
        # does, the nozzle spends longer inside the segment while the extruder
        # feeds at near-ideal rate: more than the g-code volume is deposited.
        printer = MockPrinter(
            nozzle_motion_filter=TransferFunction.from_config(
                {"type": "first_order", "time_constant": 0.05}
            ),
            extruder_motion_filter=TransferFunction.from_config(
                {"type": "first_order", "time_constant": 0.005}
            ),
        )

        volume, _, _, volumes = compute_step_volumes(printer)

        assert volumes[-1] > volume
        assert volumes[-1] < 1.02 * volume

    def test_legacy_coupled_mode_should_be_unchanged_without_filter(self):
        printer = MockPrinter()

        volume, _, extruder_speed, volumes = compute_step_volumes(printer)

        assert isinstance(
            extruder_speed.speed_profile, (FlatSpeedProfile, TrapezoidalSpeedProfile)
        )
        assert volumes[-1] == pytest.approx(volume, rel=1e-9)
