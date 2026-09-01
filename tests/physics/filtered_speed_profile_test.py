import math

import numpy as np
import pytest

from app.physics.acceleration.acceleration_profiles import AccelerationManager
from app.physics.acceleration.filtered_speed_profile import FilteredSpeedProfile
from app.physics.acceleration.flat_speed_profile import FlatSpeedProfile
from app.physics.acceleration.nozzle_speed import NozzleSpeed
from app.physics.acceleration.transfer_function import TransferFunction
from app.physics.acceleration.trapezoidal_speed_profile import TrapezoidalSpeedProfile
from app.physics.acceleration.volume import AccelerationVolume

TARGET_SPEED = 100
THRESHOLD_SPEED = 50
ACCELERATION = 100.0
FILAMENT_LENGTH = 100.0
TOTAL_TIME = 1.25  # analytic trapezoid duration for the parameters above


class MockPrinter:
    def __init__(self, nozzle_motion_filter=None):
        self.nozzle_jerk_speed = 8.0
        self.extruder_jerk_speed = 5.0
        self.nozzle_acceleration = 500.0
        self.extruder_acceleration = 1000.0
        self.feedstock_filament_diameter = 1.75
        self.nozzle_diameter = 0.4
        self.nozzle_motion_filter = nozzle_motion_filter


def make_filtered_nozzle(motion_filter):
    nozzle_speed = NozzleSpeed(
        filament_length=FILAMENT_LENGTH,
        target_speed=TARGET_SPEED,
        threshold_speed=THRESHOLD_SPEED,
        acceleration=ACCELERATION,
        motion_filter=motion_filter,
    )
    discrete_time = nozzle_speed.build_time_grid()
    nozzle_speed.calculate_displacements(discrete_time)
    return nozzle_speed, discrete_time


class TestFilteredSpeedProfile:
    def test_should_wrap_trapezoidal_profile_when_filter_is_configured(self):
        motion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.05}
        )

        nozzle_speed, _ = make_filtered_nozzle(motion_filter)

        assert isinstance(nozzle_speed.speed_profile, FilteredSpeedProfile)

    def test_should_extend_time_grid_by_settling_time(self):
        motion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.05}
        )

        nozzle_speed, discrete_time = make_filtered_nozzle(motion_filter)

        expected_end_time = TOTAL_TIME + motion_filter.settling_time
        assert discrete_time[-1] == pytest.approx(expected_end_time)
        assert discrete_time.size > int(1e5)

    def test_without_filter_should_keep_legacy_grid_and_profile(self):
        nozzle_speed, discrete_time = make_filtered_nozzle(motion_filter=None)

        assert isinstance(nozzle_speed.speed_profile, TrapezoidalSpeedProfile)
        assert discrete_time[-1] == pytest.approx(TOTAL_TIME)
        assert discrete_time.size == int(1e5)
        assert nozzle_speed.speed_profile.displacements[-1] == pytest.approx(
            FILAMENT_LENGTH, abs=1e-3
        )

    def test_flat_profile_should_not_be_filtered(self):
        motion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.05}
        )

        nozzle_speed = NozzleSpeed(
            filament_length=FILAMENT_LENGTH,
            target_speed=10,
            threshold_speed=THRESHOLD_SPEED,
            acceleration=ACCELERATION,
            motion_filter=motion_filter,
        )
        discrete_time = nozzle_speed.build_time_grid()

        assert isinstance(nozzle_speed.speed_profile, FlatSpeedProfile)
        assert discrete_time[-1] == pytest.approx(FILAMENT_LENGTH / 10)

    def test_filtered_displacement_should_be_monotone_and_conserve_net_deviation(self):
        motion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.05}
        )

        nozzle_speed, discrete_time = make_filtered_nozzle(motion_filter)

        displacements = nozzle_speed.speed_profile.displacements
        assert np.all(np.diff(displacements) > 0.0)

        # Ideal trajectory continued at the jerk speed over the same window
        ideal_speed = TrapezoidalSpeedProfile().calculate_speed_in_time(
            discrete_time=discrete_time,
            final_time=TOTAL_TIME,
            target_speed=TARGET_SPEED,
            threshold_speed=THRESHOLD_SPEED,
            acceleration=ACCELERATION,
            end_hold=THRESHOLD_SPEED,
        )
        ideal_displacement = np.zeros(discrete_time.size)
        ideal_displacement[1:] = np.cumsum(
            (ideal_speed[1:] + ideal_speed[:-1])
            * np.diff(discrete_time)
            * 0.5
        )

        net_deviation = displacements[-1] - ideal_displacement[-1]
        assert abs(net_deviation) < 1e-2

    def test_filtered_nozzle_should_take_longer_to_cover_the_filament(self):
        motion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.05}
        )

        nozzle_speed, discrete_time = make_filtered_nozzle(motion_filter)

        displacements = nozzle_speed.speed_profile.displacements
        reach_index = np.searchsorted(displacements, FILAMENT_LENGTH)
        time_to_cover_filament = discrete_time[reach_index]

        assert time_to_cover_filament > TOTAL_TIME

    def test_reverse_motion_guard_should_clip_negative_speeds(self):
        motion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.05}
        )
        profile = FilteredSpeedProfile(
            ideal_profile=TrapezoidalSpeedProfile(),
            transfer_function=motion_filter,
            boundary_speed=THRESHOLD_SPEED,
        )

        speed = np.array([10.0, -2.0, 5.0])
        guarded = profile._guard_against_reverse_motion(speed, TARGET_SPEED)

        assert np.all(guarded >= 0.0)


class TestAccelerationManagerWithFilter:
    def test_should_conserve_extruded_volume_when_nozzle_profile_is_filtered(self):
        motion_filter = TransferFunction.from_config(
            {"type": "first_order", "time_constant": 0.01}
        )
        printer = MockPrinter(nozzle_motion_filter=motion_filter)

        filament_length = 50.0
        printing_speed = 40.0
        extrusion_length = 3.0
        filament_area = math.pi * (printer.feedstock_filament_diameter / 2) ** 2
        volume = extrusion_length * filament_area

        nozzle_speed, extruder_speed = AccelerationManager.calculate_speed_profiles(
            filament_length=filament_length,
            volume=volume,
            printing_speed=printing_speed,
            printer=printer,
        )

        assert isinstance(nozzle_speed.speed_profile, FilteredSpeedProfile)

        number_simulation_steps = 10
        volumes = AccelerationVolume.calculate_volumes_with_acceleration(
            number_simulation_steps=number_simulation_steps,
            step_size=filament_length / number_simulation_steps,
            feedstock_filament_diameter=printer.feedstock_filament_diameter,
            nozzle_profile=nozzle_speed,
            extruder_profile=extruder_speed,
        )

        # Tolerance accounts for trapezoidal-integration noise at the single
        # grid cell straddling final_time in the extended time window.
        assert volumes[-1] == pytest.approx(volume, rel=1e-5)
