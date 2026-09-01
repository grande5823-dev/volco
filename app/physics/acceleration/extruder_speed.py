from app.physics.acceleration.filtered_speed_profile import FilteredSpeedProfile
from app.physics.acceleration.flat_speed_profile import FlatSpeedProfile
from app.physics.acceleration.speed import Speed
from app.physics.acceleration.trapezoidal_speed_profile import TrapezoidalSpeedProfile

import numpy as np
import math


class ExtruderSpeed(Speed):
    def __init__(
        self,
        volume,
        threshold_speed,
        acceleration,
        total_time,
        printer,
        nozzle_speed=None,
        extrusion_filter=None,
    ):
        # Convert volume to extrusion length
        filament_area = math.pi * (printer.feedstock_filament_diameter/2)**2
        self._travel_length = volume / filament_area

        self._target_speed = None
        self._threshold_speed = threshold_speed
        self._acceleration = acceleration
        self._total_time = total_time
        self._speed_profile = None
        self._nozzle_speed = nozzle_speed
        self._extrusion_filter = extrusion_filter

    @property
    def travel_length(self):
        return self._travel_length

    @property
    def target_speed(self):
        return self._target_speed

    @property
    def threshold_speed(self):
        return self._threshold_speed

    @property
    def acceleration(self):
        return self._acceleration

    @property
    def total_time(self):
        return self._total_time

    @property
    def speed_profile(self):
        return self._speed_profile

    def calculate_displacements(self, discrete_time=None):
        self._find_target_speed_and_speed_profile()

        if discrete_time is None:
            length_time = int(1e5)
            discrete_time = np.linspace(0, self.total_time, num=length_time)

        time_vec = discrete_time

        self.speed_profile.calculate_displacements_in_time(
            discrete_time=time_vec,
            final_time=self.total_time,
            target_speed=self.target_speed,
            threshold_speed=self.threshold_speed,
            acceleration=self.acceleration,
        )

    def _find_target_speed_and_speed_profile(self):
        if self._extrusion_filter is not None:
            self._find_decoupled_speed_profile()
            return

        if self.travel_length / self.total_time < self.threshold_speed:
            self._speed_profile = FlatSpeedProfile()
            self._target_speed = self.travel_length / self.total_time
            return

        coeff = [
            1.0 / self.acceleration,
            -self.total_time - 2.0 * self.threshold_speed / self.acceleration,
            self.threshold_speed**2 / self.acceleration + self.travel_length,
        ]
        possible_target_speeds = np.roots(coeff)

        t1 = (possible_target_speeds - self.threshold_speed) / self.acceleration
        t2 = (
            self.total_time
            - 2.0 * possible_target_speeds / self.acceleration
            + 2.0 * self.threshold_speed / self.acceleration
            + t1
        )

        for (t1_i, t2_i, Ve_i) in zip(t1, t2, possible_target_speeds):
            if t2_i > t1_i and t1_i > 0 and t2_i > 0:
                self._target_speed = Ve_i

        self._speed_profile = TrapezoidalSpeedProfile()
        return

    def _find_decoupled_speed_profile(self):
        """
        Decoupled extrusion model: the ideal extrusion trajectory is derived
        from the ideal motion trajectory, scaled so that a constant volume per
        unit length is extruded, and then filtered with the configured transfer
        function to model the extrusion system dynamics (e.g. pressure lag).

        In this mode the extruder's own jerk speed and acceleration from the
        printer configuration are not used; the kinematic limits are inherited
        from the motion trajectory.
        """
        nozzle_speed = self._nozzle_speed
        if nozzle_speed is None:
            raise ValueError(
                "ExtruderSpeed in decoupled mode requires the nozzle_speed reference"
            )

        scale = self.travel_length / nozzle_speed.travel_length

        self._target_speed = scale * nozzle_speed.target_speed
        self._threshold_speed = scale * nozzle_speed.threshold_speed
        self._acceleration = scale * nozzle_speed.acceleration

        if nozzle_speed.target_speed < nozzle_speed.threshold_speed:
            # Flat ideal motion: the ideal extrusion is constant, so filtering
            # it with matching steady-state boundary conditions is the identity.
            self._speed_profile = FlatSpeedProfile()
            return

        self._speed_profile = FilteredSpeedProfile(
            ideal_profile=TrapezoidalSpeedProfile(),
            transfer_function=self._extrusion_filter,
            boundary_speed=self.threshold_speed,
        )
        return