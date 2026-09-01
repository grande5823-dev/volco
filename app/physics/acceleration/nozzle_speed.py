from app.physics.acceleration.filtered_speed_profile import FilteredSpeedProfile
from app.physics.acceleration.flat_speed_profile import FlatSpeedProfile
from app.physics.acceleration.speed import Speed
from app.physics.acceleration.trapezoidal_speed_profile import TrapezoidalSpeedProfile

import numpy as np


class NozzleSpeed(Speed):
    TIME_GRID_POINTS = int(1e5)
    MAX_TIME_GRID_POINTS = int(5e5)

    def __init__(
        self,
        filament_length,
        target_speed,
        threshold_speed,
        acceleration,
        motion_filter=None,
    ):
        self._travel_length = filament_length
        self._target_speed = target_speed
        self._threshold_speed = threshold_speed
        self._acceleration = acceleration
        self._motion_filter = motion_filter
        self._total_time = None
        self._speed_profile = None
        self._settling_time = 0.0
        self._end_time = None

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

    def build_time_grid(self, additional_settling_time=0.0):
        """
        Build the discrete time vector shared by the nozzle and extruder
        profiles. When a motion filter is configured (or additional settling
        time is requested, e.g. for a lagging extruder), the window is extended
        so the filtered trajectories can settle, keeping the same time step as
        the unfiltered grid.
        """
        self._find_total_time_and_speed_profile()

        settling_time = max(self._settling_time, additional_settling_time)
        self._end_time = self._total_time + settling_time

        number_points = self.TIME_GRID_POINTS
        if settling_time > 0.0:
            number_points = int(
                round(self.TIME_GRID_POINTS * self._end_time / self._total_time)
            )
            if number_points > self.MAX_TIME_GRID_POINTS:
                # Very short segments (e.g. arc tessellation) have durations
                # far below the filter's settling time; keeping the nominal
                # time step would explode the grid. Coarsen the step instead —
                # it remains far finer than the filter dynamics.
                number_points = self.MAX_TIME_GRID_POINTS

        return np.linspace(0, self._end_time, num=number_points)

    def calculate_displacements(self, discrete_time=None):
        self._find_total_time_and_speed_profile()

        if discrete_time is None:
            discrete_time = self.build_time_grid()

        # On a time window extending beyond final_time (settling tails), an
        # unfiltered ideal trajectory is held at its segment-end velocity,
        # consistent with the jerk boundary conditions. Filtered profiles
        # define their own boundary hold and ignore this argument.
        end_hold = None
        if not isinstance(self.speed_profile, FilteredSpeedProfile):
            if discrete_time[-1] > self.total_time:
                end_hold = self._segment_end_speed()

        self.speed_profile.calculate_displacements_in_time(
            discrete_time=discrete_time,
            final_time=self.total_time,
            target_speed=self.target_speed,
            threshold_speed=self.threshold_speed,
            acceleration=self.acceleration,
            end_hold=end_hold,
        )

    def _segment_end_speed(self):
        if isinstance(self.speed_profile, FlatSpeedProfile):
            return self.target_speed

        return self.threshold_speed

    def _find_total_time_and_speed_profile(self):
        if self._total_time is not None:
            return

        if self.target_speed < self.threshold_speed:
            self._total_time = self.travel_length / self.target_speed
            self._speed_profile = FlatSpeedProfile()
        else:
            self._total_time = (
                self.target_speed / self.acceleration
                + self.travel_length / self.target_speed
                + self.threshold_speed**2 / self.target_speed / self.acceleration
                - 2 * self.threshold_speed / self.acceleration
            )

            ideal_profile = TrapezoidalSpeedProfile()

            if self._motion_filter is not None:
                self._speed_profile = FilteredSpeedProfile(
                    ideal_profile=ideal_profile,
                    transfer_function=self._motion_filter,
                    boundary_speed=self.threshold_speed,
                )
                self._settling_time = self._motion_filter.settling_time
            else:
                self._speed_profile = ideal_profile

        self._end_time = self._total_time + self._settling_time
        return
