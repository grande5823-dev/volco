import logging

import numpy as np

from app.physics.acceleration.speed_profile import SpeedProfile

logger = logging.getLogger(__name__)


class FilteredSpeedProfile(SpeedProfile):
    """
    Speed profile obtained by filtering an ideal speed profile (trapezoidal or
    flat) through a continuous-time transfer function, modelling the limited
    bandwidth of the motion system.

    The ideal trajectory is assumed to continue at `boundary_speed` (typically
    the jerk speed) before and after the segment, so the filtered trajectory
    starts and ends at steady state. Since the transfer function has unit DC
    gain, the net deviation between ideal and filtered displacement is zero.
    """

    def __init__(self, ideal_profile, transfer_function, boundary_speed):
        self._ideal_profile = ideal_profile
        self._transfer_function = transfer_function
        self._boundary_speed = boundary_speed

    @property
    def transfer_function(self):
        return self._transfer_function

    def calculate_displacements_in_time(
        self,
        discrete_time,
        final_time,
        target_speed,
        threshold_speed,
        acceleration,
        end_hold=None,
    ):
        if end_hold is None:
            end_hold = self._boundary_speed

        ideal_speed = self._ideal_profile.calculate_speed_in_time(
            discrete_time=discrete_time,
            final_time=final_time,
            target_speed=target_speed,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
            end_hold=end_hold,
        )

        filtered_speed = self._transfer_function.apply(
            ideal_speed, discrete_time, steady_input=self._boundary_speed
        )

        filtered_speed = self._guard_against_reverse_motion(
            filtered_speed, target_speed
        )

        displacements, total_length = self.calculate_integral_trapezoidal(
            discrete_time, filtered_speed
        )

        self._displacements = displacements
        self._simulation_length = total_length

    @property
    def displacements(self):
        return self._displacements

    @property
    def simulation_length(self):
        return self._simulation_length

    def _guard_against_reverse_motion(self, filtered_speed, target_speed):
        min_speed = float(np.min(filtered_speed))

        if min_speed < 0.0:
            if min_speed < -1e-6 * max(1.0, target_speed):
                logger.warning(
                    "Filtered speed profile becomes negative (min = %.6f mm/s). "
                    "Negative values are clipped to zero to keep the displacement "
                    "monotone. Consider a less aggressive transfer function.",
                    min_speed,
                )
            filtered_speed = np.clip(filtered_speed, 0.0, None)

        return filtered_speed
