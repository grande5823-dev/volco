from app.physics.acceleration.speed_profile import SpeedProfile

import numpy as np


class FlatSpeedProfile(SpeedProfile):
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
            end_hold = 0.0

        speed = self.calculate_speed_in_time(
            discrete_time=discrete_time,
            final_time=final_time,
            target_speed=target_speed,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
            end_hold=end_hold,
        )

        displacements_nozzle, total_length = self.calculate_integral_trapezoidal(
            discrete_time, speed
        )

        self._displacements = displacements_nozzle
        self._simulation_length = total_length

    def calculate_speed_in_time(
        self,
        discrete_time,
        final_time,
        target_speed,
        threshold_speed,
        acceleration,
        end_hold=0.0,
    ):
        """
        Evaluate the ideal constant velocity on a discrete time vector.

        For times beyond final_time, the velocity is held at `end_hold`
        (0.0 by default).
        """
        speed = np.full(discrete_time.size, target_speed)
        speed[discrete_time > final_time] = end_hold

        return speed

    @property
    def displacements(self):
        return self._displacements

    @property
    def simulation_length(self):
        return self._simulation_length
