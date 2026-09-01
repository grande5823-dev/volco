from app.physics.acceleration.speed_profile import SpeedProfile

import numpy as np


class TrapezoidalSpeedProfile(SpeedProfile):
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

        displacements, total_length = self.calculate_integral_trapezoidal(
            discrete_time, speed
        )

        self._displacements = displacements
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
        Evaluate the ideal trapezoidal velocity on a discrete time vector.

        For times beyond final_time, the velocity is held at `end_hold`
        (0.0 by default). A filtered profile uses a non-zero end_hold to model
        the ideal trajectory continuing at the jerk speed into the next segment.
        """
        ascending_time = (target_speed - threshold_speed) / acceleration

        cruising_time = final_time - ascending_time

        # Vectorized equivalent of the previous branch loop; per-element
        # comparisons and arithmetic are unchanged.
        speed = np.empty(discrete_time.size)

        mask_ascending = discrete_time <= ascending_time
        mask_cruising = (~mask_ascending) & (discrete_time <= cruising_time)
        mask_descending = (
            (~mask_ascending) & (~mask_cruising) & (discrete_time <= final_time)
        )
        mask_hold = ~(mask_ascending | mask_cruising | mask_descending)

        speed[mask_ascending] = (
            acceleration * discrete_time[mask_ascending] + threshold_speed
        )
        speed[mask_cruising] = target_speed
        speed[mask_descending] = (
            -acceleration * discrete_time[mask_descending]
            + threshold_speed
            + acceleration * final_time
        )
        speed[mask_hold] = end_hold

        return speed

    @property
    def displacements(self):
        return self._displacements

    @property
    def simulation_length(self):
        return self._simulation_length
