from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class SpeedProfile(ABC):
    @abstractmethod
    def calculate_displacements_in_time(
        self,
        discrete_time,
        final_time,
        target_speed,
        threshold_speed,
        acceleration,
        end_hold=None,
    ):
        """
        Compute the displacement profile on a discrete time vector.

        `end_hold` is the velocity the ideal trajectory is held at for times
        beyond final_time when the time vector extends past it. None selects
        the profile's default (0.0 for ideal profiles; the boundary speed for
        filtered profiles).
        """
        pass

    def calculate_integral_trapezoidal(self, x, y):
        # Numerically identical to the previous sequential loop: numpy's
        # cumsum performs the additions in the same left-to-right order.
        terms = (y[1:] + y[:-1]) * (x[1:] - x[:-1]) * 0.5

        intg_vec = np.empty(len(x))
        intg_vec[0] = 0.0
        if len(x) > 1:
            np.cumsum(terms, out=intg_vec[1:])

        return intg_vec, intg_vec[-1]

    @abstractproperty
    def displacements(self):
        pass

    @abstractproperty
    def simulation_length(self):
        pass
