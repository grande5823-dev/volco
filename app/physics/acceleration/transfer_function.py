import math

import numpy as np
from scipy import signal


class TransferFunction:
    """
    Continuous-time LTI transfer function used to filter ideal speed profiles.

    The transfer function must be stable, proper, free of integrators, and have
    unit DC gain so that the filtered trajectory covers the same distance as the
    ideal one.
    """

    DC_GAIN_TOLERANCE = 1e-3
    SETTLING_TOLERANCE = 1e-3

    def __init__(self, num, den):
        self.num = np.asarray(num, dtype=float)
        self.den = np.asarray(den, dtype=float)
        self._poles = None
        self._validate()

    @classmethod
    def from_config(cls, config):
        """
        Build a TransferFunction from a configuration dictionary.

        Supported formats:
        - {"type": "first_order", "time_constant": tau}
        - {"type": "second_order", "natural_frequency": wn, "damping": zeta}
        - {"type": "num_den", "num": [...], "den": [...]}
          (coefficients in descending powers of s)
        """
        tf_type = config.get("type")

        if tf_type == "first_order":
            time_constant = float(config["time_constant"])
            if time_constant <= 0:
                raise ValueError("time_constant must be positive")
            return cls(num=[1.0], den=[time_constant, 1.0])

        if tf_type == "second_order":
            natural_frequency = float(config["natural_frequency"])
            damping = float(config["damping"])
            if natural_frequency <= 0:
                raise ValueError("natural_frequency must be positive")
            if damping <= 0:
                raise ValueError("damping must be positive")
            return cls(
                num=[natural_frequency**2],
                den=[1.0, 2.0 * damping * natural_frequency, natural_frequency**2],
            )

        if tf_type == "num_den":
            return cls(num=config["num"], den=config["den"])

        raise ValueError(
            f"Unknown transfer function type '{tf_type}'. "
            "Supported types: 'first_order', 'second_order', 'num_den'."
        )

    @property
    def dc_gain(self):
        return self.num[-1] / self.den[-1]

    @property
    def settling_time(self):
        """
        Estimate of the time the filter takes to settle back to steady state
        after the input stops changing. Used to extend the evaluation window
        of filtered speed profiles.
        """
        slowest_pole = max(pole.real for pole in self.poles)
        return -math.log(self.SETTLING_TOLERANCE) / abs(slowest_pole)

    @property
    def poles(self):
        if self._poles is None:
            self._poles = np.roots(self.den)
        return self._poles

    def apply(self, values, discrete_time, steady_input):
        """
        Causally filter `values` sampled on `discrete_time`.

        The filter state is initialized to the steady state corresponding to a
        constant input of `steady_input`, modelling that the input was held at
        this value before the start of the time vector.

        The continuous-time transfer function is discretized at the sampling
        rate of the time vector (zero-order hold, exact for step and ramp
        inputs) and applied with lfilter; this matches scipy.signal.lsim to
        within machine precision at the simulation's sampling rates while
        being orders of magnitude faster.
        """
        dt = discrete_time[1] - discrete_time[0]

        num_discrete, den_discrete, _ = signal.cont2discrete(
            (self.num, self.den), dt, method="zoh"
        )
        b = np.asarray(num_discrete).ravel()
        a = np.asarray(den_discrete).ravel()

        # Steady-state initial conditions for a constant input of steady_input
        initial_state = signal.lfilter_zi(b, a) * steady_input

        filtered_values, _ = signal.lfilter(b, a, values, zi=initial_state)

        return filtered_values

    def _validate(self):
        if self.den.size == 0 or self.num.size == 0:
            raise ValueError("Transfer function numerator and denominator must be non-empty")

        if self.den[0] == 0:
            raise ValueError("Transfer function denominator leading coefficient must be non-zero")

        if self.den.size < self.num.size:
            raise ValueError("Transfer function must be proper (order of denominator >= order of numerator)")

        if self.den[-1] == 0:
            raise ValueError("Transfer function must not contain an integrator (poles at the origin)")

        if any(pole.real >= 0 for pole in self.poles):
            raise ValueError("Transfer function must be stable (all poles must have negative real parts)")

        if abs(self.dc_gain - 1.0) > self.DC_GAIN_TOLERANCE:
            raise ValueError(
                f"Transfer function must have unit DC gain to conserve distance, got {self.dc_gain}"
            )
