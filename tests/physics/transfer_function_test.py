import numpy as np
import pytest
from scipy import signal

from app.physics.acceleration.transfer_function import TransferFunction


class TestTransferFunction:
    def test_first_order_should_have_unit_dc_gain_and_expected_denominator(self):
        tf = TransferFunction.from_config({"type": "first_order", "time_constant": 0.05})

        assert tf.dc_gain == pytest.approx(1.0)
        assert list(tf.den) == [0.05, 1.0]
        assert list(tf.num) == [1.0]

    def test_second_order_should_have_unit_dc_gain(self):
        tf = TransferFunction.from_config(
            {"type": "second_order", "natural_frequency": 50.0, "damping": 0.7}
        )

        assert tf.dc_gain == pytest.approx(1.0)

    def test_settling_time_should_match_slowest_pole(self):
        tf = TransferFunction.from_config({"type": "first_order", "time_constant": 0.05})

        assert tf.settling_time == pytest.approx(-np.log(1e-3) * 0.05)

    def test_apply_should_keep_constant_input_unchanged_with_steady_state_initialization(self):
        tf = TransferFunction.from_config({"type": "first_order", "time_constant": 0.05})

        discrete_time = np.linspace(0, 0.5, num=10001)
        values = np.full(discrete_time.size, 8.0)

        filtered = tf.apply(values, discrete_time, steady_input=8.0)

        assert np.max(np.abs(filtered - 8.0)) < 1e-9

    def test_apply_should_match_analytic_first_order_step_response(self):
        time_constant = 0.05
        tf = TransferFunction.from_config(
            {"type": "first_order", "time_constant": time_constant}
        )

        discrete_time = np.linspace(0, 1.0, num=100001)
        values = np.ones(discrete_time.size)

        filtered = tf.apply(values, discrete_time, steady_input=0.0)

        analytic = 1.0 - np.exp(-discrete_time / time_constant)
        assert np.max(np.abs(filtered - analytic)) < 1e-9

    def test_apply_should_conserve_net_area_for_bump_input(self):
        # With unit DC gain, input and output cover the same area whenever the
        # input returns to its initial value and the output settles back to
        # steady state within the time window (zero net displacement deviation).
        tf = TransferFunction.from_config({"type": "first_order", "time_constant": 0.02})

        discrete_time = np.linspace(0, 2.0, num=200001)
        # Bump: ramp up over [0, 0.25], hold, ramp down over [0.75, 1.0],
        # zero afterwards so the filter fully settles within the window.
        ideal = np.clip(4.0 * discrete_time, 0.0, 1.0) - np.clip(
            4.0 * discrete_time - 3.0, 0.0, 1.0
        )

        filtered = tf.apply(ideal, discrete_time, steady_input=0.0)

        net_deviation = np.trapezoid(filtered - ideal, discrete_time)
        assert abs(net_deviation) < 1e-3

    def test_apply_should_match_continuous_time_lsim_reference(self):
        # The ZOH-discretized lfilter path must agree with the continuous-time
        # simulation (scipy.signal.lsim, which linearly interpolates the input
        # between samples). For ramping inputs the two discretization views
        # differ transiently at the sub-micrometer level in displacement,
        # orders of magnitude below the voxel resolution; the displacement
        # (what the model integrates) must agree tightly.
        tf = TransferFunction.from_config(
            {"type": "second_order", "natural_frequency": 50.0, "damping": 0.7}
        )

        discrete_time = np.linspace(0, 2.0, num=100001)
        # trapezoid-like speed trace around a jerk-speed baseline:
        # ramp up over [0, 0.5], cruise at 28, ramp down over [1.0, 1.5], settle at 8
        values = 8.0 + np.clip(discrete_time * 40.0, 0.0, 20.0) - np.clip(
            discrete_time * 40.0 - 40.0, 0.0, 20.0
        )

        filtered = tf.apply(values, discrete_time, steady_input=8.0)

        system = signal.lti(tf.num, tf.den).to_ss()
        x0 = -np.linalg.solve(
            np.asarray(system.A), np.asarray(system.B).ravel() * 8.0
        )
        _, reference, _ = signal.lsim(system, U=values, T=discrete_time, X0=x0)

        # constant sections (cruise and final settle) agree tightly
        constant_section = (discrete_time > 0.7) & (discrete_time < 0.9) | (
            discrete_time > 1.7
        )
        assert (
            np.max(np.abs(filtered[constant_section] - reference[constant_section]))
            < 1e-6
        )

        # displacement agrees to sub-micrometer level over the whole move
        displacements_filtered = np.concatenate(
            ([0.0], np.cumsum((filtered[1:] + filtered[:-1]) * np.diff(discrete_time) * 0.5))
        )
        displacements_reference = np.concatenate(
            ([0.0], np.cumsum((reference[1:] + reference[:-1]) * np.diff(discrete_time) * 0.5))
        )
        assert np.max(np.abs(displacements_filtered - displacements_reference)) < 1e-3

    def test_unknown_type_should_raise(self):
        with pytest.raises(ValueError, match="Unknown transfer function type"):
            TransferFunction.from_config({"type": "bogus"})

    def test_non_unit_dc_gain_should_raise(self):
        with pytest.raises(ValueError, match="unit DC gain"):
            TransferFunction.from_config(
                {"type": "num_den", "num": [2.0], "den": [1.0, 1.0]}
            )

    def test_unstable_pole_should_raise(self):
        with pytest.raises(ValueError, match="stable"):
            TransferFunction.from_config(
                {"type": "num_den", "num": [1.0], "den": [1.0, -1.0]}
            )

    def test_integrator_should_raise(self):
        with pytest.raises(ValueError, match="integrator"):
            TransferFunction.from_config(
                {"type": "num_den", "num": [1.0], "den": [1.0, 0.0]}
            )

    def test_improper_transfer_function_should_raise(self):
        with pytest.raises(ValueError, match="proper"):
            TransferFunction.from_config(
                {"type": "num_den", "num": [1.0, 1.0], "den": [1.0]}
            )

    def test_non_positive_time_constant_should_raise(self):
        with pytest.raises(ValueError, match="time_constant"):
            TransferFunction.from_config({"type": "first_order", "time_constant": -0.1})
