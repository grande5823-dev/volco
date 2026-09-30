import logging

logger = logging.getLogger(__name__)


def default_increase_solver_tolerance(point_a, point_b):
    return False


class BisectionMethod:
    """
    Accelerated Root Solver using Brent's Method.
    
    Combines superlinear convergence (Inverse Quadratic Interpolation & Secant)
    with guaranteed linear convergence (Bisection fallback) to handle discrete
    voxel steps safely while maintaining full compatibility with sphere.py.
    """

    def execute(
        self,
        fun,
        initial_point,
        tolerance,
        increment,
        args,
        fun_increase_tolerance=default_increase_solver_tolerance,
        max_iterations=None,
    ):
        if initial_point == 0.0:
            point_b = increment
        else:
            point_b = initial_point

        fb, out_b, point_a, point_b = self._loop_fb(
            fun, point_b, increment, tolerance, args, max_iterations
        )

        # If bracketing scan already lands within tolerance, return immediately
        if abs(fb) <= tolerance:
            return fb, out_b

        return self._loop_brent(
            fun, point_a, point_b, fb, out_b, tolerance, fun_increase_tolerance, args, max_iterations
        )

    def _loop_fb(self, fun, point_b, inc, tolerance, args, max_iterations):
        fb = -1.0
        point_a = 0.0
        iterations = 0
        out = None

        while fb < 0.0 and abs(fb) > tolerance:
            fb, out = fun(point_b, *args)

            if fb < 0:
                point_a = point_b
                point_b += inc

            iterations += 1
            self._check_iteration_limit(iterations, max_iterations)

        return fb, out, point_a, point_b

    def _loop_brent(
        self, fun, a, b, fb, out_b, tolerance, fun_increase_tolerance, args, max_iterations
    ):
        # Evaluate f(a) to establish bracket bounds
        fa, out_a = fun(a, *args)

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            out_a, out_b = out_b, out_a

        c, fc, out_c = a, fa, out_a
        d = e = b - a
        m_flag = True

        iterations = 0

        while abs(fb) > tolerance:
            if fun_increase_tolerance(a, b):
                logger.debug(
                    "[BisectionMethod]: increasing tolerance because point_b and point_a are too close"
                )
                tolerance *= 10.0

            # Step 1: Interpolation Attempt with ZeroDivision Safeguard
            try:
                if fa != fc and fb != fc and fa != fb:
                    # Inverse Quadratic Interpolation
                    s = (
                        (a * fb * fc) / ((fa - fb) * (fa - fc))
                        + (b * fa * fc) / ((fb - fa) * (fb - fc))
                        + (c * fa * fb) / ((fc - fa) * (fc - fb))
                    )
                elif fa != fb:
                    # Secant Method
                    s = b - fb * (b - a) / (fb - fa)
                else:
                    # Fallback to Bisection when evaluations match
                    s = (a + b) * 0.5
                    m_flag = True
            except ZeroDivisionError:
                s = (a + b) * 0.5
                m_flag = True

            # Step 2: Validate interpolation step; revert to Bisection if unsafe
            cond1 = not (((3 * a + b) / 4 <= s <= b) or (b <= s <= (3 * a + b) / 4))
            cond2 = m_flag and (abs(s - b) >= abs(b - c) / 2)
            cond3 = (not m_flag) and (abs(s - b) >= abs(c - d) / 2)
            cond4 = m_flag and (abs(b - c) < tolerance)
            cond5 = (not m_flag) and (abs(c - d) < tolerance)

            if cond1 or cond2 or cond3 or cond4 or cond5:
                s = (a + b) * 0.5
                m_flag = True
            else:
                m_flag = False

            # Step 3: Evaluate target function at candidate point s
            fs, out_s = fun(s, *args)

            d = e
            e = b - a
            c, fc, out_c = b, fb, out_b

            if fa * fs < 0:
                b, fb, out_b = s, fs, out_s
            else:
                a, fa, out_a = s, fs, out_s

            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                out_a, out_b = out_b, out_a

            iterations += 1
            self._check_iteration_limit(iterations, max_iterations)

        return fb, out_b

    def _check_iteration_limit(self, iterations, max_iterations):
        if max_iterations is not None and iterations > max_iterations:
            raise RuntimeError(
                f"BisectionMethod did not converge after {max_iterations} "
                "iterations; check that the target volume is reachable "
                "(e.g. non-finite volumes or unfillable voxel space)"
            )