import logging


logger = logging.getLogger(__name__)


def default_increase_solver_tolerance(point_a, point_b):
    return False


class BisectionMethod:
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

        _, _, point_a, point_b = self._loop_fb(
            fun, point_b, increment, tolerance, args, max_iterations
        )

        return self._loop_fc(
            fun, point_a, point_b, tolerance, fun_increase_tolerance, args, max_iterations
        )

    def _loop_fb(self, fun, point_b, inc, tolerance, args, max_iterations):
        fb = -1.0

        point_a = 0.0

        iterations = 0

        # The scan stops when fb >= 0 OR when the overshoot is already within
        # the solver tolerance: in saturated (heavily overlapping) regions the
        # fill can only just undershoot the target, and scanning onward would
        # grow the radius far beyond the physical scale of the bead.
        while fb < 0.0 and abs(fb) > tolerance:
            fb, out = fun(point_b, *args)

            if fb < 0:
                point_a = point_b
                point_b += inc

            iterations += 1
            self._check_iteration_limit(iterations, max_iterations)

        return fb, out, point_a, point_b

    def _loop_fc(self, fun, point_a, point_b, tolerance, fun_increase_tolerance, args, max_iterations):
        fc = 2.0 * tolerance

        iterations = 0

        while abs(fc) > tolerance:
            point_c = (point_a + point_b) * 0.5

            fc, out = fun(point_c, *args)

            if fun_increase_tolerance(point_a, point_b):
                logger.debug(
                    "[BisectionMethod]: increasing tolerance because point_b and point_a are too close"
                )
                tolerance = tolerance * 10

            if fc < 0.0:
                point_a = point_c
            else:
                point_b = point_c

            iterations += 1
            self._check_iteration_limit(iterations, max_iterations)

        return fc, out

    def _check_iteration_limit(self, iterations, max_iterations):
        if max_iterations is not None and iterations > max_iterations:
            raise RuntimeError(
                f"BisectionMethod did not converge after {max_iterations} "
                "iterations; check that the target volume is reachable "
                "(e.g. non-finite volumes or unfillable voxel space)"
            )
