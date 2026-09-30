import math
import numpy as np

from app.geometry.geometry_math import GeometryMath
from app.solvers.bisection_method import BisectionMethod


class Sphere:
    MAX_SOLVER_ITERATIONS = 500

    def __init__(self, centre_coordinates, voxel_size):
        self.centre_coordinates = centre_coordinates
        self.voxel_size = voxel_size
        self.filled_voxel_count = None

    def deposit_sphere(
        self,
        voxel_space,
        nozzle_height,
        sphere_volume,
        voxel_space_target_volume,
        solver_tolerance,
        radius_increment,
        baseline_filled_voxels=None,
    ):
        """
        Deposit a sphere into the voxel space, choosing its radius by bisection
        so that the total filled volume matches voxel_space_target_volume.

        Evaluation is done in place on a working copy of the bounding-box
        region only (the region is restored after each trial radius), and the
        filled volume is tracked incrementally from baseline_filled_voxels, so
        neither the whole voxel space copy nor the whole-space volume count
        scales with the size of the print. If the space had to be grown for a
        trial radius, the growth is kept; it only adds empty boundary slabs.

        baseline_filled_voxels: number of voxels already filled in voxel_space
        before this sphere (computed from the array if not provided).

        After the call, self.filled_voxel_count holds the new total number of
        filled voxels in the returned voxel space.
        """
        if baseline_filled_voxels is None:
            baseline_filled_voxels = int(np.count_nonzero(voxel_space))

        self._working_space = voxel_space
        self._baseline_filled_voxels = baseline_filled_voxels
        self._last_radius = None

        initial_radius = self.estimate_initial_radius(sphere_volume)

        BisectionMethod().execute(
            self._deposit_sphere,
            initial_point=initial_radius,
            tolerance=solver_tolerance,
            increment=radius_increment,
            fun_increase_tolerance=self._increase_solver_tolerance,
            args=(nozzle_height, voxel_space_target_volume),
            max_iterations=self.MAX_SOLVER_ITERATIONS,
        )

        # Re-apply the accepted radius for good: the bisection evaluations
        # above revert their changes after measuring.
        working_space = self.deform_voxel_space_for_big_spheres(
            self._working_space, self._last_radius
        )
        lower_indexes, upper_indexes = self.find_sphere_limits(
            self._last_radius, nozzle_height
        )
        region = self._region_of(working_space, lower_indexes, upper_indexes)
        newly_filled = self._fill_region(region, self._last_radius, lower_indexes)

        self._working_space = working_space
        self.filled_voxel_count = self._baseline_filled_voxels + newly_filled

        return self._working_space

    def fill_voxels(self, voxel_space, radius, lower_indexes, upper_indexes):
        region = self._region_of(voxel_space, lower_indexes, upper_indexes)
        self._fill_region(region, radius, lower_indexes)

        return voxel_space

    def estimate_initial_radius(self, volume):
        return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    def find_sphere_limits(self, radius, nozzle_height):
        # subtract 1 because the voxel space starts at position [0,0]
        min_indexes = [
            self._find_index(centre_coordinate - radius)
            for centre_coordinate in self.centre_coordinates
        ]

        # Clamp to the low boundary on all axes: a negative index would make
        # numpy slicing wrap around to the far end of the array, filling
        # voxels at the wrong location (and, via the bisection loop, growing
        # the voxel space without bound).
        min_indexes = [max(min_index, 0) for min_index in min_indexes]

        max_indexes = [
            self._find_index(centre_coordinate + radius)
            for centre_coordinate in self.centre_coordinates
        ]

        _, _, z0 = self.centre_coordinates
        if z0 + radius > nozzle_height:
            max_indexes[2] = self._find_index(nozzle_height)
        else:
            max_indexes[2] = self._find_index(z0 + radius)

        return min_indexes, max_indexes

    """
    Checks if the sphere violates the boundaries of the voxel space. If it does,
    the voxel space is expanded to accommodate the sphere.
    """

    def deform_voxel_space_for_big_spheres(self, voxel_space, radius):
        max_indexes = [
            self._find_index(coord + radius) for coord in self.centre_coordinates
        ]

        for axis_number in range(0, 3):
            voxel_space = self._maybe_expand_voxel_space(
                voxel_space, max_indexes[axis_number], axis_number
            )

        return voxel_space

    def _maybe_expand_voxel_space(self, voxel_space, max_index, axis_number):
        size = voxel_space.shape

        index_size = size[axis_number]

        if max_index < index_size:
            return voxel_space

        number_to_be_added = max_index - index_size + 1

        new_size = list(size)
        new_size[axis_number] = number_to_be_added

        mat_add = np.zeros(new_size, dtype=np.int8)

        voxel_space = np.concatenate((voxel_space, mat_add), axis=axis_number)

        return voxel_space

    def _deposit_sphere(self, radius, nozzle_height, target_volume):
        """
        Trial evaluation for the bisection method: fills the sphere's
        bounding-box region in place, measures the resulting total volume from
        the incremental count, then restores the region.
        """
        working_space = self.deform_voxel_space_for_big_spheres(
            self._working_space, radius
        )
        self._working_space = working_space

        lower_indexes, upper_indexes = self.find_sphere_limits(
            radius, nozzle_height
        )

        region = self._region_of(working_space, lower_indexes, upper_indexes)
        region_backup = region.copy()

        newly_filled = self._fill_region(region, radius, lower_indexes)

        current_volume = (
            self._baseline_filled_voxels + newly_filled
        ) * self.voxel_size**3

        volume_overshoot = current_volume / target_volume - 1.0

        region[...] = region_backup

        self._last_radius = radius

        return volume_overshoot, None

    def _region_of(self, voxel_space, lower_indexes, upper_indexes):
        lower_i, lower_j, lower_k = lower_indexes
        upper_i, upper_j, upper_k = upper_indexes

        return voxel_space[
            lower_i : upper_i + 1, lower_j : upper_j + 1, lower_k : upper_k + 1
        ]

    def _fill_region(self, region, radius, lower_indexes):
        """
        Set to 1 the empty voxels of `region` whose centre is within `radius`
        of the sphere centre. Returns the number of newly filled voxels.
        Vectorized equivalent of the previous per-voxel loop; coordinate and
        distance formulas per voxel are unchanged.
        """
        if not (region == 0).any():
            return 0

        shape_i, shape_j, shape_k = region.shape

        indexes_i = np.arange(lower_indexes[0], lower_indexes[0] + shape_i)
        indexes_j = np.arange(lower_indexes[1], lower_indexes[1] + shape_j)
        indexes_k = np.arange(lower_indexes[2], lower_indexes[2] + shape_k)

        coordinates_i = self.voxel_size * (2 * (indexes_i + 1) - 1) * 0.5
        coordinates_j = self.voxel_size * (2 * (indexes_j + 1) - 1) * 0.5
        coordinates_k = self.voxel_size * (2 * (indexes_k + 1) - 1) * 0.5

        centre_i, centre_j, centre_k = self.centre_coordinates

        delta_i = coordinates_i[:, None, None] - centre_i
        delta_j = coordinates_j[None, :, None] - centre_j
        delta_k = coordinates_k[None, None, :] - centre_k

        distance = np.sqrt(delta_i**2 + delta_j**2 + delta_k**2)

        within_radius = distance <= radius + self.voxel_size * 1e-8

        mask = within_radius & (region == 0)

        newly_filled = int(mask.sum())
        region[mask] = 1

        return newly_filled

    def _increase_solver_tolerance(self, radius_a, radius_b):
        return radius_b - radius_a < self.voxel_size * 0.5

    def _find_index(self, coordinate):
        return GeometryMath.find_index(coordinate, self.voxel_size)
