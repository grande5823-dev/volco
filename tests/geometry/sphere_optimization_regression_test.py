import math

import numpy as np
import pytest

from app.geometry.geometry_math import GeometryMath
from app.geometry.sphere import Sphere
from app.solvers.bisection_method import BisectionMethod


class LegacySphere:
    """
    Byte-for-byte re-implementation of the pre-optimization deposition
    algorithm: full voxel-space copy per bisection evaluation, per-voxel
    python fill loop, whole-space volume count. Used as the golden reference
    for the optimized in-place implementation.
    """

    def __init__(self, centre_coordinates, voxel_size):
        self.centre_coordinates = centre_coordinates
        self.voxel_size = voxel_size

    def deposit_sphere(
        self,
        voxel_space,
        nozzle_height,
        sphere_volume,
        voxel_space_target_volume,
        solver_tolerance,
        radius_increment,
    ):
        initial_radius = self.estimate_initial_radius(sphere_volume)

        _, voxel_space = BisectionMethod().execute(
            self._deposit_sphere,
            initial_point=initial_radius,
            tolerance=solver_tolerance,
            increment=radius_increment,
            fun_increase_tolerance=self._increase_solver_tolerance,
            args=(voxel_space, nozzle_height, voxel_space_target_volume),
        )

        return voxel_space

    def estimate_initial_radius(self, volume):
        return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    def find_sphere_limits(self, radius, nozzle_height):
        min_indexes = [
            self._find_index(centre_coordinate - radius)
            for centre_coordinate in self.centre_coordinates
        ]
        if min_indexes[2] < 0:
            min_indexes[2] = 0
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

    def deform_voxel_space_for_big_spheres(self, voxel_space, radius):
        max_indexes = [
            self._find_index(coord + radius) for coord in self.centre_coordinates
        ]
        for axis_number in range(0, 3):
            size = voxel_space.shape
            index_size = size[axis_number]
            if max_indexes[axis_number] >= index_size:
                number_to_be_added = max_indexes[axis_number] - index_size + 1
                new_size = list(size)
                new_size[axis_number] = number_to_be_added
                voxel_space = np.concatenate(
                    (voxel_space, np.zeros(new_size, dtype=np.int8)),
                    axis=axis_number,
                )

        return voxel_space

    def _deposit_sphere(self, radius, voxel_space, nozzle_height, target_volume):
        copy_voxel_space = voxel_space.copy()

        copy_voxel_space = self.deform_voxel_space_for_big_spheres(
            copy_voxel_space, radius
        )

        lower_indexes, upper_indexes = self.find_sphere_limits(radius, nozzle_height)

        copy_voxel_space = self._fill_voxels(
            copy_voxel_space, radius, lower_indexes, upper_indexes
        )

        current_volume = np.count_nonzero(copy_voxel_space) * self.voxel_size**3

        volume_overshoot = current_volume / target_volume - 1.0

        return volume_overshoot, copy_voxel_space

    def _fill_voxels(self, voxel_space, radius, lower_indexes, upper_indexes):
        lower_i, lower_j, lower_k = lower_indexes
        upper_i, upper_j, upper_k = upper_indexes

        mat_aux = voxel_space[
            lower_i : upper_i + 1, lower_j : upper_j + 1, lower_k : upper_k + 1
        ]

        empty_voxels = np.transpose(np.where(mat_aux == 0))
        empty_voxels = [
            [elem[0] + lower_i, elem[1] + lower_j, elem[2] + lower_k]
            for elem in empty_voxels
        ]

        for voxel in empty_voxels:
            voxel_coordinate = [
                self.voxel_size * (2 * (index + 1) - 1) * 0.5 for index in voxel
            ]
            d = 0.0
            for (p1_i, p2_i) in zip(voxel_coordinate, self.centre_coordinates):
                d += (p1_i - p2_i) ** 2
            distance = math.sqrt(d)

            if distance <= radius + self.voxel_size * 1e-8:
                voxel_space[tuple(voxel)] = 1

        return voxel_space

    def _increase_solver_tolerance(self, radius_a, radius_b):
        return radius_b - radius_a < self.voxel_size * 0.5

    def _find_index(self, coordinate):
        return GeometryMath.find_index(coordinate, self.voxel_size)


SCENARIOS = [
    # (centre, voxel_size, radius scale via target volume, prefill, description)
    ([5.0, 5.0, 2.0], 0.1, False, "empty space"),
    ([5.0, 5.0, 2.0], 0.1, True, "overlapping previously deposited material"),
    ([9.6, 5.0, 2.0], 0.1, True, "near edge requiring voxel-space growth"),
    ([5.0, 5.0, 1.95], 0.05, True, "finer voxels, z clipped by nozzle height"),
]


class TestOptimizedSphereMatchesLegacy:
    @pytest.mark.parametrize(
        "centre, voxel_size, prefill, description", SCENARIOS
    )
    def test_final_voxel_space_matches_legacy_bitwise(
        self, centre, voxel_size, prefill, description
    ):
        solver_tolerance = 0.0001
        radius_increment = 0.1

        base_space = np.zeros((100, 100, 40), dtype=np.int8)
        if prefill:
            # first deposit a sphere with both implementations to build a
            # non-trivial starting arrangement
            first_of_legacy = LegacySphere([4.5, 5.2, 2.05], voxel_size)
            base_space = first_of_legacy.deposit_sphere(
                base_space,
                nozzle_height=centre[2],
                sphere_volume=0.35,
                voxel_space_target_volume=0.35,
                solver_tolerance=solver_tolerance,
                radius_increment=radius_increment,
            )

        target_volume = np.count_nonzero(base_space) * voxel_size**3 + 0.42

        legacy = LegacySphere(centre, voxel_size)
        legacy_space = legacy.deposit_sphere(
            base_space.copy(),
            nozzle_height=centre[2],
            sphere_volume=0.42,
            voxel_space_target_volume=target_volume,
            solver_tolerance=solver_tolerance,
            radius_increment=radius_increment,
        )

        optimized = Sphere(centre, voxel_size)
        optimized_space = optimized.deposit_sphere(
            base_space.copy(),
            nozzle_height=centre[2],
            sphere_volume=0.42,
            voxel_space_target_volume=target_volume,
            solver_tolerance=solver_tolerance,
            radius_increment=radius_increment,
        )

        # The optimized implementation may keep extra empty slabs grown while
        # probing larger trial radii; the shared region must match exactly and
        # the extra region must be empty.
        for axis in range(3):
            assert optimized_space.shape[axis] >= legacy_space.shape[axis]

        slices = tuple(slice(0, n) for n in legacy_space.shape)
        assert np.array_equal(optimized_space[slices], legacy_space), description

        # Equal shared region plus equal total fill implies empty extra slabs
        assert np.count_nonzero(optimized_space) == np.count_nonzero(legacy_space)
        assert optimized.filled_voxel_count == int(np.count_nonzero(legacy_space))

    def test_incremental_counting_across_consecutive_spheres_matches_legacy(self):
        voxel_size = 0.1
        space_legacy = np.zeros((100, 100, 40), dtype=np.int8)
        space_optimized = space_legacy.copy()

        running = int(np.count_nonzero(space_optimized))
        deposited_volume = 0.0

        for n in range(5):
            centre = [3.0 + n * 0.8, 5.0, 2.0]
            deposited_volume += 0.42
            target_volume = deposited_volume

            legacy = LegacySphere(centre, voxel_size)
            space_legacy = legacy.deposit_sphere(
                space_legacy, 2.0, 0.42, target_volume, 0.0001, 0.1
            )

            optimized = Sphere(centre, voxel_size)
            space_optimized = optimized.deposit_sphere(
                space_optimized,
                2.0,
                0.42,
                target_volume,
                0.0001,
                0.1,
                baseline_filled_voxels=running,
            )
            running = optimized.filled_voxel_count

        slices = tuple(slice(0, n) for n in space_legacy.shape)
        assert np.array_equal(space_optimized[slices], space_legacy)
        assert running == int(np.count_nonzero(space_legacy))
