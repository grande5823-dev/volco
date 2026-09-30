import logging
import numpy as np
import math

from app.configs.simulation import Simulation
from app.configs.printer import Printer
from app.geometry.sphere import Sphere
from app.instructions.instruction import Instruction
from app.geometry.geometry_math import GeometryMath
from app.physics.volume import Volume


logger = logging.getLogger(__name__)


class VoxelSpace:
    def __init__(
        self, instruction: Instruction, simulation_config: Simulation, printer: Printer):
        coord_limits = instruction.coordinate_limits

        [x_min, x_max] = coord_limits["x"]
        [y_min, y_max] = coord_limits["y"]
        [z_min, z_max] = coord_limits["z"]

        x_offset = simulation_config.x_offset
        y_offset = simulation_config.y_offset
        z_offset = simulation_config.z_offset

        dim = [
            x_max - x_min + 2.0 * x_offset,
            y_max - y_min + 2.0 * y_offset,
            z_max - z_min + z_offset,
        ]

        dimensions = [int(dim_i / simulation_config.voxel_size) for dim_i in dim]
        self.dimensions = {
            "x": dimensions[0],
            "y": dimensions[1],
            "z": dimensions[2],
        }

        self.filament_translations = {"x": x_offset - x_min, "y": y_offset - y_min}

        self._instruction = instruction
        self._simulation = simulation_config
        self._printer = printer
        self._consider_acceleration = self._simulation.consider_acceleration

    def initialize_space(self):
        self.space = np.zeros(
            (self.dimensions["x"], self.dimensions["y"], self.dimensions["z"]),
            dtype=np.int8,
        )

    def print(self):
        number_printed_filaments = 0
        number_printed_layers = 0
        skipped_zero_length_moves = 0
        skipped_zero_length_volume = 0.0
        skipped_below_bed_moves = 0
        skipped_below_bed_volume = 0.0

        initial_z_coordinate = 0.0

        # Filled volume is tracked incrementally across spheres and filaments
        # instead of recounting the whole voxel space for every sphere.
        running_filled_voxels = int(np.count_nonzero(self.space))

        for filament_coordinates in self._instruction.filaments_coordinates:
            (
                initial_coordinate,
                final_coordinate,
            ) = self._find_initial_and_final_filament_coordinates(filament_coordinates)

            volume = filament_coordinates[2]  # Now this is volume, not E

            filament_length = GeometryMath.distance(
                initial_coordinate, final_coordinate
            )

            if filament_length == 0.0:
                # Stationary extrusion move (prime/unretract): there is no
                # direction along which material could be deposited.
                skipped_zero_length_moves += 1
                skipped_zero_length_volume += volume
                continue

            if min(initial_coordinate[2], final_coordinate[2]) <= 0.0:
                # Filament at or below the bed plane: the simulated bead is
                # placed below the nozzle tip, and nothing can be deposited
                # when the nozzle touches the bed.
                skipped_below_bed_moves += 1
                skipped_below_bed_volume += volume
                continue

            direction_vector = GeometryMath.direction_vector(
                initial_coordinate, final_coordinate
            )

            printing_speed = filament_coordinates[1][4]

            number_printed_filaments += 1

            if final_coordinate[2] > initial_z_coordinate:
                number_printed_layers += 1
                initial_z_coordinate = final_coordinate[2]

            number_simulation_steps, step_size = self._find_simulation_step_info(
                filament_length
            )

            volumes = Volume.get_volumes_for_filament(
                number_simulation_steps=number_simulation_steps,
                total_volume=volume,
                consider_acceleration=self._consider_acceleration,
                filament_length=filament_length,
                printing_speed=printing_speed,
                printer=self._printer,
            )

            running_filled_voxels = self._deposit_filament(
                number_simulation_steps=number_simulation_steps,
                step_size=step_size,
                direction_vector=direction_vector,
                filament_initial_coordinates=initial_coordinate,
                volumes=volumes,
                baseline_filled_voxels=running_filled_voxels,
            )

        if skipped_zero_length_moves > 0:
            logger.warning(
                "Skipped %d zero-length extrusion move(s) (stationary "
                "prime/unretract); %.6f mm^3 of material were not deposited",
                skipped_zero_length_moves,
                skipped_zero_length_volume,
            )

        if skipped_below_bed_moves > 0:
            logger.warning(
                "Skipped %d extrusion move(s) at or below the bed plane "
                "(z <= 0); %.6f mm^3 of material were not deposited",
                skipped_below_bed_moves,
                skipped_below_bed_volume,
            )

    def _find_initial_and_final_filament_coordinates(self, filament_coordinates):
        initial = [
            filament_coordinates[0][0] + self.filament_translations["x"],
            filament_coordinates[0][1] + self.filament_translations["y"],
            filament_coordinates[0][2],
        ]
        final = (
            filament_coordinates[1][0] + self.filament_translations["x"],
            filament_coordinates[1][1] + self.filament_translations["y"],
            filament_coordinates[1][2],
        )
        return initial, final

    def _find_simulation_step_info(self, filament_length):
        number_simulation_steps = int(
            round(filament_length / self._simulation.step_size)
        )

        # Segments shorter than half the step size are simulated as one step
        number_simulation_steps = max(1, number_simulation_steps)

        step_size = filament_length / number_simulation_steps

        return number_simulation_steps, step_size

    def _deposit_filament(
        self,
        number_simulation_steps,
        step_size,
        direction_vector,
        filament_initial_coordinates,
        volumes,
        baseline_filled_voxels,
    ):
        total_deposited_volume = baseline_filled_voxels * self._simulation.voxel_size**3

        running_filled_voxels = baseline_filled_voxels

        for step_n in range(0, number_simulation_steps):

            # o --------------------- X --------------------- o --------------------- X --------------------- o
            # starting_point      centre_coordinates
            #
            # |<--------------------------------------------->|
            #              step_size
            #
            # |<--------------------------------------------------------------------------------------------->|
            #                                           filament length

            displacement = step_n * step_size
            starting_point = [
                pi_i + dir_vec_i * displacement
                for (pi_i, dir_vec_i) in zip(
                    filament_initial_coordinates, direction_vector
                )
            ]

            displacement_to_centre = step_size * 0.5
            centre_coordinates = [
                pstart_i + dir_vec_i * displacement_to_centre
                for (pstart_i, dir_vec_i) in zip(starting_point, direction_vector)
            ]

            nozzle_height = centre_coordinates[2]

            # Correcting the centre of the sphere based on the configured z-offset
            centre_coordinates[2] = (
                centre_coordinates[2] - self._simulation.sphere_z_offset
            )

            sphere_volume = volumes[step_n]
            volume_target = total_deposited_volume + sphere_volume

            sphere = Sphere(
                centre_coordinates=centre_coordinates,
                voxel_size=self._simulation.voxel_size,
            )

            logger.info(
                f"Depositing filament: step = {step_n + 1}/{number_simulation_steps}"
            )

            self.space = sphere.deposit_sphere(
                voxel_space=self.space,
                nozzle_height=nozzle_height,
                sphere_volume=sphere_volume,
                voxel_space_target_volume=volume_target,
                solver_tolerance=self._simulation.solver_tolerance,
                radius_increment=self._simulation.radius_increment,
                baseline_filled_voxels=running_filled_voxels,
            )

            running_filled_voxels = sphere.filled_voxel_count

        return running_filled_voxels
