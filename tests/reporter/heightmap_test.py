import importlib.util

import numpy as np
import pytest

from app.reporter.heightmap import compute_heightmap

needs_trimesh = pytest.mark.skipif(
    importlib.util.find_spec("trimesh") is None,
    reason="app.reporter.report requires trimesh",
)


class TestComputeHeightmap:
    def test_highest_voxel_above_plane_is_reported(self):
        space = np.zeros((2, 1, 10), dtype=np.int8)
        space[0, 0, 2] = 1  # spans z (2, 3] at voxel_size=1
        space[0, 0, 7] = 1  # spans z (7, 8]
        space[1, 0, 4] = 1  # spans z (4, 5]

        heights = compute_heightmap(space, voxel_size=1.0, z_min=0.0)
        assert heights[0, 0] == pytest.approx(8.0)
        assert heights[1, 0] == pytest.approx(5.0)

    def test_voxels_below_the_candidate_plane_are_ignored(self):
        space = np.zeros((1, 1, 10), dtype=np.int8)
        space[0, 0, 3] = 1  # only material, below the plane z >= 5

        heights = compute_heightmap(space, voxel_size=1.0, z_min=5.0)
        assert np.isnan(heights[0, 0])

    def test_plane_boundary_is_inclusive(self):
        space = np.zeros((1, 1, 10), dtype=np.int8)
        space[0, 0, 5] = 1  # lower face exactly at z = 5

        heights = compute_heightmap(space, voxel_size=1.0, z_min=5.0)
        assert heights[0, 0] == pytest.approx(6.0)

    def test_plane_above_all_material_returns_all_missing(self):
        space = np.zeros((2, 3, 4), dtype=np.int8)
        space[0, 1, 3] = 1

        heights = compute_heightmap(space, voxel_size=1.0, z_min=100.0)
        assert np.isnan(heights).all()
        assert heights.shape == (2, 3)

    def test_custom_missing_value(self):
        space = np.zeros((2, 1, 4), dtype=np.int8)
        space[0, 0, 1] = 1

        heights = compute_heightmap(space, 1.0, 0.0, missing_value=-9999.0)
        assert heights[0, 0] > 0.0
        assert heights[1, 0] == -9999.0

    def test_z_crop_offset(self):
        # Full space would have 10 planes; this view starts at k_first = 4.
        view = np.zeros((1, 1, 3), dtype=np.int8)
        view[0, 0, 2] = 1  # global k = 6 -> surface at z = 7

        heights = compute_heightmap(view, voxel_size=1.0, z_min=0.0, k_first=4)
        assert heights[0, 0] == pytest.approx(7.0)

    def test_noninteger_plane_and_voxel_size(self):
        space = np.zeros((1, 1, 100), dtype=np.int8)
        # voxel 15 spans (0.75, 0.8]; plane z >= 0.76 must exclude it,
        # plane z >= 0.75 must include it.
        space[0, 0, 15] = 1

        assert np.isnan(compute_heightmap(space, 0.05, 0.76)[0, 0])
        assert compute_heightmap(space, 0.05, 0.75)[0, 0] == pytest.approx(0.8)


class FakeVoxelSpace:
    def __init__(self, space, tx=0.0, ty=0.0):
        self.space = space
        self.filament_translations = {"x": tx, "y": ty}


class FakeSimulation:
    def __init__(self, voxel_size=1.0, results_folder="", name="test"):
        self.voxel_size = voxel_size
        self.x_crop = ["all", "all"]
        self.y_crop = ["all", "all"]
        self.z_crop = ["all", "all"]
        self.results_folder = results_folder
        self.simulation_name = name
        self.stl_ascii = False


@needs_trimesh
class TestSimulationOutputHeightmap:
    def _make_output(self, space, tx=0.0, ty=0.0, voxel_size=1.0, **kwargs):
        from app.reporter.report import SimulationOutput

        return SimulationOutput(
            voxel_space=FakeVoxelSpace(space, tx, ty),
            simulation=FakeSimulation(voxel_size=voxel_size, **kwargs),
        )

    def test_extract_heightmap_heights(self):
        space = np.zeros((3, 2, 10), dtype=np.int8)
        space[1, 0, 2:6] = 1  # column top voxel k=5 -> height 6
        space[2, 1, 9] = 1

        output = self._make_output(space)
        result = output.extract_heightmap(z_min=0.0)

        assert result["heights"][1, 0] == pytest.approx(6.0)
        assert result["heights"][2, 1] == pytest.approx(10.0)
        assert np.isnan(result["heights"][0, 0])
        # output is cached for later access
        assert output.heightmap is result

    def test_axes_are_in_gcode_coordinates(self):
        # voxel_size = 1, 4 columns in x, 3 in y, translation 2 in x
        space = np.zeros((4, 3, 2), dtype=np.int8)
        output = self._make_output(space, tx=2.0, ty=-1.0)
        result = output.extract_heightmap(z_min=0.0)

        # column index i covers (i, i+1] voxel mm -> center i + 0.5,
        # gcode coordinate subtracts the filament translation
        assert result["x"].tolist() == [-1.5, -0.5, 0.5, 1.5]
        assert result["y"].tolist() == [1.5, 2.5, 3.5]

    def test_axes_respect_xy_crop(self):
        space = np.zeros((10, 10, 4), dtype=np.int8)
        output = self._make_output(space, voxel_size=1.0)
        output._simulation.x_crop = [2.0, 5.0]
        output._simulation.y_crop = [1.0, 3.0]

        result = output.extract_heightmap(z_min=0.0)
        # repo crop convention: coordinate c -> index ceil(c/vs)-1, so
        # [2.0, 5.0] keeps indexes 1..4 with centers at (i + 0.5)*vs
        assert result["x"].tolist() == [1.5, 2.5, 3.5, 4.5]
        assert result["y"].tolist() == [0.5, 1.5, 2.5]

    def test_export_heightmap_npz(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        space = np.zeros((2, 2, 5), dtype=np.int8)
        space[0, 0, 4] = 1

        output = self._make_output(space, results_folder="", name="sim")
        path = output.export_heightmap(z_min=2.0, missing_value=-1.0)

        data = np.load(path)
        assert path.endswith("sim_heightmap.npz")
        assert data["heights"][0, 0] == pytest.approx(5.0)
        assert data["heights"][1, 1] == -1.0
        assert data["z_min"] == 2.0
        assert set(data.files) >= {"heights", "x", "y", "z_min", "voxel_size",
                                   "missing_value"}


@needs_trimesh
class TestRunSimulationHeightmap:
    GCODE = (
        "M83\n"
        "G90\n"
        "G1 Z0.2 F3000\n"
        "G1 X0 Y0\n"
        "G1 X3 E0.3\n"
        "G1 Y3 E0.3\n"
        "G1 Z0.4\n"
        "G1 X0 E0.3\n"
        "G1 Z0.6\n"
        "G1 Y0 E0.3\n"  # top layer extrusion at z = 0.6
    )

    PRINTER = {
        "nozzle_diameter": 0.4,
        "feedstock_filament_diameter": 1.75,
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
    }

    def _sim_config(self, tmp_path):
        return {
            "voxel_size": 0.1,
            "step_size": 0.2,
            "x_offset": 1.0,
            "y_offset": 1.0,
            "z_offset": 0,
            "sphere_z_offset": 0.2,
            "simulation_name": "heightmap_e2e",
            "results_folder": str(tmp_path),
            "radius_increment": 0.1,
            "solver_tolerance": 0.0001,
        }

    def test_run_simulation_exports_heightmap_only(self, tmp_path):
        from volco import run_simulation

        output = run_simulation(
            gcode=self.GCODE,
            printer_config=self.PRINTER,
            sim_config=self._sim_config(tmp_path),
            heightmap_z_min=0.45,
        )

        # Mesh generation is skipped; only the heightmap is produced.
        assert output.mesh is None
        heightmap = output.heightmap
        assert heightmap is not None

        exported = tmp_path / "heightmap_e2e_heightmap.npz"
        assert not exported.exists()  # export only happens from __main__
        path = output.export_heightmap(z_min=0.45)
        data = np.load(path)

        heights = data["heights"]
        valid = heights[~np.isnan(heights)]
        assert valid.size > 0
        # top layer printed at z = 0.6 -> surface must be above the plane
        assert valid.min() >= 0.45 - 1e-9
        # nothing above the printed stack + margin
        assert valid.max() <= 0.6 + 0.2 + 0.1
        # corners outside the printed square must be missing
        assert np.isnan(heights).any()

    def test_run_simulation_without_heightmap_unchanged(self, tmp_path):
        from volco import run_simulation

        output = run_simulation(
            gcode=self.GCODE,
            printer_config=self.PRINTER,
            sim_config=self._sim_config(tmp_path),
        )
        assert output.mesh is not None
        assert output.heightmap is None
