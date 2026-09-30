#!/usr/bin/env python3
"""
simplify_outer.py
==================
Shrinks a VOLCO deposition-simulation STL by keeping only its outer surface.

VOLCO's output is huge partly because overlapping filament passes leave a lot
of *internal* duplicate surface -- geometry you'd never see unless you cut
the part open. This script:

  1. Voxelizes the mesh surface (marks every voxel the mesh touches).
  2. Flood-fills only the FULLY ENCLOSED interior cavities (the internal
     print-texture noise) -- any air pocket that connects to the outside
     world through the part's actual through-holes is intentionally left
     alone, since that's real geometry, not noise.
  3. Re-extracts a clean outer surface with marching cubes.
  4. Optionally decimates (reduces triangle count) to hit a target file size.

Trade-off: finer --pitch keeps more real surface texture (the filament
ridges/bumps) but produces a bigger file; coarser --pitch is smaller but
smoother. Adjust --pitch and --target-tris together to land in whatever
file-size budget you need -- see README.md for example numbers.

Usage:
    python simplify_outer.py INPUT.stl OUTPUT.stl --pitch 0.08 --target-tris 1500000

Requires: open3d, numpy, scipy, scikit-image
"""

import argparse
import os

import numpy as np
import open3d as o3d
from scipy import ndimage
from skimage import measure


def simplify_outer(src: str, out: str, pitch: float, target_tris: int | None, pad: int = 4) -> None:
    print(f"loading {src}")
    mesh = o3d.io.read_triangle_mesh(src)
    min_b = mesh.get_min_bound()
    max_b = mesh.get_max_bound()
    print("bbox", min_b, max_b, "pitch", pitch)

    origin = min_b - pad * pitch
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=pitch,
        min_bound=origin, max_bound=max_b + pad * pitch,
    )
    voxels = vg.get_voxels()
    print("surface voxels:", len(voxels))

    dims = np.ceil((max_b + pad * pitch - origin) / pitch).astype(int) + 1
    occ = np.zeros(dims, dtype=bool)
    idx = np.array([v.grid_index for v in voxels])
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    print("occupancy grid shape", occ.shape, "-- filling enclosed cavities only ...")
    filled = ndimage.binary_fill_holes(occ)
    print("solid voxels before/after fill:", occ.sum(), filled.sum())

    print("extracting outer surface via marching cubes ...")
    verts, faces, _normals, _ = measure.marching_cubes(
        filled.astype(np.float32), level=0.5, spacing=(pitch, pitch, pitch)
    )
    verts = verts + origin

    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(verts)
    out_mesh.triangles = o3d.utility.Vector3iVector(faces)
    out_mesh.compute_vertex_normals()
    out_mesh.remove_duplicated_vertices()
    out_mesh.remove_degenerate_triangles()
    print("triangles before decimation:", len(out_mesh.triangles))

    if target_tris and len(out_mesh.triangles) > target_tris:
        out_mesh = out_mesh.simplify_quadric_decimation(target_tris)
        out_mesh.remove_duplicated_vertices()
        out_mesh.remove_degenerate_triangles()
        out_mesh.compute_vertex_normals()
    print("triangles after decimation:", len(out_mesh.triangles))

    o3d.io.write_triangle_mesh(out, out_mesh, write_ascii=False)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_stl")
    p.add_argument("output_stl")
    p.add_argument("--pitch", type=float, default=0.12,
                    help="Voxel size in mm. Smaller = more surface detail kept, bigger file. "
                         "~0.15mm for a small/smooth file, ~0.08mm to keep visible print texture. (default: 0.12)")
    p.add_argument("--target-tris", type=int, default=None,
                    help="Decimate to this many triangles after surface extraction. "
                         "Binary STL is ~50 bytes/triangle, so e.g. 1,500,000 tris ~= 75 MB. "
                         "Omit to skip decimation and keep whatever marching cubes produces.")
    p.add_argument("--pad", type=int, default=4,
                    help="Voxel margin around the part so hole air-paths reach the grid border "
                         "(needed for the enclosed-cavity fill to work correctly). Default: 4")
    args = p.parse_args()
    simplify_outer(args.input_stl, args.output_stl, args.pitch, args.target_tris, args.pad)


if __name__ == "__main__":
    main()
