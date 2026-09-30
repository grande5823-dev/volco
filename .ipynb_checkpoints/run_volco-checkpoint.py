#!/usr/bin/env python3
"""
run_volco.py
============
Converts a G-code file into a simulated-deposition STL mesh using the VOLCO
repository (https://github.com/weixuanzh/volco).

VOLCO walks through every extrusion move in the G-code and deposits a little
box of material for each one, so the output STL is a *realistic simulated
print* -- bumps, over-extrusion, layer seams and all -- not an idealized CAD
model. Because of that, one box per extrusion move, the output is usually a
big, high polygon-count ASCII STL (a full-size part can easily be 1-2 GB).
If you need something smaller/faster to open in a normal STL viewer, see the
"Shrinking the output" section in README.md.

Usage
-----
    python run_volco.py --gcode path/to/file.gcode [options]

This script does NOT install VOLCO or its dependencies for you -- follow
README.md in this folder first (git clone + pip install -r requirements.txt).
It just wraps the `python volco.py ...` call VOLCO itself expects, with
sensible defaults and some size/sanity reporting so it's easy to reuse across
different G-code files without re-typing the whole command each time.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gcode", required=True, type=Path, help="Path to the input .gcode file")
    p.add_argument("--repo", default=Path("volco"), type=Path,
                   help="Path to the cloned VOLCO repository (default: ./volco)")
    p.add_argument("--sim", default=None, type=Path,
                   help="Simulation settings JSON (default: <repo>/examples/simulation_settings.json, "
                        "VOLCO's full-resolution default -- this is what gives the ~1.7 GB output)")
    p.add_argument("--printer", default=None, type=Path,
                   help="Printer settings JSON (default: <repo>/examples/printer_settings.json)")
    p.add_argument("--outdir", default=None, type=Path,
                   help="Where to copy the resulting STL after VOLCO finishes "
                        "(default: leave it where VOLCO wrote it)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo = args.repo.resolve()
    print(repo)
    volco_py = repo / "volco.py"
    if not volco_py.exists():
        sys.exit(
            f"Can't find {volco_py}.\n"
            f"Did you clone the repo and pass the right --repo path? See README.md."
        )

    gcode = args.gcode.resolve()
    if not gcode.exists():
        sys.exit(f"G-code file not found: {gcode}")

    sim = (args.sim or repo / "examples" / "simulation_settings.json").resolve()
    printer = (args.printer or repo / "examples" / "printer_settings.json").resolve()
    for f in (sim, printer):
        if not f.exists():
            sys.exit(f"Settings file not found: {f}")

    cmd = [
        sys.executable, str(volco_py),
        f"--gcode={gcode}",
        f"--sim={sim}",
        f"--printer={printer}",
    ]
    print("Running:", " ".join(cmd))
    print(f"(cwd={repo})  -- this can take a few minutes for a large G-code file\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=repo)
    elapsed = time.time() - start

    if result.returncode != 0:
        sys.exit(f"VOLCO exited with code {result.returncode} after {elapsed:.1f}s")

    # VOLCO writes into <repo>/Results_<sim_name>/ by convention; find the newest STL there.
    candidates = sorted(repo.glob("Results_*/**/*.stl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print(f"Finished in {elapsed:.1f}s, but no output STL was found under {repo}/Results_*/ "
              f"-- check VOLCO's console output above.")
        return

    stl_path = candidates[0]
    size_mb = stl_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output STL: {stl_path}  ({size_mb:.1f} MB)")

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        dest = args.outdir / stl_path.name
        shutil.copy2(stl_path, dest)
        print(f"Copied to: {dest}")


if __name__ == "__main__":
    main()
