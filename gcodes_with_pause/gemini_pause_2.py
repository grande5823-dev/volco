#!/usr/bin/env python3
"""Insert an automated Bambu P1S pause block every N layers into a G-code or .gcode.3mf file.

Examples:
    python gcode_pause_bambu_p1s_every_n.py input.gcode 15
    python gcode_pause_bambu_p1s_every_n.py input.gcode.3mf 15 --absolute-z 200
    python gcode_pause_bambu_p1s_every_n.py input.gcode 10 --output paused.gcode
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


def build_pause_block(
    current_z: float,
    absolute_z: float = 100.0,
    dwell_ms: int = 15000,
    retract_mm: float = 2.0,
    park_x: float = 128.0,
    park_y: float = 250.0,
) -> str:
    """Return an automated Bambu P1S pause block with optimized safe speeds."""
    safe_target_z = max(absolute_z, current_z + 10.0)

    lines = [
        "; === Bambu P1S automated absolute park & dwell ===",
        "M400 ; finish buffer moves",
        "G91 ; relative positioning for retraction",
        f"G1 E-{retract_mm:.2f} F2400 ; retract filament at 40 mm/s",
        "G90 ; absolute positioning",
        f"G1 Z{safe_target_z:.2f} F1200 ; drop bed at safe 20 mm/s",
        f"G1 X{park_x:.1f} Y{park_y:.1f} F18000 ; park toolhead rapidly at 300 mm/s",
        f"G4 P{dwell_ms} ; wait {dwell_ms / 1000.0:.1f} seconds",
        f"G1 Z{current_z:.2f} F1200 ; restore bed to print height at 20 mm/s",
        "G91 ; relative positioning for prime",
        f"G1 E{retract_mm:.2f} F2400 ; un-retract filament at 40 mm/s",
        "G90 ; restore absolute movement",
        "M83 ; enforce relative extrusion mode",
        "; === end pause block ===",
    ]
    return "\n".join(lines) + "\n"


def find_layer_markers(lines: list[str]) -> list[int]:
    """Return indices of layer-change markers in the G-code."""
    markers = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("; CHANGE_LAYER") or stripped.startswith(";LAYER_CHANGE"):
            markers.append(i)
    return markers


def insert_pause_every_n_layers(
    gcode_text: str, every_n: int, absolute_z: float = 200.0, dwell_ms: int = 15000
) -> str:
    """Insert the pause block after every Nth layer change."""
    if every_n <= 0:
        raise ValueError("every_n must be positive")

    lines = gcode_text.splitlines()
    layer_markers = find_layer_markers(lines)
    if not layer_markers:
        raise ValueError("No layer-change markers found; expected lines starting with '; CHANGE_LAYER' or ';LAYER_CHANGE'.")

    new_lines: list[str] = []
    layer_count = 0
    current_z = 0.0
    z_pattern = re.compile(r'\bZ([\d\.]+)')

    for line in lines:
        # Strip comments before checking for Z height updates
        code_part = line.split(";", 1)[0].strip()
        if code_part.startswith(("G0", "G1", "G00", "G01")):
            z_match = z_pattern.search(code_part)
            if z_match:
                current_z = float(z_match.group(1))

        new_lines.append(line)

        stripped = line.strip()
        if stripped.startswith("; CHANGE_LAYER") or stripped.startswith(";LAYER_CHANGE"):
            layer_count += 1
            if layer_count % every_n == 0:
                block = build_pause_block(current_z=current_z, absolute_z=absolute_z, dwell_ms=dwell_ms).splitlines()
                new_lines.extend([""] + block)

    if layer_count == 0:
        raise ValueError("No layers detected in file.")

    return "\n".join(new_lines) + "\n"


def add_pause_every_n_layers(
    input_gcode: str | Path,
    every_n: int,
    output_gcode: str | Path | None = None,
    absolute_z: float = 200.0,
    dwell_ms: int = 15000,
) -> Path:
    """Write a paused G-code or .gcode.3mf file and return its resolved path."""
    input_path = Path(input_gcode).expanduser()
    
    # Safely compute project root regardless of execution context
    try:
        project_root = Path(__file__).resolve().parent.parent
    except NameError:
        project_root = Path.cwd()

    candidates = [input_path]
    if not input_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / input_path,
                Path.cwd() / "gcode" / input_path,
                project_root / input_path,
                project_root / "gcode" / input_path,
            ]
        )

    source_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if source_path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Input file not found. Searched: {searched}")

    is_3mf = source_path.name.lower().endswith(".3mf")

    if output_gcode is None:
        output_path = source_path.parent / f"{source_path.stem}_paused_every_{every_n}{source_path.suffix}"
    else:
        output_path = Path(output_gcode).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_3mf:
        with zipfile.ZipFile(source_path, "r") as src, zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename.endswith(".gcode"):
                    gcode_text = data.decode("utf-8")
                    modified_gcode = insert_pause_every_n_layers(
                        gcode_text, every_n=every_n, absolute_z=absolute_z, dwell_ms=dwell_ms
                    )
                    dst.writestr(item.filename, modified_gcode.encode("utf-8"))
                else:
                    dst.writestr(item, data)
    else:
        source = source_path.read_text(encoding="utf-8", errors="ignore")
        output = insert_pause_every_n_layers(
            source, every_n=every_n, absolute_z=absolute_z, dwell_ms=dwell_ms
        )
        output_path.write_text(output, encoding="utf-8")

    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert an automated Bambu P1S pause block after every Nth layer")
    parser.add_argument("gcode", type=Path, help="Input G-code or .gcode.3mf file path")
    parser.add_argument("every_n", type=int, help="Insert pause after every Nth layer")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--absolute-z", type=float, default=200.0, help="Target absolute Z height (bed position in mm) during pause")
    parser.add_argument("--dwell-ms", type=int, default=15000, help="Pause duration in milliseconds")
    args = parser.parse_args()

    out_path = add_pause_every_n_layers(
        args.gcode,
        args.every_n,
        output_gcode=args.output,
        absolute_z=args.absolute_z,
        dwell_ms=args.dwell_ms,
    )
    print(f"Inserted automated pause blocks every {args.every_n} layers into: {out_path}")


if __name__ == "__main__":
    main()