#!/usr/bin/env python3
"""Insert a pause/home/dwell block every N layers into a G-code file.

Examples:
    python gcode_pause_2.py input.gcode 5
    python gcode_pause_2.py input.gcode 5 --output paused.gcode
    python gcode_pause_2.py input.gcode 10 --resume-command "M24"
    python gcode_pause_2.py input.gcode 10 --resume-command ""  # disable resume command

This script finds layer-change markers of the form:
    ; CHANGE_LAYER
    ; LAYER_HEIGHT: 0.2
    ; ...

and inserts a pause block after every Nth layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_pause_block(resume_command: str = "M24", x_pos: int = 200, y_pos: int = 200, z_lift: float = 10.0, dwell_ms: int = 10000) -> str:
    """Return a Bambu P1S-safe pause block as G-code."""
    lines = [
        "; === Bambu P1S-safe pause block ===",
        "M400 U1 ; pause print safely for Bambu firmware",
        "G91 ; switch to relative positioning",
        f"G1 Z{z_lift} F600 ; lift nozzle to avoid dragging",
        "G90 ; switch back to absolute positioning",
        f"G1 X{x_pos} Y{y_pos} F12000 ; move to a safe parking position",
        f"G4 P{dwell_ms} ; wait {dwell_ms / 1000.0:.0f} seconds",
        "G91 ; return to relative positioning",
        f"G1 Z-{z_lift} F600 ; return to print height",
        "G90 ; return to absolute positioning",
    ]
    if resume_command and resume_command.strip():
        lines.append(f"{resume_command.strip()} ; resume print")
    lines.append("; === end pause block ===")
    return "\n".join(lines) + "\n"


def find_layer_change_indices(lines: list[str]) -> list[int]:
    """Return indices of likely layer change markers in the G-code."""
    indices: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("; CHANGE_LAYER") or stripped.startswith(";LAYER_CHANGE"):
            indices.append(i)
    return indices


def insert_pause_every_n_layers(
    gcode_text: str,
    every_n: int,
    *,
    resume_command: str = "M24",
    park_x: int = 250,
    park_y: int = 250,
    z_lift: float = 2.0,
) -> str:
    """Insert a pause block after every Nth layer change."""
    if every_n <= 0:
        raise ValueError("every_n must be a positive integer")

    lines = gcode_text.splitlines()
    layer_indices = find_layer_change_indices(lines)
    if not layer_indices:
        raise ValueError("No layer-change markers found. Expected lines starting with '; CHANGE_LAYER'.")

    inserted_count = 0
    block = build_pause_block(resume_command=resume_command, x_pos=park_x, y_pos=park_y, z_lift=z_lift).splitlines()

    # Insert blocks after each layer marker that lands on a multiple of every_n.
    # Counting starts at layer 1, so after layer 5, 10, 15... etc.
    new_lines: list[str] = []
    layer_number = 0
    for idx, line in enumerate(lines):
        new_lines.append(line)
        if line.strip().startswith("; CHANGE_LAYER"):
            layer_number += 1
            if layer_number % every_n == 0:
                # insert after this layer-change marker, before the next layer's moves begin
                new_lines.extend([""] + block)
                inserted_count += 1

    if inserted_count == 0:
        raise ValueError(f"No inserts made. The file has {layer_number} detected layers, which are not multiples of {every_n}.")

    return "\n".join(new_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert a pause/home/dwell block into a G-code file every N layers.")
    parser.add_argument("gcode", type=Path, help="Path to the input G-code file")
    parser.add_argument("every_n", type=int, help="Insert pause after every Nth layer")
    parser.add_argument("--output", type=Path, help="Output file path. Defaults to <input>_paused_every_N.gcode")
    parser.add_argument("--resume-command", default="M24", help="Firmware resume command to append after the dwell. Use empty string to disable.")
    parser.add_argument("--park-x", type=float, default=200.0, help="X position for the safe parking move on a Bambu P1S")
    parser.add_argument("--park-y", type=float, default=200.0, help="Y position for the safe parking move on a Bambu P1S")
    parser.add_argument("--z-lift", type=float, default=10.0, help="Z lift distance in mm before parking on a Bambu P1S")

    args = parser.parse_args()

    if not args.gcode.exists():
        raise FileNotFoundError(f"G-code file not found: {args.gcode}")

    source = args.gcode.read_text(encoding="utf-8", errors="ignore")
    output_text = insert_pause_every_n_layers(
        source,
        args.every_n,
        resume_command=args.resume_command,
        park_x=int(args.park_x),
        park_y=int(args.park_y),
        z_lift=args.z_lift,
    )

    out_path = args.output or args.gcode.with_name(f"{args.gcode.stem}_paused_every_{args.every_n}{args.gcode.suffix}")
    out_path.write_text(output_text, encoding="utf-8")
    print(f"Inserted pause block every {args.every_n} layers into {out_path}")


if __name__ == "__main__":
    main()
