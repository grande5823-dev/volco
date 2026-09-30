#!/usr/bin/env python3
"""Insert a pause/home/dwell/resume block into a G-code file.

This script adds a standard print pause sequence:
    M0 ; pause print
    G28 ; move axes to home
    G4 P10000 ; wait 10 seconds
    M24 ; resume print (firmware dependent)

Usage examples:
    python gcode_pause_1.py input.gcode
    python gcode_pause_1.py input.gcode --output paused.gcode
    python gcode_pause_1.py input.gcode --line 250
    python gcode_pause_1.py input.gcode --after ";LAYER_CHANGE"
    python gcode_pause_1.py input.gcode --resume-command ""  # disable automatic resume command
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


def insert_block(lines: list[str], insert_at: int | None, marker: str | None) -> list[str]:
    """Insert the block into the G-code at the given line or after a marker."""
    block = build_pause_block().splitlines()

    if marker:
        for idx, line in enumerate(lines):
            if marker in line:
                return lines[: idx + 1] + block + lines[idx + 1 :]
        raise ValueError(f"Marker not found: {marker!r}")

    if insert_at is None:
        # Default: append at the end.
        return lines + [""] + block

    safe_index = max(0, min(insert_at, len(lines)))
    return lines[:safe_index] + [""] + block + lines[safe_index:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert a print pause/home/delay block into a G-code file.")
    parser.add_argument("gcode", type=Path, help="Input G-code file path")
    parser.add_argument("--output", type=Path, help="Output file path (default: <input>_paused.gcode)")
    parser.add_argument("--line", type=int, help="1-based line number where the block should be inserted")
    parser.add_argument("--after", type=str, help="Insert the block after the first line containing this text")
    parser.add_argument(
        "--resume-command",
        default="M24",
        help="Resume command to append after the dwell. Set to an empty string to disable it.",
    )
    args = parser.parse_args()

    if not args.gcode.exists():
        raise FileNotFoundError(f"G-code file not found: {args.gcode}")

    source = args.gcode.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()

    # Let the block be created from the user-selected resume command.
    block_lines = build_pause_block(args.resume_command).splitlines()
    if args.after:
        insertion_index = None
        for idx, line in enumerate(lines):
            if args.after in line:
                insertion_index = idx + 1
                break
        if insertion_index is None:
            raise ValueError(f"Marker not found: {args.after!r}")
        lines = lines[:insertion_index] + [""] + block_lines + lines[insertion_index:]
    elif args.line is not None:
        safe_index = max(0, min(args.line - 1, len(lines)))
        lines = lines[:safe_index] + [""] + block_lines + lines[safe_index:]
    else:
        lines = lines + [""] + block_lines

    output_path = args.output or args.gcode.with_name(f"{args.gcode.stem}_paused{args.gcode.suffix}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Inserted pause/home/delay block into: {output_path}")


if __name__ == "__main__":
    main()
