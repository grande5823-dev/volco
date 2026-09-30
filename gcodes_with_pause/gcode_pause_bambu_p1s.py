#!/usr/bin/env python3
"""Insert a Bambu P1S-friendly pause block into a G-code file.

The generated block does this:
    - pause print safely
    - move bed (or Z axis) downward to minimum Z
    - home XY/nozzle as needed for the printer
    - wait 15 seconds
    - resume print

This script is written for a Bambu P1S-style motion model, where Z motion is
usually relative to the toolhead/gantry and the bed stays fixed in normal print
operation. The generated block is intentionally conservative and avoids an
unsafe full G28 during the print.

Examples:
    python gcode_pause_bambu_p1s.py input.gcode
    python gcode_pause_bambu_p1s.py input.gcode --output paused.gcode
    python gcode_pause_bambu_p1s.py input.gcode --line 250
    python gcode_pause_bambu_p1s.py input.gcode --after "; CHANGE_LAYER"
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_pause_block(resume_command: str = "M24", dwell_ms: int = 15000) -> str:
    """Return a Bambu P1S-compatible pause block as G-code."""
    lines = [
        "; === Bambu P1S pause / park / dwell / resume ===",
        "M400 U1 ; pause print safely for Bambu firmware",
        "G91 ; switch to relative positioning",
        "G1 Z-999 F600 ; move bed/toolhead down to lowest safe Z position",
        "G90 ; switch back to absolute positioning",
        "G28 X Y ; home XY axes (safe nozzle park)",
        f"G4 P{dwell_ms} ; wait {dwell_ms / 1000.0:.0f} seconds",
        "G28 Z ; home Z after dwell",
        "G1 Z10 F600 ; lift nozzle slightly before continuing",
    ]
    if resume_command and resume_command.strip():
        lines.append(f"{resume_command.strip()} ; resume print")
    lines.append("; === end pause block ===")
    return "\n".join(lines) + "\n"


def insert_pause_block(gcode_text: str, *, insert_at: int | None = None, marker: str | None = None, resume_command: str = "M24") -> str:
    """Insert the pause block at the given location or at a marker."""
    block_lines = build_pause_block(resume_command=resume_command).splitlines()
    lines = gcode_text.splitlines()

    if marker:
        for idx, line in enumerate(lines):
            if marker in line:
                return "\n".join(lines[: idx + 1] + [""] + block_lines + lines[idx + 1 :]) + "\n"
        raise ValueError(f"Marker not found: {marker!r}")

    if insert_at is None:
        lines = lines + [""] + block_lines
    else:
        safe_index = max(0, min(insert_at, len(lines)))
        lines = lines[:safe_index] + [""] + block_lines + lines[safe_index:]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert a Bambu P1S pause block into G-code.")
    parser.add_argument("gcode", type=Path, help="Input G-code file path")
    parser.add_argument("--output", type=Path, help="Output G-code file path")
    parser.add_argument("--line", type=int, help="Insert at a 1-based line number")
    parser.add_argument("--after", type=str, help="Insert after the first line containing this text")
    parser.add_argument("--resume-command", default="M24", help="Resume command to append after dwell; set to empty string to disable")
    args = parser.parse_args()

    if not args.gcode.exists():
        raise FileNotFoundError(f"Input file not found: {args.gcode}")

    source = args.gcode.read_text(encoding="utf-8", errors="ignore")
    output = insert_pause_block(
        source,
        insert_at=(args.line - 1) if args.line is not None else None,
        marker=args.after,
        resume_command=args.resume_command,
    )

    if args.output:
        out_path = args.output
    else:
        out_path = args.gcode.with_name(f"{args.gcode.stem}_paused.gcode")

    out_path.write_text(output, encoding="utf-8")
    print(f"Inserted pause block into: {out_path}")


if __name__ == "__main__":
    main()
