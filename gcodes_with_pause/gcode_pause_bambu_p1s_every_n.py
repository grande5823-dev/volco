#!/usr/bin/env python3
"""Insert a Bambu P1S-friendly pause block every N layers into a G-code file.

Examples:
    python gcode_pause_bambu_p1s_every_n.py input.gcode 15
    python gcode_pause_bambu_p1s_every_n.py input.gcode 15 --output paused.gcode
    python gcode_pause_bambu_p1s_every_n.py input.gcode 10 --resume-command ""
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


def find_layer_markers(lines: list[str]) -> list[int]:
    """Return indices of likely layer-change markers in the G-code."""
    markers = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("; CHANGE_LAYER") or stripped.startswith(";LAYER_CHANGE"):
            markers.append(i)
    return markers


def insert_pause_every_n_layers(gcode_text: str, every_n: int, resume_command: str = "M24") -> str:
    """Insert the pause block after every Nth layer change."""
    if every_n <= 0:
        raise ValueError("every_n must be positive")

    lines = gcode_text.splitlines()
    layer_markers = find_layer_markers(lines)
    if not layer_markers:
        raise ValueError("No layer-change markers found; expected lines starting with '; CHANGE_LAYER'.")

    block = build_pause_block(resume_command=resume_command).splitlines()
    new_lines: list[str] = []
    layer_count = 0

    for line in lines:
        new_lines.append(line)
        if line.strip().startswith("; CHANGE_LAYER"):
            layer_count += 1
            if layer_count % every_n == 0:
                new_lines.extend([""] + block)

    if layer_count == 0:
        raise ValueError("No layers detected in file.")

    return "\n".join(new_lines) + "\n"


def add_pause_every_n_layers(
    input_gcode: str | Path,
    every_n: int,
    output_gcode: str | Path | None = None,
    resume_command: str = "M24",
) -> Path:
    """Write a paused G-code file and return its path.

    A bare input filename is searched for in the current directory and the
    project-level ``gcode`` directory, so it can be called from JupyterLab.
    """
    input_path = Path(input_gcode).expanduser()
    project_root = Path(__file__).resolve().parent.parent
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
        raise FileNotFoundError(f"Input G-code not found. Searched: {searched}")

    source = source_path.read_text(encoding="utf-8", errors="ignore")
    output = insert_pause_every_n_layers(source, every_n, resume_command=resume_command)

    if output_gcode is None:
        output_path = Path(__file__).resolve().parent / (
            f"{source_path.stem}_paused_every_{every_n}{source_path.suffix}"
        )
    else:
        output_path = Path(output_gcode).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert a Bambu P1S pause block after every Nth layer")
    parser.add_argument("gcode", type=Path, help="Input G-code file path")
    parser.add_argument("every_n", type=int, help="Insert pause after every Nth layer")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--resume-command", default="M24", help="Resume command to append after dwell; set to empty string to disable")
    args = parser.parse_args()

    out_path = add_pause_every_n_layers(
        args.gcode,
        args.every_n,
        output_gcode=args.output,
        resume_command=args.resume_command,
    )
    print(f"Inserted pause blocks every {args.every_n} layers into: {out_path}")


if __name__ == "__main__":
    main()
