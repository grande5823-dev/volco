#!/usr/bin/env python3
"""Trim a G-code file so that only the top N printed layers remain.

Preprocessing tool for VOLCO simulations that only target the top-surface
topography of a printed part. The simulator builds a voxel space spanning the
full Z range of the input file and deposits every extruded segment, so most of
the computation is wasted on bottom layers when only the top surface matters.
This script rewrites a G-code file keeping just its top layers, which:

  * removes all extrusions outside the top layers (purge/prime lines, brim,
    lower layers, calibration patterns in the machine start G-code),
  * removes the machine end G-code (final Z lifts inflate the voxel-space
    Z dimension even though they extrude nothing),
  * shifts the remaining layers down towards the bed so the voxel space is
    only as tall as the kept stack (unless --no-shift is given).

Semantics are preserved exactly, mirroring ``app/instructions/gcode.py``:

  * the kept section is normalized to absolute positioning (G90), millimeter
    units (G21) and relative extrusion (M83); every kept E value is rewritten
    to the exact per-move extrusion delta of the original file (absolute-E
    files with G92 resets are handled),
  * X/Y/Z of kept moves are rewritten to their tracked absolute coordinates
    (G91 relative sections are converted), Z values minus the Z shift,
  * arc moves (G2/G3) keep their I/J/R parameters untouched,
  * a positioning travel move plus G90/G21/M83/G92 E0 block is inserted right
    before the kept section so the simulator state is unambiguous.

Everything before the first extrusion of the file is kept verbatim (heater,
fan and homing setup, slicer metadata comments), so the trimmed file is still
printable on a real printer as well.

Usage:
    python trim_top_layers.py part.gcode                 # keep top 5 layers
    python trim_top_layers.py part.gcode -n 8            # keep top 8 layers
    python trim_top_layers.py part.gcode --height 1.0    # keep top ~1 mm
    python trim_top_layers.py a.gcode b.gcode            # batch processing
"""

import argparse
import os
import re
import sys

MOTION_WORDS = ("G0", "G1", "G2", "G3")
MODE_WORDS = ("G90", "G91", "M82", "M83", "G92", "G20", "G21")

# Same numeric syntax as app/instructions/gcode.py
NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)$")


def _format_number(value, max_decimals):
    """Compact fixed-point formatting (never scientific notation, no '-0')."""
    if abs(value) < 0.5 * 10 ** (-max_decimals):
        value = 0.0
    text = f"{value:.{max_decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


class _Line:
    """One analysed G-code line."""

    __slots__ = (
        "raw",
        "code",
        "comment",
        "word",
        "params",
        "is_motion",
        "has_e",
        "e_delta",
        "x",
        "y",
        "z",
        "feed",
        "z_key",
    )

    def __init__(self, raw):
        self.raw = raw
        self.word = None
        self.params = {}
        self.is_motion = False
        self.has_e = False
        self.e_delta = 0.0
        self.x = self.y = self.z = 0.0  # absolute position after this line
        self.feed = None  # modal feedrate (mm/min) after this line
        self.z_key = None  # layer key if this line deposits material

        body = raw.lstrip("﻿")
        code, _, comment = body.partition(";")
        self.comment = comment
        self.code = code.rstrip()

        tokens = code.upper().split()
        # Drop a leading line number (N...)
        if (
            tokens
            and tokens[0].startswith("N")
            and len(tokens[0]) > 1
            and tokens[0][1:].isdigit()
        ):
            tokens.pop(0)
        if not tokens:
            return

        self.word = tokens[0]
        self.is_motion = self.word in MOTION_WORDS
        for token in tokens[1:]:
            token = token.split("*", 1)[0]  # drop checksums
            if len(token) < 2:
                continue
            letter, value_text = token[0], token[1:]
            if letter not in "XYZEFIJKR":
                continue
            if not NUMBER_RE.match(value_text):
                continue  # vendor syntax etc.; irrelevant to the simulator
            self.params[letter] = float(value_text)
        self.has_e = "E" in self.params


class _Trimmer:
    """Tracks printer state while scanning a file, mirroring the parser."""

    def __init__(self, text, source_name="<input>"):
        self.lines = [_Line(raw) for raw in text.splitlines()]
        self.source_name = source_name
        # Printer state, mirroring app/instructions/gcode.py:
        # positioning/extrusion units and modes are tracked independently.
        self.x = self.y = self.z = 0.0
        self.e_reference = 0.0  # absolute-E register (M82 mode)
        self.absolute_positioning = True  # G90
        self.absolute_extrusion = True  # M82
        self.feed = None  # mm/min, like the parser's vprint (but in mm/min)
        self.arc_plane = "XY"  # G17
        self.first_e_line_index = None  # first motion line carrying an E value
        self.last_e_line_index = None
        self.deposit_z_keys = []  # rounded Z of every material-depositing move

    def scan(self):
        for index, line in enumerate(self.lines):
            self._apply(line, index)
        if not self.deposit_z_keys:
            raise ValueError(
                f"{self.source_name}: the file contains no extrusion moves"
            )
        assert self.first_e_line_index is not None
        assert self.last_e_line_index is not None

    # -- state tracking ------------------------------------------------------

    def _apply(self, line, index):
        line.x, line.y, line.z = self.x, self.y, self.z
        word = line.word
        params = line.params

        if "F" in params:
            self.feed = params["F"]

        if word == "G90":
            self.absolute_positioning = True
        elif word == "G91":
            self.absolute_positioning = False
        elif word == "G20":
            raise ValueError(
                f"{self.source_name}: G20 (inch units) is not supported"
            )
        elif word == "G17":
            self.arc_plane = "XY"
        elif word in ("G18", "G19"):
            self.arc_plane = "XZ" if word == "G18" else "YZ"
        elif word == "M82":
            self.absolute_extrusion = True
        elif word == "M83":
            self.absolute_extrusion = False
        elif word == "G92":
            if "E" in params:
                self.e_reference = params["E"]

        if not line.is_motion:
            line.feed = self.feed
            return

        if word in ("G2", "G3") and self.arc_plane != "XY":
            # Skipped by the simulator; only the extrusion reference moves on.
            if "E" in params and self.absolute_extrusion:
                self.e_reference = params["E"]
            line.feed = self.feed
            return

        if "E" in params:
            e_now = params["E"]
            if self.absolute_extrusion:
                line.e_delta = e_now - self.e_reference
                self.e_reference = e_now
            else:
                line.e_delta = e_now

        moved = False
        for axis in "XYZ":
            if axis in params:
                moved = True
                value = params[axis]
                if not self.absolute_positioning:
                    value += getattr(self, axis.lower())
                setattr(self, axis.lower(), value)
        if word in ("G2", "G3"):
            moved = True  # arcs always travel (full circle if no X/Y given)

        line.x, line.y, line.z = self.x, self.y, self.z
        line.feed = self.feed

        if line.has_e:
            if self.first_e_line_index is None:
                self.first_e_line_index = index
            self.last_e_line_index = index
        if moved and line.e_delta > 0.0:
            line.z_key = round(self.z, 3)
            self.deposit_z_keys.append(line.z_key)

    # -- trimming ------------------------------------------------------------

    def build_trimmed_text(self, n_layers=None, height=None, shift_z=True):
        layer_zs = sorted(set(self.deposit_z_keys))
        z_top = layer_zs[-1]

        if height is not None:
            candidates = [z for z in layer_zs if z_top - z <= height + 1e-9]
            z_cut = candidates[0] if candidates else layer_zs[0]
        else:
            if n_layers is None:
                n_layers = 5
            z_cut = layer_zs[max(0, len(layer_zs) - n_layers)]
        n_layers = len(layer_zs) - layer_zs.index(z_cut)

        cut_position = layer_zs.index(z_cut)
        if cut_position > 0:
            first_layer_height = z_cut - layer_zs[cut_position - 1]
        else:
            first_layer_height = z_cut  # everything kept; nothing to shift
        z_shift = (z_cut - first_layer_height) if shift_z else 0.0

        # First kept deposit, then backtrack to the line that moved Z there.
        cut_index = None
        for index, line in enumerate(self.lines):
            if line.z_key is not None and line.z_key >= z_cut:
                cut_index = index
                break
        if cut_index is None:  # pragma: no cover - defensive
            raise RuntimeError("could not locate the first layer to keep")
        for index in range(cut_index - 1, -1, -1):
            candidate = self.lines[index]
            if candidate.is_motion and "Z" in candidate.params:
                cut_index = index
                break

        header_end = self.first_e_line_index
        end_index = self.last_e_line_index
        assert header_end is not None and end_index is not None
        cut_index = max(cut_index, header_end)

        out = []
        out.append("; ------------------------------------------------")
        out.append(
            f"; Trimmed to the top {n_layers} layer(s)"
            f" (original Z {z_cut:g}..{z_top:g} mm) by trim_top_layers.py"
        )
        out.append(
            f"; Kept layers are shifted down by {z_shift:g} mm"
            if shift_z
            else "; Kept layers are left at their original Z heights"
        )
        out.append("; ------------------------------------------------")

        warned_g92 = False
        # Header travels are kept verbatim except for Z moves that climb far
        # above the kept stack (prime/wipe routines, bed lowering): nothing is
        # deposited in the header, but such moves would still inflate the
        # voxel-space Z dimension, undoing the point of the trim.
        z_cap = (z_top - z_shift) + 1.0
        for line in self.lines[:header_end]:
            if line.word == "G92" and (
                any(a in line.params for a in "XYZ")
                or line.params.get("E", 0.0) != 0.0
            ):
                out.append(f"; stripped unsupported G92: {line.raw}")
                warned_g92 = True
            else:
                out.append(self._rewrite_header_line(line, z_cap))
        if warned_g92:
            print(
                f"warning: {self.source_name}: unsupported G92 line(s) in "
                "the header were commented out",
                file=sys.stderr,
            )

        # Normalization + positioning block, so parser state is unambiguous.
        pre = self.lines[cut_index - 1] if cut_index > 0 else None
        start_x = pre.x if pre else 0.0
        start_y = pre.y if pre else 0.0
        start_z = (pre.z if pre else 0.0) - z_shift
        feed = pre.feed if pre and pre.feed else 1800.0
        out.append("G90 ; trimmed file: absolute positioning")
        out.append("G21 ; trimmed file: millimeter units")
        out.append("M83 ; trimmed file: relative extrusion")
        out.append("G92 E0")
        out.append(f"G1 F{_format_number(feed, 3)}")
        out.append(
            "G1"
            f" X{_format_number(start_x, 4)}"
            f" Y{_format_number(start_y, 4)}"
            f" Z{_format_number(start_z, 4)} ; travel to trim point"
        )

        # Kept section.
        for line in self.lines[cut_index : end_index + 1]:
            out.append(self._rewrite_kept_line(line, z_shift))

        stats = {
            "layers_total": len(layer_zs),
            "layers_kept": n_layers,
            "z_cut": z_cut,
            "z_top": z_top,
            "z_shift": z_shift,
            "first_layer_height": first_layer_height,
            "lines_in": len(self.lines),
            "lines_out": len(out),
            "extrusion_moves_kept": sum(
                1
                for line in self.lines[cut_index : end_index + 1]
                if line.is_motion and line.e_delta > 0.0
            ),
        }
        return "\n".join(out) + "\n", stats

    def _rewrite_header_line(self, line, z_cap):
        """Strip stale checksums and above-cap Z moves from a header line."""
        code = line.code
        comment = line.comment
        if line.is_motion and "Z" in line.params and line.z > z_cap:
            code = " ".join(
                token
                for token in code.split()
                if not token.upper().startswith("Z")
            )
            comment = f"Z move above kept stack removed ;{comment}"
        if "*" in code:
            # Stale checksum (the simulator rejects malformed numbers).
            code = code.split("*", 1)[0].rstrip()
        if comment:
            return f"{code} ;{comment}"
        return code

    def _rewrite_kept_line(self, line, z_shift):
        if not line.is_motion:
            if line.word in MODE_WORDS:
                return f"; stripped mode command: {line.raw.strip()}"
            return line.raw

        tokens = line.code.split()
        rebuilt = []
        if (
            tokens
            and tokens[0].startswith("N")
            and len(tokens[0]) > 1
            and tokens[0][1:].isdigit()
        ):
            tokens.pop(0)  # checksums would no longer match anyway
        for token in tokens:
            token = token.split("*", 1)[0]
            if not token:
                continue
            letter = token[0].upper()
            if letter == "X" and "X" in line.params:
                token = "X" + _format_number(line.x, 4)
            elif letter == "Y" and "Y" in line.params:
                token = "Y" + _format_number(line.y, 4)
            elif letter == "Z" and "Z" in line.params:
                token = "Z" + _format_number(line.z - z_shift, 4)
            elif letter == "E" and "E" in line.params:
                token = "E" + _format_number(line.e_delta, 6)
            rebuilt.append(token)
        new_code = " ".join(rebuilt)
        if line.comment:
            return f"{new_code} ;{line.comment}"
        return new_code


def trim_text(text, n_layers=None, height=None, shift_z=True, source_name="<input>"):
    """Trim G-code content given as a string; returns (text, stats)."""
    trimmer = _Trimmer(text, source_name=source_name)
    trimmer.scan()
    return trimmer.build_trimmed_text(
        n_layers=n_layers, height=height, shift_z=shift_z
    )


def _default_output_path(input_path, stats):
    root, _ = os.path.splitext(input_path)
    return f"{root}_top{stats['layers_kept']}layers.gcode"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Trim G-code files so that only the top printed layers remain, "
            "for fast VOLCO top-surface simulations."
        )
    )
    parser.add_argument("inputs", nargs="+", help="G-code file(s) to trim")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "-n",
        "--layers",
        type=int,
        default=5,
        help="number of top layers to keep (default: 5)",
    )
    selection.add_argument(
        "--height",
        type=float,
        default=None,
        help="height in mm of the top region to keep (alternative to -n)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file (only valid with a single input); "
        "default: <input>_top<N>layers.gcode",
    )
    parser.add_argument(
        "--out-dir",
        help="directory for output files (default: next to each input)",
    )
    parser.add_argument(
        "--no-shift",
        dest="shift_z",
        action="store_false",
        help="keep the original Z heights (NOT recommended: the voxel space "
        "will still span the full original height)",
    )
    args = parser.parse_args(argv)

    if args.layers is not None and args.layers < 1:
        parser.error("--layers must be >= 1")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be > 0")
    if args.output and len(args.inputs) > 1:
        parser.error("-o/--output is only valid with a single input file")

    status = 0
    for input_path in args.inputs:
        try:
            with open(input_path, "r") as handle:
                text = handle.read()
            trimmed, stats = trim_text(
                text,
                n_layers=args.layers,
                height=args.height,
                shift_z=args.shift_z,
                source_name=input_path,
            )
        except (OSError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            status = 1
            continue

        if args.output:
            output_path = args.output
        elif args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            output_path = os.path.join(
                args.out_dir, os.path.basename(_default_output_path(input_path, stats))
            )
        else:
            output_path = _default_output_path(input_path, stats)

        with open(output_path, "w") as handle:
            handle.write(trimmed)

        print(
            f"{input_path} -> {output_path}\n"
            f"  layers: kept top {stats['layers_kept']} of "
            f"{stats['layers_total']} (Z {stats['z_cut']:g}..{stats['z_top']:g} mm)\n"
            f"  Z shift: {stats['z_shift']:g} mm"
            f" (first kept layer now at {stats['first_layer_height']:g} mm)\n"
            f"  extrusion moves kept: {stats['extrusion_moves_kept']}, "
            f"lines: {stats['lines_in']} -> {stats['lines_out']}"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
