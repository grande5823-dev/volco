import logging
import math
import re

from app.instructions.instruction import Instruction

logger = logging.getLogger(__name__)


# The following checks are currently performed during gcode parsing:
# 1. UTF-8 BOM removal (line_raw = line_raw.lstrip('﻿'))
# 2. Comment stripping (line = line_raw.split(";", 1)[0])
# 3. Case normalization (line = line.upper())
# 4. Whitespace handling (line = line.strip())
# 5. Line number handling (N-code removal)
# 6. Parameter value validation:
#    - Missing numeric values for parameters
#    - Malformed numeric values (regex check)
#    Validation is strict only for commands consumed by the simulator
#    (G0/G1 movements, G92 position resets). Parameters of all other
#    commands are parsed best-effort, so vendor-specific commands with
#    non-numeric arguments (e.g. "M1002 gcode_claim_action : 29") are
#    tolerated.
# 7. Unit conversion (inches to mm)
# 8. Positioning mode tracking (absolute vs relative)
# 9. Extrusion mode tracking (absolute vs relative)
# 10. Position reset handling (G92)
# 11. Unsupported M-code warnings
# 12. Movement coordinate calculation based on modes
# 13. G2/G3 arc movements in the XY plane (G17) are tessellated into linear
#     segments of up to `arc_segment_length` (I/J and R forms, optional
#     helical Z). Extrusion is distributed proportionally to segment length.
#     Malformed arcs are skipped with a warning (their extrusion reference is
#     still consumed so subsequent absolute extrusion deltas stay correct).
#     Arcs in other planes (G18/G19) are not supported and are skipped.


class Gcode(Instruction):
    def __init__(self, gcode_path=None, gcode_content=None, default_nozzle_speed=40.0, printer=None, arc_segment_length=0.1):
        self.gcode_path = gcode_path
        self.gcode_content = gcode_content
        self._movements = list()
        self._coordinate_limits = {}
        self._number_printed_filaments = 0
        self._filaments_coordinates = list()
        self._default_nozzle_speed = default_nozzle_speed
        self._printer = printer  # Needed for E to volume conversion
        self.arc_segment_length = arc_segment_length

    @property
    def movements(self):
        return self._movements

    @property
    def coordinate_limits(self):
        return self._coordinate_limits

    @property
    def number_printed_filaments(self):
        return self._number_printed_filaments

    @property
    def filaments_coordinates(self):
        return self._filaments_coordinates

    @property
    def default_nozzle_speed(self):
        return self._default_nozzle_speed

    def read(self):
        # Default values
        flag_relative = 0  # G90 -> absolute printing
        e_relative = 0  # M82 -> absolute extrusion
        unit_mode = 'mm'  # default units ('mm' or 'inches')
        arc_plane = 'XY'  # G17 default; only the XY plane is supported for arcs

        # Coord list = [Xabs, Yabs, Zabs, Erelative, Vprint]
        movements = list()
        movements.append([0.0, 0.0, 0.0, 0.0, self.default_nozzle_speed])

        vprint = self.default_nozzle_speed

        extrusion_old = 0.0

        logger.info("Processing .gcode ...")

        # Determine if we're reading from a file or from content
        if self.gcode_path:
            gcode_lines = open(self.gcode_path, "r")
        elif self.gcode_content:
            gcode_lines = self.gcode_content.splitlines()
        else:
            raise ValueError("Either gcode_path or gcode_content must be provided")

        for line_raw in gcode_lines:
            # Strip UTF-8 BOM if present
            line_raw = line_raw.lstrip('\ufeff')
            # 1. Strip comments
            line = line_raw.split(";", 1)[0]
            # 2. Convert to uppercase
            line = line.upper()
            # 3. Strip leading/trailing whitespace
            line = line.strip()

            if not line:  # Skip empty lines or lines that were only comments
                continue

            parts = line.split()

            # 4. Handle line numbers (N codes)
            if parts[0].startswith('N') and len(parts[0]) > 1 and parts[0][1:].isdigit():
                parts.pop(0)
                if not parts:  # Line might have only contained N number
                    continue

            command = parts[0]
            # Parameters are consumed only for movement and position-reset
            # commands; validate strictly there and tolerate anything else
            # (vendor-specific commands, junk tokens, ...).
            strict_param_command = command in ("G0", "G1", "G92")
            params = {}
            for part in parts[1:]:
                # 5. Missing numeric value
                if len(part) == 1:
                    if strict_param_command and part in "XYZEF":
                        logger.warning(
                            f"Missing numeric value for parameter '{part}' from line: {line_raw.strip()}")
                        raise ValueError("Please correct gcode format and retry")
                    continue
                # I, J, K, R parameters are consumed by arc movements (G2/G3)
                if part[0] not in "GMXYZEFIJKR":
                    # Silently ignore parameters we don't understand
                    continue

                param_letter = part[0]
                val_str = part[1:]
                # 6. Malformed number
                if not re.match(r'^[-+]?(?:\d+\.?\d*|\.\d+)$', val_str):
                    if strict_param_command and param_letter in "XYZEF":
                        logger.warning(
                            f"Malformed numeric value '{val_str}' in param '{part}' from line: {line_raw.strip()}")
                        raise ValueError(
                            "Please correct gcode format and retry")
                    continue
                try:
                    params[param_letter] = float(val_str)
                except ValueError:
                    if strict_param_command and param_letter in "XYZEF":
                        logger.warning(
                            f"Could not parse parameter value in '{part}' from line: {line_raw.strip()}")
                        raise ValueError(
                            "Please correct gcode format and retry")
                    continue

            # Convert coordinates from inches to mm if needed
            if unit_mode == 'inches':
                for axis in ("X", "Y", "Z", "I", "J", "K", "R"):
                    if axis in params:
                        params[axis] *= 25.4

            # Update feedrate if F parameter exists
            if "F" in params:
                vprint = params["F"] / 60.0  # transforming to mm/s

            # Update positioning modes
            if command == "G90":
                flag_relative = 0  # G90 -> absolute positioning
            elif command == "G91":
                flag_relative = 1  # G91 -> relative positioning
            # Units: G20 (inches), G21 (mm)
            elif command == "G20":
                unit_mode = 'inches'
            elif command == "G21":
                unit_mode = 'mm'
            # Arc plane selection
            elif command == "G17":
                arc_plane = 'XY'
            elif command == "G18" or command == "G19":
                arc_plane = 'XZ' if command == "G18" else 'YZ'
                logger.warning(
                    f"Arc plane '{command}' is not supported (only G17/XY); "
                    "arc movements will be skipped")

            # Update extrusion modes
            if command == "M82":
                e_relative = 0  # M82 -> absolute extrusion
            elif command == "M83":
                e_relative = 1  # M83 -> relative extrusion
            # Warn on unsupported M-codes
            elif command.startswith("M"):
                logger.warning(
                    f"Unsupported M-code '{command}' on line: {line_raw.strip()}")

            # Handle position reset (critical for absolute extrusion)
            if command == "G92":
                # Handle G92 resets
                if "E" in params:
                    if params["E"] == 0.0:
                        extrusion_old = 0.0
                        logger.debug("Resetting extrusion reference (G92 E0)")
                    else:
                        logger.warning(
                            f"G92 sets E to non-zero ({params['E']}) on line: {line_raw.strip()} – unexpected extrusion reset")
                        raise ValueError(
                            "Please correct gcode format and retry")
                if any(axis in params for axis in ("X", "Y", "Z")):
                    axes = [axis for axis in params if axis in "XYZ"]
                    logger.warning(
                        f"G92 resets position axes {axes} on line: {line_raw.strip()} – unsupported coordinate reset")
                    raise ValueError("Please correct gcode format and retry")

            # Handle movement commands
            elif command == "G1" or command == "G0":
                # Check if there's actual movement data
                if any(axis in params for axis in "XYZE"):
                    coord_new, extrusion_old = self._define_movement(
                        params,  # Pass parsed parameters
                        movements[-1],
                        flag_relative,
                        e_relative,
                        vprint,
                        extrusion_old,
                    )
                    movements.append(coord_new)

            # Arc movements: tessellated into linear segments
            elif command == "G2" or command == "G3":
                if any(axis in params for axis in "XYZE"):
                    if arc_plane != 'XY':
                        logger.warning(
                            f"Skipping arc movement in unsupported plane on line: {line_raw.strip()}")
                        extrusion_old = self._consume_extrusion(
                            params, e_relative, extrusion_old)
                    else:
                        arc_movements, extrusion_old = self._define_arc_movements(
                            command,
                            params,
                            movements[-1],
                            flag_relative,
                            e_relative,
                            vprint,
                            extrusion_old,
                        )
                        movements.extend(arc_movements)

        # Close file if we opened one
        if self.gcode_path:
            gcode_lines.close()

        # Ensure coordinate limits calculation happens *after* the loop
        xlim, ylim, zlim, nfil, coord_fil = self._max_min_extru_coordinates(
            movements)

        logger.info("Done processing .gcode!")

        self._number_printed_filaments = nfil
        self._movements = movements
        self._coordinate_limits = {"x": xlim, "y": ylim, "z": zlim}
        self._filaments_coordinates = coord_fil

    def _define_movement(
        self, params, coord_old, flag_relative, e_relative, vprint, extrusion_old
    ):
        # Initialize new coordinates with old ones
        xnew, ynew, znew = coord_old[0], coord_old[1], coord_old[2]
        enew = 0.0  # Default to no extrusion for this move

        # Update coordinates based on parameters and relative/absolute mode
        if "X" in params:
            xnew = params["X"] + flag_relative * coord_old[0]
        if "Y" in params:
            ynew = params["Y"] + flag_relative * coord_old[1]
        if "Z" in params:
            znew = params["Z"] + flag_relative * coord_old[2]

        # Handle extrusion
        if "E" in params:
            enow = params["E"]
            if e_relative == 0:  # Absolute extrusion
                enew = enow - extrusion_old  # Calculate relative extrusion for this move
                extrusion_old = enow  # Update the absolute reference
            else:  # Relative extrusion
                enew = enow
                # extrusion_old doesn't need updating in relative mode, but G92 E0 resets it

        # Store the calculated movement details
        # Note: enew here represents the *relative* extrusion for this specific segment
        coord_new = [xnew, ynew, znew, enew, vprint]

        return coord_new, extrusion_old  # Return updated absolute reference for E

    def _consume_extrusion(self, params, e_relative, extrusion_old):
        """
        Update the absolute extrusion reference for a command whose motion is
        skipped, so that subsequent absolute extrusion deltas stay correct.
        """
        if "E" in params and e_relative == 0:
            return params["E"]

        return extrusion_old

    def _define_arc_movements(
        self, command, params, coord_old, flag_relative, e_relative, vprint,
        extrusion_old
    ):
        """
        Tessellate a G2 (clockwise) / G3 (counter-clockwise) arc in the XY
        plane into linear segments. Supports the centre-offset (I/J) form, the
        radius (R) form, and helical arcs with a linear Z change. Extrusion
        and Z change are distributed proportionally to segment length.
        """
        x0, y0, z0 = coord_old[0], coord_old[1], coord_old[2]

        x1, y1, z1 = x0, y0, z0
        if "X" in params:
            x1 = params["X"] + flag_relative * x0
        if "Y" in params:
            y1 = params["Y"] + flag_relative * y0
        if "Z" in params:
            z1 = params["Z"] + flag_relative * z0

        enew = 0.0
        if "E" in params:
            enow = params["E"]
            if e_relative == 0:  # Absolute extrusion
                enew = enow - extrusion_old
                extrusion_old = enow
            else:  # Relative extrusion
                enew = enow

        clockwise = command == "G2"

        if "R" in params:
            centre = self._arc_centre_from_radius(x0, y0, x1, y1, params["R"], clockwise)
            if centre is None:
                logger.warning(
                    f"Skipping malformed arc (invalid R form) on movement from "
                    f"({x0}, {y0}) to ({x1}, {y1})")
                return [], extrusion_old
        elif "I" in params or "J" in params:
            centre = (x0 + params.get("I", 0.0), y0 + params.get("J", 0.0))
        else:
            logger.warning(
                f"Skipping malformed arc (neither I/J nor R parameters) on "
                f"movement towards ({x1}, {y1})")
            return [], extrusion_old

        centre_x, centre_y = centre
        radius_start = math.hypot(x0 - centre_x, y0 - centre_y)
        radius_end = math.hypot(x1 - centre_x, y1 - centre_y)

        if radius_start <= 0.0 or radius_end <= 0.0:
            logger.warning(
                f"Skipping malformed arc (zero radius) on movement from "
                f"({x0}, {y0}) to ({x1}, {y1})")
            return [], extrusion_old

        angle_start = math.atan2(y0 - centre_y, x0 - centre_x)
        angle_end = math.atan2(y1 - centre_y, x1 - centre_x)

        if x1 == x0 and y1 == y0:
            sweep = 2.0 * math.pi  # full circle
        elif clockwise:
            sweep = (angle_start - angle_end) % (2.0 * math.pi)
        else:
            sweep = (angle_end - angle_start) % (2.0 * math.pi)

        number_segments = max(
            1,
            int(math.ceil(sweep * 0.5 * (radius_start + radius_end)
                          / self.arc_segment_length)),
        )

        arc_movements = []
        for step in range(1, number_segments + 1):
            fraction = step / number_segments

            if clockwise:
                angle = angle_start - sweep * fraction
            else:
                angle = angle_start + sweep * fraction

            # linear interpolation of the radius covers imprecise endpoints
            radius = radius_start + (radius_end - radius_start) * fraction

            arc_movements.append([
                centre_x + radius * math.cos(angle),
                centre_y + radius * math.sin(angle),
                z0 + (z1 - z0) * fraction,
                enew / number_segments,
                vprint,
            ])

        return arc_movements, extrusion_old

    def _arc_centre_from_radius(self, x0, y0, x1, y1, radius_param, clockwise):
        """
        Compute the arc centre for the R form. R > 0 selects the minor arc
        (sweep <= pi), R < 0 the major arc. Returns None for malformed input.
        """
        dx, dy = x1 - x0, y1 - y0
        chord = math.hypot(dx, dy)

        if chord == 0.0 or radius_param == 0.0:
            return None

        radius = abs(radius_param)
        if chord > 2.0 * radius * (1.0 + 1e-12):
            return None

        height = math.sqrt(max(radius**2 - (chord * 0.5) ** 2, 0.0))
        mid_x, mid_y = (x0 + x1) * 0.5, (y0 + y1) * 0.5

        # left normal of the chord direction
        left_x, left_y = -dy / chord, dx / chord

        # CW arcs keep the centre right of the travel direction for R > 0;
        # negative R flips the side.
        side = (-1.0 if clockwise else 1.0) * (1.0 if radius_param > 0.0 else -1.0)

        return (mid_x + side * height * left_x, mid_y + side * height * left_y)

    """
    This function returns the minimum and maximum printing coordinates;
    """

    def _max_min_extru_coordinates(self, coord_list):
        extru_now = coord_list[0][3]

        xlist = list()
        ylist = list()
        zlist = list()

        nfil = 0  # total number of filaments

        coord_fil = (
            list()
        )  # list of initial and final coordinates of each printed filament
        # coord_fil[0] = [coord_i, coord_f] -> of filament 1

        for index in range(1, len(coord_list)):

            coord_now = coord_list[index]

            # sum the relative extrusion coordinates.
            extru_now += coord_now[3]
            # this is done in order to account for retractiong movements.

            zlist.append(coord_now[2])

            if extru_now > 0:
                nfil += 1

                # initial coordinates of the filament
                coord_old = coord_list[index - 1]

                # Convert E to volume
                volume = extru_now * math.pi * \
                    (self._printer.feedstock_filament_diameter/2)**2

                coord_fil.append([coord_old, coord_now, volume])

                extru_now = 0.0  # reference is set to zero after printing a filament

                xlist.append(coord_old[0])
                xlist.append(coord_now[0])

                ylist.append(coord_old[1])
                ylist.append(coord_now[1])

                zlist.append(coord_old[2])
                zlist.append(coord_now[2])

        if not xlist:
            raise ValueError(
                "The provided gcode contains no extrusion moves to simulate"
            )

        xlim = [min(xlist), max(xlist)]
        ylim = [min(ylist), max(ylist)]
        zlim = [0.0, max(zlist)]

        return xlim, ylim, zlim, nfil, coord_fil
