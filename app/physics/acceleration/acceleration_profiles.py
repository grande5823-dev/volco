import math
import numpy as np

from app.physics.acceleration.nozzle_speed import NozzleSpeed
from app.physics.acceleration.extruder_speed import ExtruderSpeed


class AccelerationManager:
    """
    Manages the calculation of speed profiles considering acceleration.
    """

    @staticmethod
    def calculate_speed_profiles(
        filament_length,
        volume,
        printing_speed,
        printer,
    ):
        """
        Calculate nozzle and extruder speed profiles considering acceleration.
        
        Args:
            filament_length: Length of the filament segment
            volume: Volume to be deposited
            printing_speed: Target printing speed
            printer: Printer configuration
            
        Returns:
            Tuple of (nozzle_speed, extruder_speed) profiles
        """
        extrusion_filter = getattr(printer, "extruder_motion_filter", None)

        # Calculate nozzle speed profile
        nozzle_speed = NozzleSpeed(
            filament_length=filament_length,
            target_speed=printing_speed,
            threshold_speed=printer.nozzle_jerk_speed,
            acceleration=printer.nozzle_acceleration,
            motion_filter=getattr(printer, "nozzle_motion_filter", None),
        )

        # Build a time vector shared by both profiles: the volume mapping
        # aligns nozzle and extruder displacements element-wise, so both must
        # be sampled on the same grid (extended for filter settling tails).
        additional_settling_time = (
            extrusion_filter.settling_time if extrusion_filter is not None else 0.0
        )
        discrete_time = nozzle_speed.build_time_grid(
            additional_settling_time=additional_settling_time
        )
        nozzle_speed.calculate_displacements(discrete_time)

        # Calculate extruder speed profile. When an extrusion filter is
        # configured, the extruder is decoupled: its ideal trajectory is the
        # ideal motion trajectory scaled to extrude a constant volume per unit
        # length, filtered to model the extrusion system dynamics.
        extruder_speed = ExtruderSpeed(
            volume=volume,
            threshold_speed=printer.extruder_jerk_speed,
            acceleration=printer.extruder_acceleration,
            total_time=nozzle_speed.total_time,
            printer=printer,
            nozzle_speed=nozzle_speed if extrusion_filter is not None else None,
            extrusion_filter=extrusion_filter,
        )
        extruder_speed.calculate_displacements(discrete_time)

        return nozzle_speed, extruder_speed