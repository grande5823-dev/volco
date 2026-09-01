import json

from app.physics.acceleration.transfer_function import TransferFunction


class Printer:
    def __init__(self, config_path=None, config_dict=None):
        if config_path:
            self._read_config_file(config_path)
        elif config_dict:
            self._load_config_from_dict(config_dict)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

    def _read_config_file(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
            self._load_config_from_dict(config)

    def _load_config_from_dict(self, config):
        self.nozzle_jerk_speed = config["nozzle_jerk_speed"]
        self.extruder_jerk_speed = config["extruder_jerk_speed"]
        self.nozzle_acceleration = config["nozzle_acceleration"]
        self.extruder_acceleration = config["extruder_acceleration"]
        self.feedstock_filament_diameter = config["feedstock_filament_diameter"]
        self.nozzle_diameter = config["nozzle_diameter"]

        # Optional filters modelling motion-system dynamics, e.g.
        # {"type": "first_order", "time_constant": 0.05}.
        # Only take effect when consider_acceleration is enabled.
        # nozzle_motion_filter: filters the nozzle motion profile.
        # extruder_motion_filter: decouples the extruder from the nozzle; the
        # ideal extrusion trajectory (scaled ideal nozzle velocity) is filtered
        # with this transfer function to obtain the actual extrusion trajectory.
        self.nozzle_motion_filter = None
        self.extruder_motion_filter = None
        for key in ("nozzle_motion_filter", "extruder_motion_filter"):
            filter_config = config.get(key)
            if filter_config is not None:
                setattr(self, key, TransferFunction.from_config(filter_config))
