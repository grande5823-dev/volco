import argparse


class Arguments:
    @staticmethod
    def get_options():
        parser = argparse.ArgumentParser()
        parser.add_argument("--gcode", type=str)
        parser.add_argument("--sim", type=str)
        parser.add_argument("--printer", type=str)
        parser.add_argument(
            "--heightmap",
            type=float,
            default=None,
            metavar="Z_MIN",
            help="export only a 2D top-surface heightmap (skips mesh/STL "
            "generation), considering model voxels above the candidate top "
            "surface z >= Z_MIN [mm]",
        )

        return parser.parse_args()
