from __future__ import absolute_import, division, print_function
import os, sys
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)
platform = sys.platform


class DolphinParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument parsers for everything")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to data, please note this path is an absolute path",
                                 default=os.path.join("dataset", "DOLPHIN"))

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # Visualisation ----------------------------------------------------------
        self.parser.add_argument("--save_visualise_image",
                                 type=self.str2bool,
                                 help="Save Visualisation?",
                                 default=False)

        self.parser.add_argument("--play_visualise",
                                    type=self.str2bool,
                                    help="Play Visualisation?",
                                    default=False)

        # Deep Learning Options ---------------------------------------
        self.parser.add_argument("--device",
                                type=str,
                                help="Your model device",
                                default='cpu')

        self.parser.add_argument("--model_file",
                                type=str,
                                help="Your path to model parameter",
                                default='models')

    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


class OurModelParser(DolphinParser):

    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Argument parsers for everything")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to data, please note this path is an absolute path",
                                 default=os.path.join("dataset", "DOLPHIN"))

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # Deep Learning Options
        self.parser.add_argument("--device",
                                 type=str,
                                 help="Your model device",
                                 default='cpu')

        self.parser.add_argument("--model_file",
                                 type=str,
                                 help="Your path to model parameter",
                                 default='models')

        self.parser.add_argument("--mode",
                                 type=str,
                                 help="Train/Test")


class GeneralParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument parsers for everything")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to data, please note this path is an absolute path",
                                 default=os.path.join("dataset", "video-vrd"))

        # PATHS
        self.parser.add_argument("--model_path",
                                 type=str,
                                 help="Path to your detection model .pth file(Leave blank if none)",
                                 default=os.path.join("model","helper","param"))

        self.parser.add_argument("--device",
                                 type=str,
                                 help="Select your device to run your model/ CUDA/CPU",
                                 default='cpu')

        self.parser.add_argument("--debug",
                                 type=self.str2bool,
                                 help="Select Yes for debugging",
                                 default=False)
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def dir_path(self, path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

    def parse(self):
        self.options = self.parser.parse_args()
        _ = self.dir_path(self.options.data_path)
        return self.options



if __name__ == '__main__':
	testLoader = GeneralParser()
	print(testLoader.parse())