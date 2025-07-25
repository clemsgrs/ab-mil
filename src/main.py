import argparse
import subprocess
import sys

from src.utils.config import get_cfg_from_args


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("ab-mil", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command. For Yacs configs, use space-separated \"PATH.KEY VALUE\" pairs. For python-based LazyConfig, use \"path.key=value\".",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory to save logs and checkpoints",
    )
    return parser


def classification(config_file):
    print("Running train/classification.py...")
    cmd = [
        sys.executable,
        "src/train/classification.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Classification failed. Exiting.")
        sys.exit(result.returncode)


def regression(config_file):
    print("Running train/regression.py...")
    cmd = [
        sys.executable,
        "src/train/regression.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Regression failed. Exiting.")
        sys.exit(result.returncode)


def main(args):

    config_file = args.config_file
    cfg = get_cfg_from_args(args)

    if cfg.task == "classification":
        classification(config_file)
    elif cfg.task == "regression":
        regression(config_file)
    else:
        print(f"Unsupported task: {cfg.task}. Exiting.")
        sys.exit(1)


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)
