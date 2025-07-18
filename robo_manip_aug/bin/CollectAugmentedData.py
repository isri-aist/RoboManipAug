import argparse
import importlib
import importlib.util
import sys

import yaml
from robo_manip_baselines.common import get_env_names

from robo_manip_aug import CollectAugmentedDataBase


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a meta argument parser for augmented data collection switching between different environments. The actual arguments are handled by another internal argument parser.",
        add_help=False,
    )
    parser.add_argument(
        "env",
        type=str,
        help="environment",
        default=None,
        choices=get_env_names(),
    )
    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and continue"
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.env is None:
        parser.print_help()
        return
    elif args.help:
        parser.print_help()
        print("\n================================\n")
        sys.argv += ["--help"]

    operation_module = importlib.import_module(
        f"robo_manip_baselines.envs.operation.Operation{args.env}"
    )
    OperationEnvClass = getattr(operation_module, f"Operation{args.env}")

    # The order of parent classes must not be changed in order to maintain the method resolution order (MRO)
    class CollectAugmentedData(OperationEnvClass, CollectAugmentedDataBase):
        pass

    if args.config is None:
        config = {}
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    collect = CollectAugmentedData(**config)
    collect.run()


if __name__ == "__main__":
    main()
