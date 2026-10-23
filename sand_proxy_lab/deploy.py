"""Publish a final shadow-refitted candidate, preserving the previous active parameters."""

import argparse
from pathlib import Path

from .paths import add_terrain_argument
from .shadow.refit import deploy_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_terrain_argument(parser)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    deploy_model(args.model.resolve(), terrain=args.terrain)


if __name__ == "__main__":
    main()
