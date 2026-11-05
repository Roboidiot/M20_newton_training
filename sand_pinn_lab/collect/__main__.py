"""Run with: .venv/bin/python -m sand_pinn_lab.collect --help."""

import json

from .config import parse_args


def main():
    args = parse_args()
    if args.print_config:
        print(json.dumps(args.physics.to_dict(), indent=2))
        return
    from .collector import run

    run(args)


if __name__ == "__main__":
    main()
