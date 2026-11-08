
"""Entry point for the AI Emergency Response System."""

import argparse

from src.application import run_application


def main():
    parser = argparse.ArgumentParser(
        description="AI Emergency Response System"
    )

    parser.add_argument(
        "scenario",
        nargs="?",
        default="Scenarios/Scenario1",
        help="Scenario directory to process"
    )

    args = parser.parse_args()

    run_application(args.scenario)


if __name__ == "__main__":
    main()
