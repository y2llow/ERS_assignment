"""Entry point for the AI Emergency Response System.

Usage:  python run.py [Scenarios/Scenario1] [--output output]
Creates plan.json and statistics.txt in the output directory.
"""

import argparse

from src.application import run_application


def main():
    parser = argparse.ArgumentParser(description="AI Emergency Response System")
    parser.add_argument("scenario", nargs="?", default="Scenarios/Scenario1",
                        help="Scenario directory to process")
    parser.add_argument("--output", default="output",
                        help="Directory for plan.json and statistics.txt")
    args = parser.parse_args()

    run_application(args.scenario, args.output)


if __name__ == "__main__":
    main()