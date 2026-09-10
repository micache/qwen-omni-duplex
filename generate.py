"""Text-timeline generation entry point reserved for a later session."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise SystemExit("Generation is not implemented in Session 01.")


if __name__ == "__main__":
    main()
