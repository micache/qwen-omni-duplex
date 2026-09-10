"""Prepared-sample inspection entry point reserved for a later session."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample")
    parser.parse_args()
    raise SystemExit("Sample inspection is not implemented in Session 01.")


if __name__ == "__main__":
    main()
