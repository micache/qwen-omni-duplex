"""Qwen Thinker compatibility probe reserved for a later session."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.parse_args()
    raise SystemExit("Model probing is not implemented in Session 01.")


if __name__ == "__main__":
    main()
