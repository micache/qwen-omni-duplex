"""Train the Session 08 Qwen2.5-Omni Thinker LoRA/QLoRA adapter."""

import argparse
from pathlib import Path

from duplex.training import load_training_config, run_smoke_from_config, train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Require the config's explicit synthetic smoke-data path.",
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    synthetic = bool(config["data"].get("synthetic_smoke", False))
    if args.smoke_test != synthetic:
        raise SystemExit(
            "--smoke-test and data.synthetic_smoke: true must be selected together."
        )
    trainer = run_smoke_from_config(config) if synthetic else train_from_config(config)
    print(
        f"Finished {config['training']['run_label']} at optimizer step "
        f"{trainer.state.global_step}; adapter saved under "
        f"{config['training']['output_dir']}."
    )


if __name__ == "__main__":
    main()
