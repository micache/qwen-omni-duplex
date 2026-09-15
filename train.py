"""Train the Qwen2.5-Omni Thinker LoRA or QLoRA duplex adapter."""

import argparse
from pathlib import Path

from duplex.training import load_training_config, train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_training_config(args.config)
    trainer = train_from_config(config)
    print(
        f"Finished {config['training']['run_label']} at optimizer step "
        f"{trainer.state.global_step}; adapter saved under "
        f"{config['training']['output_dir']}."
    )


if __name__ == "__main__":
    main()
