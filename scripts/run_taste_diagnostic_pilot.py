"""Train a short, explicitly step-limited turn-packed diagnostic from the base."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from duplex.training import (assert_optimizer_scope, build_datasets_and_collator,
    build_training_model, load_training_config, make_trainer, seed_everything)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/turn_packed_one_epoch_a100.yaml')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--learning-rate', required=True, type=float)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0:
        raise ValueError('Steps and learning rate must be positive.')
    if args.output.exists():
        raise FileExistsError('Use a fresh output folder for a controlled pilot.')
    config = load_training_config(args.config)
    config['training'].update(output_dir=str(args.output), learning_rate=args.learning_rate,
        eval_strategy='no', save_steps=50, save_total_limit=4)
    config.setdefault('diagnostic', {})['max_steps_override'] = args.steps
    args.output.mkdir(parents=True)
    (args.output / 'pilot.yaml').write_text(yaml.safe_dump({
        'diagnostic_max_steps_override': args.steps, 'config': config}, sort_keys=False))
    seed_everything(config['training']['seed'])
    model, processor, targets = build_training_model(config)
    train, dev, collator = build_datasets_and_collator(config, processor, model)
    trainer, audit = make_trainer(config, model, processor, targets, train, dev, collator, max_steps=args.steps)
    trainer.train()
    assert_optimizer_scope(trainer)
    if not audit.nonzero_gradient_names:
        raise RuntimeError('No nonzero LoRA gradient was observed.')
    trainer.save_model(args.output / 'final')
    print(f'Diagnostic pilot complete: {trainer.state.global_step} updates; not a full epoch.', flush=True)


if __name__ == '__main__':
    main()
