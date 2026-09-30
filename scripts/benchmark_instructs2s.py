"""Measure full and long-sequence training batches before choosing a frame budget."""

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from duplex.batching import FrameBudgetBatchSampler
from duplex.instructs2s import InstructS2SFirstTurnDataset
from duplex.training import MODEL_INPUTS, build_training_model, load_training_config, seed_everything
from duplex.turn_packed import NoiseAugmentationConfig, TurnPackedCollator


def main():
    config = load_training_config('configs/instructs2s_one_epoch_a100.yaml')
    seed_everything(211)
    model, processor, _ = build_training_model(config)
    model.train()
    model.base_thinker.audio_tower.eval()
    train = InstructS2SFirstTurnDataset(config['data']['root'], split='train')
    collator = TurnPackedCollator(audio_processor=processor, audio_tower=model.base_thinker.audio_tower,
        tokenizer=processor.tokenizer, control_tokens=model.control_tokens,
        thinker_bos_token_id=model.base_thinker.config.bos_token_id,
        interruption_probability=0, noise_dataset=None, noise_config=NoiseAugmentationConfig(),
        min_assistant_frames=4, augmentation_seed=214)
    # Allocate Adam's states so the benchmark includes the optimizer memory.
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0, fused=True)
    results = []
    for budget in (3360, 3680, 4000, 4320):
        sampler = FrameBudgetBatchSampler(train.frame_lengths, max_frames=budget, max_batch_size=128, seed=212)
        batches = list(sampler)
        worst_long = max(batches, key=lambda b: max(train.frame_lengths[i] for i in b))
        failed = False
        for indices in (batches[0], worst_long):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch = prediction = None
            try:
                batch = {k: v.to('cuda') for k, v in collator([train[i] for i in indices]).items()
                         if k in MODEL_INPUTS and isinstance(v, torch.Tensor)}
                started = time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(**batch)
                prediction.loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                results.append({'budget': budget, 'batch_size': len(indices),
                    'longest_frames': int(batch['labels'].shape[1]), 'padded_frames': int(batch['labels'].numel()),
                    'seconds': time.monotonic()-started,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3, 'status': 'passed'})
            except torch.cuda.OutOfMemoryError:
                results.append({'budget': budget, 'batch_size': len(indices), 'status': 'out_of_memory'})
                failed = True
            finally:
                optimizer.zero_grad(set_to_none=True)
                del batch, prediction
                gc.collect()
                torch.cuda.empty_cache()
            print(json.dumps(results[-1]), flush=True)
            Path('outputs/instructs2s-one-epoch/batch-benchmark.json').write_text(json.dumps(results, indent=2)+'\n')
            if failed:
                break
        if failed:
            break


if __name__ == '__main__':
    main()
