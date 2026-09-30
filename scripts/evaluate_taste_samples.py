"""Reload an adapter and compare fixed train/dev samples with their references."""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch
import numpy as np

from duplex.metrics import event_classification_metrics
from duplex.streaming import QwenDuplexStreamer, write_trace_jsonl
from duplex.training import (MODEL_INPUTS, _load_adapter_weights, build_training_model,
                             load_training_config)
from duplex.turn_packed import (NoiseAugmentationConfig, TurnPackedCollator,
                                augment_packed_interruption, load_local_taste)


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--samples-per-split', type=int, default=3)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        raise ValueError('Select at least one sample per split.')
    config = load_training_config(args.config)
    output = Path(config['training']['output_dir'])
    adapter = args.adapter or output / 'final'
    started = time.perf_counter()
    model, processor, _ = build_training_model(config)
    _load_adapter_weights(model, adapter)
    model.eval()
    model.gradient_checkpointing_disable()
    collator = TurnPackedCollator(
        audio_processor=processor, audio_tower=model.base_thinker.audio_tower,
        tokenizer=processor.tokenizer, control_tokens=model.control_tokens,
        thinker_bos_token_id=model.base_thinker.config.bos_token_id,
        interruption_probability=0, min_assistant_frames=4, noise_dataset=None,
        noise_config=NoiseAugmentationConfig(), augmentation_seed=214,
    )
    streamer = QwenDuplexStreamer(model, processor, max_silent_chunks=4,
                                  max_new_tokens=128, sample=False)
    cases = []
    for split in ('train', 'dev'):
        dataset = load_local_taste(config['data']['root'], split=split)
        selected = [dataset[i] for i in range(min(args.samples_per_split, len(dataset)))]
        for index, conversation in enumerate(selected):
            timeline = collator._timeline(conversation)
            batch = {k: v.to('cuda') for k, v in collator([timeline]).items()
                     if k in MODEL_INPUTS and isinstance(v, torch.Tensor)}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(**batch)
            gold = batch['labels'][0].tolist()
            ids = prediction.logits.argmax(-1)[0].tolist()
            lexical = [i for i, event in enumerate(timeline.targets.events) if event.kind.value == 'TEXT']
            case = {
                'split': split, 'index': index, 'sample_id': conversation.conversation_id,
                'prompt': dataset.dataset[index]['message'][0]['text'],
                'reference_text': conversation.assistant_text,
                'user_duration_s': len(conversation.user_waveform) / 16000,
                'training_duration_s': len(timeline.user_waveform) / 16000,
                'teacher_forced': {
                    'weighted_loss': float(prediction.loss),
                    'group_losses': {k: float(v) for k, v in prediction.group_losses.items()},
                    'lexical_token_accuracy': sum(ids[i] == gold[i] for i in lexical) / max(1, len(lexical)),
                    'text_at_reference_frames': processor.tokenizer.decode([ids[i] for i in lexical],
                        clean_up_tokenization_spaces=False),
                    'events': event_classification_metrics(gold, ids, control_tokens=model.control_tokens),
                },
                'streams': {},
            }
            del prediction, batch
            sf.write(output / f'{split}-{conversation.conversation_id}-user.wav',
                     conversation.user_waveform, 16000)
            modes = [('aligned_audio', timeline.user_waveform),
                     ('user_only', conversation.user_waveform),
                     ('user_chunk_padded', np.pad(conversation.user_waveform,
                         (0, (-len(conversation.user_waveform)) % 32000)))]
            interrupted = augment_packed_interruption(
                timeline, selected[(index + 1) % len(selected)].user_waveform,
                probability=1, min_assistant_frames=4, rng=random.Random(214 + index),
                control_tokens=model.control_tokens,
                thinker_bos_token_id=model.base_thinker.config.bos_token_id,
            )
            if interrupted.interruption is not None:
                modes.append(('interrupted', interrupted.user_waveform))
                case['interruption_onset_frame'] = interrupted.interruption.cut_frame
            for mode, waveform in modes:
                name = f'{split}-{conversation.conversation_id}-{mode}'
                stream = streamer.run(waveform, sample_id=name, context_token_ids=(), sample_rate_hz=16000)
                trace = output / 'traces' / f'{name}.jsonl'
                write_trace_jsonl(trace, stream.trace)
                expected_text = conversation.assistant_text
                if mode == 'interrupted':
                    expected_text = processor.tokenizer.decode(
                        [event.text_id for event in interrupted.targets.events
                         if event.kind.value == 'TEXT'], clean_up_tokenization_spaces=False)
                words = expected_text.split()
                case['streams'][mode] = {
                    'text': stream.text,
                    'expected_text': expected_text,
                    'exact_reference': stream.text == expected_text,
                    'word_error_rate': edit_distance(words, stream.text.split()) / max(1, len(words)),
                    'event_counts': dict(Counter(row.event_type for row in stream.trace)),
                    'timed_out': stream.timed_out, 'stop_reason': stream.stop_reason,
                    'trace': str(trace),
                }
                if mode == 'aligned_audio':
                    case['streams'][mode]['exact_target_events'] = [r.event_id for r in stream.trace[:len(gold)]] == gold
                if mode == 'interrupted':
                    onset = case['interruption_onset_frame']
                    active = False
                    lexical_before = 0
                    for row in stream.trace[:onset]:
                        if row.event_type == 'START':
                            active, lexical_before = True, 0
                        elif row.event_type == 'STOP':
                            active = False
                        elif row.event_type == 'TEXT' and active:
                            lexical_before += 1
                    stop = next((i for i, row in enumerate(stream.trace) if i >= onset and row.event_type == 'STOP'), None)
                    case['streams'][mode].update(
                        responding_at_onset=active and lexical_before > 0,
                        stop_after_onset_frame=stop,
                        stop_delay_s=None if stop is None else (stop - onset) / 25,
                    )
                print(json.dumps({'sample': name, **case['streams'][mode]}, ensure_ascii=False), flush=True)
            cases.append(case)
            (output / 'inference-report.json').write_text(json.dumps({
                'adapter': str(adapter), 'selection': 'first rows of each split; no cherry picking',
                'context_token_ids': [], 'noise_probability': 0,
                'cases': cases, 'runtime_seconds': time.perf_counter() - started,
            }, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
