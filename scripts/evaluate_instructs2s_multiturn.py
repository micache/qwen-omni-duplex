"""Inspect post-STOP frames and a continuous, audio-only multi-turn conversation."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from duplex.instructs2s import InstructS2SFirstTurnDataset
from duplex.streaming import QwenDuplexStreamer, pad_audio_to_chunk_boundary, write_trace_jsonl
from duplex.training import _load_adapter_weights, build_training_model, load_training_config, seed_everything
from duplex.turn_packed import decode_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/instructs2s_response_gain003_pilot.yaml')
    parser.add_argument('--adapter', type=Path, default=Path('outputs/instructs2s-response-gain003-pilot/final'))
    parser.add_argument('--source', type=Path, default=Path('outputs/instructs2s-inspection/part-0.parquet'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gap-seconds', type=int, default=4)
    parser.add_argument('--isolated-followups', action='store_true',
                        help='Run rounds 2 and 3 separately with a fresh cache as a history control.')
    args = parser.parse_args()
    if args.gap_seconds < 2 or args.gap_seconds % 2:
        raise ValueError('Use a positive gap consisting of complete two-second chunks.')
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_training_config(args.config)
    seed_everything(config['training']['seed'])
    used_ids = {row['id'] for split in ('train', 'dev') for row in
                InstructS2SFirstTurnDataset(config['data']['root'], split=split).records}
    rows = pq.read_table(args.source, columns=['id', 'round', 'question', 'answer', 'question_audio']).to_pylist()
    groups = defaultdict(list)
    for row in rows:
        groups[row['id']].append(row)
    # Fixed source order, chosen before generation. All turns of the selected
    # conversation are absent from the first-turn training/dev manifests.
    conversation = next(sorted(group, key=lambda row: row['round']) for sample_id, group in groups.items()
                        if sample_id not in used_ids and {1, 2, 3}.issubset({row['round'] for row in group}))[:3]
    rectangle = next(row for row in rows if row['id'] == 'instruct_en_92' and row['round'] == 1)
    model, processor, _ = build_training_model(config)
    _load_adapter_weights(model, args.adapter)
    model.eval()
    # All silence is explicitly part of the provided recording. A STOP must not
    # end processing that recording, and no synthetic EOF tail is needed here.
    streamer = QwenDuplexStreamer(model, processor, sample=False, max_silent_chunks=0)
    report = {'adapter': str(args.adapter), 'context_token_ids': [], 'gap_seconds': args.gap_seconds,
              'cache_reset_between_turns': args.isolated_followups,
              'reference_answers_fed_to_model': False, 'cases': []}

    def evaluate(name, waveform, turns):
        sf.write(args.output / f'{name}.flac', waveform, 16000, subtype='PCM_16')
        waveform, _ = sf.read(args.output / f'{name}.flac', dtype='float32')
        result = streamer.run(waveform, sample_id=name, context_token_ids=(), sample_rate_hz=16000)
        expected_frames = len(waveform) // 640
        if len(result.trace) != expected_frames:
            raise RuntimeError(f'Fixed-duration stream ended early: {len(result.trace)} / {expected_frames} frames.')
        write_trace_jsonl(args.output / f'{name}.jsonl', result.trace)
        raw = processor.tokenizer.decode([r.event_id for r in result.trace],
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        (args.output / f'{name}-raw.txt').write_text(raw + '\n')
        segments, active = [], None
        for index, row in enumerate(result.trace):
            if row.event_type == 'START':
                active = {'start_frame': index, 'text_ids': []}
            elif row.event_type == 'TEXT' and active is not None:
                active['text_ids'].append(row.event_id)
            elif row.event_type == 'STOP' and active is not None:
                active['stop_frame'] = index
                active['text'] = processor.tokenizer.decode(active.pop('text_ids'),
                    skip_special_tokens=False, clean_up_tokenization_spaces=False)
                segments.append(active)
                active = None
        if active is not None:
            active['text'] = processor.tokenizer.decode(active.pop('text_ids'),
                skip_special_tokens=False, clean_up_tokenization_spaces=False)
            active['stop_frame'] = None
            segments.append(active)
        case = {'name': name, 'input_duration_s': len(waveform)/16000, 'turns': turns,
                'frames': len(result.trace), 'event_counts': dict(Counter(r.event_type for r in result.trace)),
                'segments': segments, 'text': result.text, 'timed_out': result.timed_out,
                'last_event': result.trace[-1].event_type,
                'grammar_mask_changed_count': sum(r.grammar_mask_changed_raw_argmax for r in result.trace),
                'raw_output': str(args.output / f'{name}-raw.txt')}
        report['cases'].append(case)
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(case, ensure_ascii=False), flush=True)

    report['selection'] = 'first source-order conversation absent from both manifests with rounds 1, 2, 3'
    report['conversation_id'] = conversation[0]['id']
    if args.isolated_followups:
        for row in conversation[1:]:
            raw = decode_audio(row['question_audio'], field='question_audio')
            waveform = np.concatenate([pad_audio_to_chunk_boundary(raw),
                                       np.zeros(args.gap_seconds*16000, dtype=np.float32)])
            evaluate(f'isolated-round-{row["round"]}', waveform,
                [{'round': row['round'], 'question': row['question'].split('<USER>:')[-1].strip(),
                  'speech_start_s': 0, 'speech_end_s': len(raw)/16000, 'reference': row['answer']}])
        return

    raw = decode_audio(rectangle['question_audio'], field='question_audio')
    rectangle_audio = np.concatenate([pad_audio_to_chunk_boundary(raw), np.zeros(32000, dtype=np.float32)])
    evaluate('rectangle-complete', rectangle_audio, [{'question': rectangle['question'].split('<USER>:')[-1].strip(),
        'speech_start_s': 0, 'speech_end_s': len(raw)/16000, 'reference': rectangle['answer']}])

    pieces, turns = [], []
    samples = 0
    for row in conversation:
        raw = decode_audio(row['question_audio'], field='question_audio')
        padded = pad_audio_to_chunk_boundary(raw)
        turns.append({'round': row['round'], 'question': row['question'].split('<USER>:')[-1].strip(),
                      'speech_start_s': samples/16000, 'speech_end_s': (samples+len(raw))/16000,
                      'reference': row['answer']})
        pieces.extend([padded, np.zeros(args.gap_seconds*16000, dtype=np.float32)])
        samples += len(padded) + args.gap_seconds*16000
    evaluate('multiturn-continuous', np.concatenate(pieces), turns)


if __name__ == '__main__':
    main()
