"""Prepare a bounded first-turn, question-audio-only InstructS2S subset."""

import argparse
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from transformers import AutoTokenizer

from duplex.dataset import resample_waveform_to_16khz
from duplex.instructs2s import SOURCE_ID, SOURCE_REVISION
from duplex.training import MODEL_ID, PINNED_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/InstructS2S-first-turn'))
    parser.add_argument('--train-count', type=int, default=44000)
    parser.add_argument('--dev-count', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.root
    if (root / 'provenance.json').exists():
        provenance = json.loads((root / 'provenance.json').read_text())
        expected = {'train': args.train_count, 'dev': args.dev_count}
        if provenance['source_revision'] != SOURCE_REVISION or any(
            provenance['statistics'][split]['conversations'] != count for split, count in expected.items()):
            raise ValueError('A different subset already exists; use a fresh root.')
        print('The requested subset is already prepared; preserving its manifests.', flush=True)
        return
    for name in ('audio', 'prepared-shards', 'downloads'):
        (root / name).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=PINNED_REVISION,
                                              local_files_only=True)
    order = list(range(1, 423))
    random.Random(212).shuffle(order)
    order.insert(0, 0)
    skipped = Counter()
    sources = []
    candidates = []
    started = time.monotonic()

    def prepare(part):
        cached = root / 'prepared-shards' / f'{part}.json'
        if cached.exists():
            return json.loads(cached.read_text())
        if shutil.disk_usage(root).free < 8 * 1024**3:
            raise RuntimeError('Stop before exhausting disk: less than 8 GiB free.')
        source = root / 'downloads' / f'{part}.parquet'
        inspected = Path('outputs/instructs2s-inspection/part-0.parquet')
        if part == 0 and inspected.exists() and not source.exists():
            shutil.copyfile(inspected, source)
        url = f'https://huggingface.co/datasets/{SOURCE_ID}/resolve/{SOURCE_REVISION}/data/audio_dataset_part_{part}.parquet'
        if not source.exists():
            for attempt in range(4):
                try:
                    with requests.get(url, stream=True, timeout=(30, 180)) as response:
                        response.raise_for_status()
                        with source.with_suffix('.partial').open('wb') as handle:
                            for block in response.iter_content(1024**2):
                                handle.write(block)
                    source.with_suffix('.partial').replace(source)
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for block in iter(lambda: handle.read(1024**2), b''):
                digest.update(block)
        source_bytes = source.stat().st_size
        rows = pq.read_table(source, columns=['id', 'round', 'question', 'answer', 'question_audio']).to_pylist()
        accepted = []
        rejected = Counter()
        for row in rows:
            if row['round'] != 1:
                rejected['later_turn'] += 1
                continue
            answer = row['answer'].strip()
            tokens = tokenizer.encode(answer, add_special_tokens=False)
            if not 16 <= len(tokens) <= 512:
                rejected['response_length'] += 1
                continue
            question = re.sub(r'^\s*<USER>:\s*', '', row['question']).strip()
            if not question or not answer or '<ASSISTANT>' in question:
                rejected['invalid_text'] += 1
                continue
            try:
                audio, rate = sf.read(io.BytesIO(row['question_audio']['bytes']),
                                      dtype='float32', always_2d=True)
            except (RuntimeError, TypeError):
                rejected['invalid_audio'] += 1
                continue
            audio = resample_waveform_to_16khz(audio.mean(axis=1), rate)
            if not 0 < len(audio) <= 30 * 16000 or not np.isfinite(audio).all():
                rejected['audio_length_or_invalid'] += 1
                continue
            sample_id = str(row['id'])
            file_id = hashlib.sha256(sample_id.encode()).hexdigest()
            path = Path('audio') / f'{file_id}.flac'
            sf.write(root / path, audio, 16000, subtype='PCM_16')
            user_chunks = math.ceil(len(audio) / 32000)
            assistant_chunks = math.ceil((len(tokens) + 2) / 50)
            accepted.append({'id': sample_id, 'prompt': question, 'answer': answer,
                'audio': str(path), 'user_samples': len(audio),
                'response_tokens': len(tokens), 'assistant_samples': assistant_chunks * 32000,
                'frames': (user_chunks + assistant_chunks) * 50, 'source_shard': part})
        result = {'source': {'part': part, 'sha256': digest.hexdigest(), 'bytes': source_bytes},
                  'rows': accepted, 'skipped': dict(rejected)}
        cached.write_text(json.dumps(result, ensure_ascii=False) + '\n')
        source.unlink()
        return result

    # Deduplicate prompts as well as IDs; held-out prompts cannot occur in training.
    seen_ids, seen_prompts = set(), set()
    required = args.train_count + args.dev_count
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for offset in range(0, len(order), args.workers):
            for result in executor.map(prepare, order[offset:offset + args.workers]):
                sources.append(result['source'])
                skipped.update(result['skipped'])
                for row in result['rows']:
                    key = ' '.join(row['prompt'].casefold().split())
                    if row['id'] in seen_ids or key in seen_prompts:
                        skipped['duplicate_id_or_prompt'] += 1
                        continue
                    seen_ids.add(row['id'])
                    seen_prompts.add(key)
                    candidates.append(row)
                print(json.dumps({'prepared_shards': len(sources), 'unique_candidates': len(candidates),
                    'free_gib': shutil.disk_usage(root).free / 1024**3,
                    'elapsed_s': time.monotonic() - started}), flush=True)
            if len(candidates) >= required:
                break
    if len(candidates) < required:
        raise RuntimeError(f'Only {len(candidates)} eligible first turns found.')
    random.Random(212).shuffle(candidates)
    dev = candidates[:args.dev_count]
    train = candidates[args.dev_count:required]
    keep = {row['audio'] for row in train + dev}
    for path in (root / 'audio').glob('*.flac'):
        if str(path.relative_to(root)) not in keep:
            path.unlink()
    lengths = {}
    for split, records in (('train', train), ('dev', dev)):
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
        lengths[split] = [{'id': row['id'], 'frames': row['frames']} for row in records]
    (root / 'lengths.json').write_text(json.dumps(lengths) + '\n')
    stats = {}
    for split, records in (('train', train), ('dev', dev)):
        tokens = [row['response_tokens'] for row in records]
        stats[split] = {'conversations': len(records), 'response_tokens_total': sum(tokens),
            'response_tokens_percentiles': dict(zip(('min','p25','median','p75','p90','p99','max'),
                np.percentile(tokens, [0,25,50,75,90,99,100]).tolist())),
            'frames_total': sum(row['frames'] for row in records),
            'question_audio_seconds': sum(row['user_samples'] for row in records) / 16000}
    provenance = {'original_dataset': 'ICTNLP/InstructS2S-200K', 'source_dataset': SOURCE_ID,
        'source_revision': SOURCE_REVISION, 'original_revision': '27c93b30be652905399d46abb232003dc78942d1',
        'license': 'CC-BY-NC-4.0; research use', 'selection': 'seed-212 shuffled shards, first turns only; 16-512 full response tokens; <=30s question audio; unique normalized prompts',
        'split_seed': 212, 'audio': '16kHz mono PCM16 FLAC question only; full 2s zero padding in train and inference',
        'response_canvas': 'synthetic silence rounded to 2s blocks; enough 25Hz frames for full answer + START/STOP; no assistant speech timing available',
        'context_token_ids': [], 'sources': sources, 'skipped': dict(skipped), 'statistics': stats,
        'source_bytes_streamed': sum(row['bytes'] for row in sources),
        'elapsed_seconds': time.monotonic() - started}
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'complete': stats, 'source_bytes_streamed': provenance['source_bytes_streamed']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
