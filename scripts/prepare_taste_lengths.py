"""Index local TASTE audio lengths for padded-frame training batches."""

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/TASTE-IF-SFT-48K'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for split, pattern in [('train', 'shuffled_train_part_*.parquet'),
                           ('dev', 'shuffled_dev.parquet')]:
        rows = []
        paths = sorted((args.root / 'data').glob(pattern))
        if not paths:
            raise FileNotFoundError(f'No local TASTE {split} shards.')
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=64, columns=['idx', 'instruction_audio', 'response_audio']
            ):
                for row in batch.to_pylist():
                    counts = []
                    for key in ['instruction_audio', 'response_audio']:
                        info = sf.info(io.BytesIO(row[key]['bytes']))
                        counts.append((info.frames * 16000 + info.samplerate - 1) // info.samplerate)
                    full, partial = divmod(sum(counts), 32000)
                    preconv = max(3, partial // 160)
                    frames = full * 50
                    if partial:
                        frames += ((preconv - 1) // 2 + 1 - 2) // 2 + 1
                    rows.append({'id': row['idx'], 'frames': frames,
                                 'user_samples': counts[0], 'assistant_samples': counts[1]})
            print(split, path.name, len(rows), flush=True)
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError(f'Duplicate TASTE IDs in {split}.')
        result[split] = rows
        print(split, 'longest', sorted(rows, key=lambda r: r['frames'], reverse=True)[:5], flush=True)
    if {row['id'] for row in result['train']} & {row['id'] for row in result['dev']}:
        raise ValueError('TASTE train/dev IDs overlap.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))


if __name__ == '__main__':
    main()
