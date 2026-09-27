"""Extract one TASTE row into a local one-row Parquet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/TASTE-IF-SFT-48K/data/shuffled_train_part_0008.parquet"),
    )
    parser.add_argument("--sample-id", default="read_aloud_038934")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        f"data/TASTE-one-row-{args.sample_id}/data/shuffled_train_part_0000.parquet"
    )
    if output.exists():
        raise FileExistsError(output)
    matches = []
    for batch in pq.ParquetFile(args.source).iter_batches(batch_size=64):
        ids = batch.column(batch.schema.get_field_index("idx")).to_pylist()
        matches.extend(batch.slice(index, 1) for index, value in enumerate(ids) if value == args.sample_id)
    if len(matches) != 1:
        raise ValueError(f"Expected one {args.sample_id!r} row; found {len(matches)}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_batches(matches), output)
    print(output)


if __name__ == "__main__":
    main()
