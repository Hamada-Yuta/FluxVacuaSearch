"""Join the distributed Parquet parts without reordering any sample rows."""
from pathlib import Path
import argparse
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, default=Path("data/selected_solutions"))
    parser.add_argument("--output", type=Path, default=Path("data/selected_solutions.parquet"))
    args = parser.parse_args()
    paths = [args.parts / f"part-{i:05d}.parquet" for i in range(2)]
    table = pa.concat_tables([pq.read_table(p) for p in paths])
    if table.num_rows != 1_296_633:
        raise ValueError(f"Unexpected sample size: {table.num_rows}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    if not table.equals(pq.read_table(args.output), check_metadata=True):
        raise RuntimeError("Parquet round-trip changed the sample")
    print(f"Wrote {table.num_rows:,} ordered rows to {args.output}")


if __name__ == "__main__":
    main()
