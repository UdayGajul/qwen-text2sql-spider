"""Normalize an official Spider archive into the project's data/spider layout."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Downloaded Spider .zip archive")
    parser.add_argument("--output", type=Path, default=Path("data/spider"))
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(extracted)

        tables = find_one(extracted, "tables.json")
        dev = find_one(extracted, "dev.json")
        gold = find_one(extracted, "dev_gold.sql")
        database = next((path for path in extracted.rglob("database") if path.is_dir()), None)
        if database is None:
            raise FileNotFoundError(f"Could not find database/ under {extracted}")

        shutil.copy2(tables, args.output / "tables.json")
        shutil.copy2(dev, args.output / "dev.json")
        shutil.copy2(gold, args.output / "dev_gold.sql")
        destination = args.output / "database"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(database, destination)

    print(f"Spider data prepared at {args.output.resolve()}")


if __name__ == "__main__":
    main()
