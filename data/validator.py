import csv
from collections import Counter
from pathlib import Path

# this script reads allseasons.csv and prints how many rows exist for each season

root = Path(__file__).resolve().parent
input_path = root / "allseasons.csv"

season_counts = Counter()

with input_path.open(mode="r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        season = row.get("Season")
        if season is None:
            # skip rows without a season value
            continue
        season_counts[season] += 1

# print seasons in sorted order
for season in sorted(season_counts.keys()):
    print(f"{season}: {season_counts[season]} rows")
