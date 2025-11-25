import csv
from pathlib import Path
import tempfile
import os

root = Path(__file__).resolve().parent
input_path = root / "allseasons.csv"

# create a temporary file in the same directory
with tempfile.NamedTemporaryFile(
    "w", newline="", encoding="utf-8", dir=root, delete=False
) as tmpfile:
    reader = csv.DictReader(input_path.open("r", newline="", encoding="utf-8"))
    writer = csv.DictWriter(tmpfile, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        if row.get("IsRegular") != "0":
            writer.writerow(row)

# replace original file atomically
os.replace(tmpfile.name, input_path)

print(f"{input_path.name} has been filtered and overwritten.")
