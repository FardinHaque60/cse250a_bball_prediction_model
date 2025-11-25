import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


def infer_season(game_date):
    """
    infers nba season from a game date
    example: oct 2000 - jun 2001 -> 2000
    """
    month = game_date.month
    year = game_date.year
    if month >= 10:
        return year
    return year - 1


def load_partial_rows(partials_dir):
    """
    loads rows from all csv files in the partials directory, inferring season and
    removing duplicates based on (date, team, w_l)
    """
    expected_columns = [
        "date",
        "team",
        "W/L",
        "eFG%",
        "TOV%",
        "ORB%",
        "FT/FGA",
        "Pace",
        "ORtg",
        "IsRegular",
    ]

    seen_keys: Dict[Tuple[str, str, str], dict] = {}

    for csv_path in sorted(partials_dir.glob("*.csv")):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)

            # ensure required columns are present
            missing = [c for c in expected_columns if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"file {csv_path} is missing required columns: {missing}"
                )

            for row in reader:
                date_str = row["date"]
                team = row["team"]
                w_l = row["W/L"]

                key = (date_str, team, w_l)
                if key in seen_keys:
                    # skip exact logical duplicates based on date, team, result
                    continue

                game_date = datetime.strptime(date_str, "%Y-%m-%d")
                season = infer_season(game_date)

                full_row = {col: row[col] for col in expected_columns}
                full_row["Season"] = str(season)
                seen_keys[key] = full_row

    return list(seen_keys.values())


def write_combined_csv(rows, output_path):
    """
    writes combined rows to the output csv sorted by date with the correct column order
    """
    header = [
        "date",
        "team",
        "W/L",
        "eFG%",
        "TOV%",
        "ORB%",
        "FT/FGA",
        "Pace",
        "ORtg",
        "IsRegular",
        "Season",
    ]

    # sort by parsed date
    def parse_date(row):
        return datetime.strptime(row["date"], "%Y-%m-%d")

    rows_sorted = sorted(rows, key=parse_date)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)


if __name__ == "__main__":
    """
    combines csv files from data/partials into data/allseasons.csv
    """
    data_dir = Path(__file__).resolve().parent
    partials_dir = data_dir / "partials"
    output_path = data_dir / "allseasons.csv"

    if not partials_dir.exists():
        raise FileNotFoundError(f"partials directory not found at {partials_dir}. ")

    rows = load_partial_rows(partials_dir)
    write_combined_csv(rows, output_path)
