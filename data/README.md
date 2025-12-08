# data and scripts for collecting and combining nba game data

- **raw scraping and combination**
  - `NBA_scraper.py`: script to scrape or download nba game data (regular season) from the source site
  - `csv_combiner.py`: helper to merge csv files from partials folder into a single combined csv used by the models
  - `validator.py`: validate the combined csv with number of rows for each season
  - `allseasons.csv`: the combined csv file of all regular season games

- **csv schema**
  - `date`: game date in `%m/%d/%Y` format (for example `10/31/2000`)
  - `team`: team abbreviation (for example `LAL`)
  - `W/L`: `1` for win, `0` for loss (from the team perspective)
  - `eFG%`: effective field goal percentage
  - `TOV%`: turnover percentage
  - `ORB%`: offensive rebounding percentage
  - `FT/FGA`: free throws per field goal attempt
  - `IsRegular`: `1` for regular season games, `0` otherwise
  - `Season`: season start year as an integer (for example `2000` for the 2000–01 season)
