import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urljoin
from playwright.async_api import async_playwright
import time
import os
import signal
import sys

LAST_REQUEST_TIME = 0
MIN_SECONDS_BETWEEN_REQUESTS = 3  # <= stays below 20 req/min

BASE = "https://www.basketball-reference.com"
START_URL = "https://www.basketball-reference.com/boxscores/?month=10&day=27&year=2015"
EARLY_EXIT = (2019, 7)

OUTPUT_FILE = "nba_games_four_factors_all_years.csv"
rows = []


def save_data():
    """save collected data to csv file"""
    if not rows:
        print("[INFO] No data to save.")
        return

    df = pd.DataFrame(rows)
    if not os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
    print(f"[SAVED] {len(rows)} rows written to {OUTPUT_FILE}")


def signal_handler(sig, frame):
    """handle ctrl+c and other interrupts - autosave before exit"""
    print("\n\n[INTERRUPT] Received exit signal, saving data...")
    save_data()
    sys.exit(0)


def parse_date_from_url(url: str) -> tuple:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    year = int(qs.get("year", ["0"])[0])
    month = int(qs.get("month", ["0"])[0])
    day = int(qs.get("day", ["0"])[0])
    date_str = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    return year, month, day, date_str


def parse_line_score(soup: BeautifulSoup):
    table = soup.find("table", id="line_score")
    if table is None:
        raise ValueError("line_score table not found")

    scores = {}
    tbody = table.find("tbody")
    if not tbody:
        raise ValueError("line_score tbody missing")

    for tr in tbody.find_all("tr"):
        th = tr.find("th")
        if th is None:
            continue

        team_abbr = th.get_text(strip=True)
        t_cell = tr.find("td", attrs={"data-stat": "T"})
        if t_cell is None:
            tds = tr.find_all("td")
            if not tds:
                continue
            t_cell = tds[-1]

        scores[team_abbr] = int(t_cell.get_text(strip=True).replace("\xa0", ""))

    if len(scores) != 2:
        raise ValueError("Expected 2 teams in line_score")

    return scores


def parse_four_factors(soup: BeautifulSoup):
    table = soup.find("table", id="four_factors")
    if table is None:
        raise ValueError("four_factors table not found")

    tbody = table.find("tbody")
    if not tbody:
        raise ValueError("four_factors tbody missing")

    ff = {}
    for tr in tbody.find_all("tr"):
        th = tr.find("th")
        if not th:
            continue
        team_abbr = th.get_text(strip=True)

        def get(stat):
            td = tr.find("td", attrs={"data-stat": stat})
            return td.get_text(strip=True) if td else None

        ff[team_abbr] = {
            "pace": get("pace"),
            "eFG%": get("efg_pct"),
            "TOV%": get("tov_pct"),
            "ORB%": get("orb_pct"),
            "FT/FGA": get("ft_rate"),
            "ORtg": get("off_rtg"),
        }

    return ff


async def fetch_html(page, url: str, is_boxscore=False) -> str:
    global LAST_REQUEST_TIME

    # rate limit
    elapsed = time.time() - LAST_REQUEST_TIME
    if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
        print(f"[RATE LIMIT] Sleeping {wait:.2f}s")
        await asyncio.sleep(wait)

    LAST_REQUEST_TIME = time.time()
    print(f"[LOAD] {url}")

    await page.goto(url, wait_until="domcontentloaded", timeout=45000)

    if is_boxscore:
        try:
            await page.wait_for_selector("#line_score", timeout=8000)
            await page.wait_for_selector("#four_factors", timeout=8000)
        except:
            print("    [WARN] slow tables → retry")
            await page.wait_for_timeout(3000)

    return await page.content()


async def scrape_game(page, game_url: str, date_str: str):
    print(f"    [GAME] {game_url}")
    html = await fetch_html(page, game_url, is_boxscore=True)
    soup = BeautifulSoup(html, "html.parser")

    # detect regular vs playoffs
    h1 = soup.select_one("h1")
    if h1:
        t = h1.get_text(strip=True)
        is_regular = 0 if t[:4].isdigit() else 1
    else:
        is_regular = 1

    filename = game_url.split("/")[-1]
    home_abbr = filename.split(".")[0][-3:]

    scores = parse_line_score(soup)
    ff = parse_four_factors(soup)

    teams = list(scores.keys())
    away_abbr = teams[0] if teams[1] == home_abbr else teams[1]

    home_win = 1 if scores[home_abbr] > scores[away_abbr] else 0

    # home
    rows.append(
        {
            "date": date_str,
            "team": home_abbr,
            "W/L": home_win,
            "eFG%": ff[home_abbr]["eFG%"],
            "TOV%": ff[home_abbr]["TOV%"],
            "ORB%": ff[home_abbr]["ORB%"],
            "FT/FGA": ff[home_abbr]["FT/FGA"],
            "Pace": ff[home_abbr]["pace"],
            "ORtg": ff[home_abbr]["ORtg"],
            "IsRegular": is_regular,
        }
    )

    # away
    rows.append(
        {
            "date": date_str,
            "team": away_abbr,
            "W/L": 1 - home_win,
            "eFG%": ff[away_abbr]["eFG%"],
            "TOV%": ff[away_abbr]["TOV%"],
            "ORB%": ff[away_abbr]["ORB%"],
            "FT/FGA": ff[away_abbr]["FT/FGA"],
            "Pace": ff[away_abbr]["pace"],
            "ORtg": ff[away_abbr]["ORtg"],
            "IsRegular": is_regular,
        }
    )


def next_month_day_one(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    year = int(qs.get("year", ["0"])[0])
    month = int(qs.get("month", ["0"])[0])

    month += 1
    if month > 12:
        month = 1
        year += 1

    return f"{BASE}/boxscores/?month={month}&day=1&year={year}"


async def scrape_day(page, day_url: str):
    print(f"\n===== DAY | {day_url} =====")

    year, month, day, date_str = parse_date_from_url(day_url)

    # check if we've reached the early exit date
    if EARLY_EXIT and year >= EARLY_EXIT[0] and month >= EARLY_EXIT[1]:
        print(f"[STOP] Reached specified EARLY_EXIT: {EARLY_EXIT}")
        return None

    # stop at July 2025
    if year >= 2025 and month >= 7:
        print(f"[STOP] Reached offseason {year}-{month}")
        return None

    # skip July–Sept
    if month in [7, 8, 9]:
        print(f"    [SKIP] No games in month {month}")
        return next_month_day_one(day_url)

    html = await fetch_html(page, day_url)
    soup = BeautifulSoup(html, "html.parser")

    content = soup.select_one("#content")

    if content and "No games played" in content.get_text():
        print("    [INFO] No games today.")
    else:
        gs = content.select_one("div.game_summaries") if content else None
        if gs:
            games = gs.select("div.game_summary")
            print(f"    [INFO] Found {len(games)} games on {date_str}")

            for g in games:
                link = g.find("a", string=lambda s: s and "Box Score" in s)
                if link:
                    game_url = urljoin(BASE, link["href"])
                    try:
                        await scrape_game(page, game_url, date_str)
                    except Exception as e:
                        print(f"    [ERROR] {e}")

    next_link = soup.select_one("a.button2.next")
    if not next_link:
        print("    [INFO] No next-day link")
        return None

    return urljoin(BASE, next_link["href"])


async def main():
    # register signal handler for graceful exit with autosave
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=120)
        page = await browser.new_page()

        # 🔥 BLOCK ADS, JS, IMAGES, TRACKERS
        async def block_non_html(route, request):
            resource = request.resource_type
            url = request.url

            # allow only HTML
            if resource == "document":
                await route.continue_()
                return

            # block everything else
            if resource in {
                "script",
                "image",
                "stylesheet",
                "font",
                "xhr",
                "fetch",
                "websocket",
            }:
                await route.abort()
                return

            blocked_domains = [
                "googletagmanager.com",
                "google-analytics.com",
                "doubleclick.net",
                "scorecardresearch.com",
                "facebook.net",
                "quantserve.com",
                "adsystem.com",
            ]
            if any(d in url for d in blocked_domains):
                await route.abort()
                return

            await route.abort()

        page.route("**/*", block_non_html)

        url = START_URL

        try:
            while url:
                url = await scrape_day(page, url)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
        finally:
            print("\n[DONE] Closing browser...")
            await browser.close()
            # autosave on exit
            save_data()


if __name__ == "__main__":
    asyncio.run(main())
