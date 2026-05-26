from __future__ import annotations

from pathlib import Path
import argparse
import re
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request

from bs4 import BeautifulSoup
import pandas as pd
from nba_api.stats.endpoints import drafthistory, leaguedashplayerstats


RAW_DIR = Path("data/raw")
STATIC_DIR = Path("data/static")

RAW_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_REPUTATION_PATH = RAW_DIR / "reputation_signals.csv"
DEFAULT_VOTE_SHARE_PATH = RAW_DIR / "all_nba_vote_share.csv"
DEFAULT_BREF_ADVANCED_PATH = RAW_DIR / "bref_player_advanced.csv"
DEFAULT_DRAFT_HISTORY_PATH = STATIC_DIR / "draft_history.csv"

BREF_SEASON_REQUEST_SLEEP_SECONDS = 4.0


MEASURE_TYPES = {
    "player_base_stats.csv": "Base",
    "player_advanced_stats.csv": "Advanced",
    "player_misc_stats.csv": "Misc",
    "player_usage_stats.csv": "Usage",
    "player_scoring_stats.csv": "Scoring",
    "player_defense_stats.csv": "Defense",
}

RECENCY_MEASURE_TYPES = {
    "player_base_last20_stats.csv": "Base",
    "player_advanced_last20_stats.csv": "Advanced",
}

COLLECTION_TARGETS = [
    "stats",
    "awards",
    "reputation",
    "vote_share",
    "bref_advanced",
    "draft_history",
]


ALL_NBA_URL = "https://www.nba.com/news/history-all-nba-teams"
ALL_ROOKIE_URL = "https://www.nba.com/news/history-all-rookie-teams"
ALL_STAR_RECAP_URL = "https://www.nba.com/news/history-all-star-recap-{season}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

NAME_FIXES = {
    "Kel’El Ware": "Kel'El Ware",
    "Kel’el Ware": "Kel'El Ware",
    "Donovan Clinglan": "Donovan Clingan",
}

PLAYER_KEY_ALIASES = {
    "jimmy butler": "jimmy butler iii",
    "mike conley jr.": "mike conley",
    "omer asık": "omer asik",
    "j.j. hickson": "jj hickson",
    "karl-anthony towns *": "karl-anthony towns",
    "ben simmons *": "ben simmons",
}

BREF_ADVANCED_FEATURE_MAP = {
    "per": "BREF_PER",
    "ows": "BREF_OWS",
    "dws": "BREF_DWS",
    "ws": "BREF_WS",
    "ws_per_48": "BREF_WS_PER_48",
    "obpm": "BREF_OBPM",
    "dbpm": "BREF_DBPM",
    "bpm": "BREF_BPM",
    "vorp": "BREF_VORP",
}

MANUAL_MVP_2014_POINTS = {
    "Kevin Durant": 1232,
    "LeBron James": 891,
    "Blake Griffin": 434,
    "Joakim Noah": 322,
    "James Harden": 85,
    "Stephen Curry": 66,
    "Chris Paul": 45,
    "Al Jefferson": 34,
    "Paul George": 33,
    "LaMarcus Aldridge": 26,
    "Kevin Love": 25,
    "Tim Duncan": 21,
    "Tony Parker": 21,
    "Dirk Nowitzki": 7,
    "Carmelo Anthony": 4,
    "Goran Dragic": 3,
    "Mike Conley": 1,
}

MANUAL_MVP_2020_POINTS = {
    "Giannis Antetokounmpo": 962,
    "LeBron James": 753,
    "James Harden": 367,
    "Luka Dončić": 200,
    "Kawhi Leonard": 168,
    "Anthony Davis": 82,
    "Chris Paul": 26,
    "Damian Lillard": 23,
    "Nikola Jokić": 18,
    "Pascal Siakam": 17,
    "Jimmy Butler": 9,
    "Jayson Tatum": 1,
}

MANUAL_ALL_STAR_OVERRIDES = {
    2026: [
        "Scottie Barnes",
        "Devin Booker",
        "Cade Cunningham",
        "Jalen Duren",
        "Anthony Edwards",
        "Chet Holmgren",
        "Jalen Johnson",
        "Tyrese Maxey",
        "Jaylen Brown",
        "Jalen Brunson",
        "Kevin Durant",
        "De'Aaron Fox",
        "Brandon Ingram",
        "LeBron James",
        "Kawhi Leonard",
        "Donovan Mitchell",
        "Stephen Curry",
        "Deni Avdija",
        "Luka Dončić",
        "Shai Gilgeous-Alexander",
        "Nikola Jokić",
        "Jamal Murray",
        "Norman Powell",
        "Alperen Sengun",
        "Pascal Siakam",
        "Karl-Anthony Towns",
        "Victor Wembanyama",
        "Giannis Antetokounmpo",
    ]
}

MANUAL_MVP_OVERRIDES = {
    2025: [
        ("Shai Gilgeous-Alexander", 913.0, 1000.0),
        ("Nikola Jokić", 787.0, 1000.0),
        ("Giannis Antetokounmpo", 470.0, 1000.0),
        ("Jayson Tatum", 311.0, 1000.0),
        ("Donovan Mitchell", 74.0, 1000.0),
        ("LeBron James", 16.0, 1000.0),
        ("Cade Cunningham", 12.0, 1000.0),
        ("Anthony Edwards", 12.0, 1000.0),
        ("Stephen Curry", 2.0, 1000.0),
        ("Jalen Brunson", 1.0, 1000.0),
        ("James Harden", 1.0, 1000.0),
        ("Evan Mobley", 1.0, 1000.0),
    ]
}


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""

    value = str(name)
    value = value.replace("’", "'")
    value = value.replace("‘", "'")
    value = value.replace("`", "'")
    value = value.replace("´", "'")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = " ".join(value.split())
    return value


def normalize_vote_share_player_key(name: str) -> str:
    player_key = normalize_name(name)
    player_key = player_key.replace("ı", "i").replace("*", "").strip()
    return PLAYER_KEY_ALIASES.get(player_key, player_key)


def clean_text(text: str) -> str:
    value = str(text)
    value = value.replace("\xa0", " ")
    value = value.replace("’", "'")
    value = value.replace("‘", "'")
    value = value.replace("`", "'")
    value = value.replace("´", "'")
    value = " ".join(value.split())
    return value.strip()


def fetch_html(url: str, max_attempts: int = 5) -> str:
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(url, headers=HEADERS)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == max_attempts:
                raise

            sleep_seconds = 2 * attempt
            print(
                f"HTTP 429 for {url} on attempt {attempt}/{max_attempts}. "
                f"Sleeping {sleep_seconds}s before retry."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch HTML after {max_attempts} attempts: {url}")


def fetch_bytes(url: str, max_attempts: int = 5) -> bytes:
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(url, headers=HEADERS)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == max_attempts:
                raise

            sleep_seconds = 2 * attempt
            print(
                f"HTTP 429 for {url} on attempt {attempt}/{max_attempts}. "
                f"Sleeping {sleep_seconds}s before retry."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch bytes after {max_attempts} attempts: {url}")


def download_page_lines(url: str) -> list[str]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    return [clean_text(line) for line in text.splitlines() if clean_text(line)]


def season_to_nba_format(season_end_year: int) -> str:
    start_year = season_end_year - 1
    end_year_short = str(season_end_year)[-2:]
    return f"{start_year}-{end_year_short}"


def fetch_player_stats_for_season(
    season_end_year: int,
    measure_type: str,
    player_experience: str | None = None,
    last_n_games: int = 0,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    season_str = season_to_nba_format(season_end_year)

    for attempt in range(1, retries + 1):
        try:
            extra_info = ""
            if player_experience is not None:
                extra_info = f", player_experience={player_experience}"
            if last_n_games > 0:
                extra_info += f", last_n_games={last_n_games}"

            print(
                f"Downloading {measure_type} stats for {season_str}"
                f"{extra_info}, attempt {attempt}"
            )

            params = {
                "season": season_str,
                "season_type_all_star": "Regular Season",
                "per_mode_detailed": "PerGame",
                "measure_type_detailed_defense": measure_type,
                "timeout": 60,
            }

            if player_experience is not None:
                params["player_experience_nullable"] = player_experience
            if last_n_games > 0:
                params["last_n_games"] = str(last_n_games)

            endpoint = leaguedashplayerstats.LeagueDashPlayerStats(**params)
            df = endpoint.get_data_frames()[0]
            df["SEASON"] = season_end_year
            df["SEASON_STR"] = season_str
            return df
        except Exception as exc:
            print(
                f"Error while downloading {measure_type} stats "
                f"for {season_str}: {exc}"
            )

            if attempt == retries:
                raise

            time.sleep(sleep_seconds)

    raise RuntimeError("Unexpected download error")


def collect_player_stats(start_season: int, end_season: int) -> dict[str, pd.DataFrame]:
    frames_by_file = {filename: [] for filename in MEASURE_TYPES}
    frames_by_file.update({filename: [] for filename in RECENCY_MEASURE_TYPES})
    rookie_frames = []

    for season in range(start_season, end_season + 1):
        for filename, measure_type in MEASURE_TYPES.items():
            frames_by_file[filename].append(
                fetch_player_stats_for_season(
                    season_end_year=season,
                    measure_type=measure_type,
                )
            )
            time.sleep(1.5)

        for filename, measure_type in RECENCY_MEASURE_TYPES.items():
            frames_by_file[filename].append(
                fetch_player_stats_for_season(
                    season_end_year=season,
                    measure_type=measure_type,
                    last_n_games=20,
                )
            )
            time.sleep(1.5)

        rookie_frames.append(
            fetch_player_stats_for_season(
                season_end_year=season,
                measure_type="Base",
                player_experience="Rookie",
            )
        )
        time.sleep(1.5)

    merged_frames = {
        filename: pd.concat(frames, ignore_index=True)
        for filename, frames in frames_by_file.items()
    }
    merged_frames["player_rookie_stats.csv"] = pd.concat(
        rookie_frames,
        ignore_index=True,
    )
    return merged_frames


def collect_or_reuse_stats(
    start_season: int,
    end_season: int,
    refresh: bool,
) -> None:
    required_files = list(MEASURE_TYPES) + list(RECENCY_MEASURE_TYPES) + [
        "player_rookie_stats.csv"
    ]
    missing_files = [filename for filename in required_files if not (RAW_DIR / filename).exists()]

    if not refresh and not missing_files:
        print("Using existing NBA stats files from data/raw")
        return

    if missing_files and not refresh:
        print("Missing stats files detected, refreshing stats collection")

    merged_frames = collect_player_stats(start_season=start_season, end_season=end_season)

    print()
    print("Saved files:")
    for filename, df in merged_frames.items():
        path = RAW_DIR / filename
        df.to_csv(path, index=False)
        print(f"{path} shape: {df.shape}")


def season_end_year(season_label: str) -> int:
    start_year = int(season_label.split("-")[0])
    return start_year + 1


def detect_team_line(line: str, award_type: str):
    upper = clean_text(line).upper().replace(":", "")

    if award_type == "all_nba":
        mapping = {
            "FIRST TEAM": ("first all-nba team", 3),
            "SECOND TEAM": ("second all-nba team", 2),
            "THIRD TEAM": ("third all-nba team", 1),
        }
    elif award_type == "rookie":
        mapping = {
            "FIRST TEAM": ("first rookie all-nba team", 2),
            "SECOND TEAM": ("second rookie all-nba team", 1),
        }
    else:
        raise ValueError(f"Unknown award_type: {award_type}")

    for marker, value in mapping.items():
        if upper == marker or upper.endswith(marker):
            return value

    return None


def looks_like_player_line(line: str) -> bool:
    value = clean_text(line)

    if value.startswith("•"):
        return True

    if re.match(r"^[FGC]\s*:\s*", value):
        return True

    if "," not in value:
        return False

    possible_name = value.split(",", 1)[0].strip()
    possible_name = re.sub(r"^[FGC]\s*:\s*", "", possible_name).strip()

    if not possible_name:
        return False
    if any(ch.isdigit() for ch in possible_name):
        return False
    if len(possible_name.split()) > 6:
        return False
    if len(possible_name) < 3:
        return False

    return True


def extract_player_name(line: str) -> str:
    value = clean_text(line)

    if value.startswith("•"):
        value = value[1:].strip()

    value = re.sub(r"^[FGC]\s*:\s*", "", value).strip()
    player = value.split(",", 1)[0].strip()
    return NAME_FIXES.get(player, player)


def parse_awards_page(
    url: str,
    award_type: str,
    min_season: int,
    max_season: int,
) -> pd.DataFrame:
    lines = download_page_lines(url)
    season_pattern = re.compile(r"^>?\s*(\d{4}-\d{2})$")

    if award_type == "all_nba":
        target_col = "target_all_nba"
    elif award_type == "rookie":
        target_col = "target_rookie"
    else:
        raise ValueError(f"Unknown award_type: {award_type}")

    rows = []
    current_season = None
    current_team_label = None
    current_target = None
    team_counter: dict[tuple[int, str], int] = {}

    for line in lines:
        season_match = season_pattern.match(line)
        if season_match:
            current_season = season_end_year(season_match.group(1))
            current_team_label = None
            current_target = None
            continue

        detected_team = detect_team_line(line, award_type)
        if detected_team is not None:
            current_team_label, current_target = detected_team
            continue

        if current_season is None or current_team_label is None:
            continue

        if current_season < min_season or current_season > max_season:
            continue

        if not looks_like_player_line(line):
            continue

        counter_key = (current_season, current_team_label)
        if team_counter.get(counter_key, 0) >= 5:
            continue

        player = extract_player_name(line)
        if not player:
            continue

        rows.append(
            {
                "season": current_season,
                "player": player,
                "team_label": current_team_label,
                target_col: current_target,
            }
        )
        team_counter[counter_key] = team_counter.get(counter_key, 0) + 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No rows parsed from {url}")

    return df.drop_duplicates().sort_values(
        ["season", target_col, "player"],
        ascending=[True, False, True],
    )


def validate_counts(df: pd.DataFrame, name: str) -> None:
    print()
    print(f"{name} counts by season/team:")

    counts = (
        df.groupby(["season", "team_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["season", "team_label"])
    )
    print(counts.to_string(index=False))

    suspicious = counts[counts["count"] != 5]
    if not suspicious.empty:
        print()
        print(f"WARNING: suspicious teams in {name}:")
        print(suspicious.to_string(index=False))

    print()
    print(f"{name} total rows: {len(df)}")


def save_award_labels(min_season: int, max_season: int, refresh: bool) -> None:
    all_nba_path = RAW_DIR / "all_nba_labels.csv"
    rookie_path = RAW_DIR / "all_rookie_labels.csv"

    if not refresh and all_nba_path.exists() and rookie_path.exists():
        print("Using existing award label files from data/raw")
        return

    all_nba = parse_awards_page(
        url=ALL_NBA_URL,
        award_type="all_nba",
        min_season=min_season,
        max_season=max_season,
    )
    rookie = parse_awards_page(
        url=ALL_ROOKIE_URL,
        award_type="rookie",
        min_season=min_season,
        max_season=max_season,
    )

    all_nba.to_csv(all_nba_path, index=False)
    rookie.to_csv(rookie_path, index=False)

    print()
    print("Saved award labels:")
    print(all_nba_path)
    print(rookie_path)
    validate_counts(all_nba, "All-NBA")
    validate_counts(rookie, "All-Rookie")


def normalize_official_player_name(name: str) -> str:
    value = clean_text(name)
    if "," in value:
        parts = [part.strip() for part in value.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            value = f"{parts[1]} {parts[0]}"
    return value


def extract_pdf_text(url: str) -> str:
    pdf_bytes = fetch_bytes(url)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as pdf_file:
        pdf_file.write(pdf_bytes)
        pdf_file.flush()
        completed = subprocess.run(
            ["pdftotext", "-layout", pdf_file.name, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    return completed.stdout


def parse_all_star_selections_page(season: int) -> pd.DataFrame:
    if season >= 2022:
        return parse_all_star_roster_page(season)
    return parse_all_star_recap_page(season)


def parse_all_star_recap_page(season: int) -> pd.DataFrame:
    url = ALL_STAR_RECAP_URL.format(season=season)
    lines = download_page_lines(url)

    try:
        start_index = lines.index("All-Star Game rosters") + 1
    except ValueError as exc:
        raise RuntimeError(
            f"Could not find All-Star rosters section on page: {url}"
        ) from exc

    rows = []
    seen_keys = set()
    skip_next_player_line = False

    for line in lines[start_index:]:
        if line in {"All-Star Weekend Wrap", "Related", "Latest"}:
            break
        if line.startswith("Coach:"):
            skip_next_player_line = True
            continue
        if line.startswith("NOTE:"):
            continue
        if skip_next_player_line:
            skip_next_player_line = False
            continue
        if "(" not in line or ")" not in line:
            continue

        player = clean_text(line.split("(", 1)[0])
        if not player:
            continue

        player_key = normalize_vote_share_player_key(player)
        dedupe_key = (season, player_key)
        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        rows.append(
            {
                "season": season,
                "player": player,
                "player_key": player_key,
                "is_all_star_this_season": 1,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No All-Star rows parsed for season {season}")

    return result.sort_values(["season", "player"]).reset_index(drop=True)


def parse_all_star_roster_page(season: int) -> pd.DataFrame:
    if season >= 2024:
        url = f"https://www.nba.com/allstar/{season}/roster"
    else:
        url = f"https://www.nba.com/allstar/{season}/all-star-roster"

    lines = download_page_lines(url)
    rows = []
    seen_keys = set()

    for index, line in enumerate(lines[:-1]):
        if season >= 2025 and line == "Candace's Rising Stars" and rows:
            break

        metadata_line = (
            line.startswith("#") or " | #" in line
        ) and any(marker in line for marker in ["Frontcourt", "Guard", "Center"])

        if not metadata_line:
            continue

        player = clean_text(lines[index + 1])

        if (
            not player
            or "|" in player
            or any(ch.isdigit() for ch in player)
            or player in {"PTS", "AST", "REB", "Starters", "Reserves", "Captain"}
        ):
            continue

        player_key = normalize_vote_share_player_key(player)
        dedupe_key = (season, player_key)
        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        rows.append(
            {
                "season": season,
                "player": player,
                "player_key": player_key,
                "is_all_star_this_season": 1,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No All-Star rows parsed for season {season}")

    return result.sort_values(["season", "player"]).reset_index(drop=True)


def get_mvp_award_page_url(season: int) -> str | None:
    if season < 2014:
        return None
    if season >= 2018:
        return (
            "https://pr.nba.com/voting-results-"
            f"{season - 1}-{str(season)[-2:]}-nba-regular-season-awards/"
        )
    if season == 2014:
        return "https://official.nba.com/2014-nba-year-annual-award-voting-results/"
    if season == 2015:
        return "https://official.nba.com/2015-nba-year-annual-award-voting-results/"
    return (
        "https://official.nba.com/"
        f"{season - 1}-{str(season)[-2:]}-nba-annual-award-voting-results/"
    )


def get_mvp_pdf_url_from_award_page(season: int) -> str | None:
    award_page_url = get_mvp_award_page_url(season)
    if award_page_url is None:
        return None

    html = fetch_html(award_page_url)
    soup = BeautifulSoup(html, "html.parser")
    mvp_heading = None

    for heading in soup.find_all(["h3", "h4", "p"]):
        if "Most Valuable Player" in heading.get_text(" ", strip=True):
            mvp_heading = heading
            break

    if mvp_heading is None:
        raise RuntimeError(f"Could not find MVP heading on page: {award_page_url}")

    for sibling in mvp_heading.next_siblings:
        if not getattr(sibling, "find_all", None):
            continue

        link = sibling.find("a", href=True)
        if link is None:
            continue

        href = link.get("href", "").strip()
        if not href:
            continue
        if href.startswith("//"):
            href = f"https:{href}"
        elif href.startswith("/"):
            href = f"https://official.nba.com{href}"

        return href

    raise RuntimeError(f"Could not find MVP results link on page: {award_page_url}")


def parse_mvp_totals_pdf_text(season: int, pdf_text: str) -> pd.DataFrame:
    rows = []

    for raw_line in pdf_text.splitlines():
        line = clean_text(raw_line)
        if not line:
            continue

        match = re.match(
            r"^(?P<name_team>.+?)\s+"
            r"(?P<v1>\d+)\s+(?P<v2>\d+)\s+(?P<v3>\d+)\s+"
            r"(?P<v4>\d+)\s+(?P<v5>\d+)\s+(?P<points>\d+)$",
            line,
        )
        if match is None:
            continue

        name_team = clean_text(match.group("name_team"))
        if "(" in name_team:
            player = normalize_official_player_name(name_team.split("(", 1)[0])
        else:
            player = clean_text(name_team.split(",", 1)[0])

        rows.append(
            {
                "season": season,
                "player": player,
                "player_key": normalize_vote_share_player_key(player),
                "mvp_vote_points": float(match.group("points")),
                "votes_first": int(match.group("v1")),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No MVP totals rows parsed for season {season}")

    max_points = float(result["votes_first"].sum() * 10)
    result["mvp_vote_max_points"] = max_points
    result["mvp_vote_share"] = result["mvp_vote_points"] / max_points

    return result[
        [
            "season",
            "player",
            "player_key",
            "mvp_vote_points",
            "mvp_vote_max_points",
            "mvp_vote_share",
        ]
    ]


def parse_mvp_ballot_pdf_text(season: int, pdf_text: str) -> pd.DataFrame:
    point_values = [10, 7, 5, 3, 1]
    player_points: dict[str, float] = {}
    ballot_count = 0

    for raw_line in pdf_text.splitlines():
        line = (
            raw_line.replace("\xa0", " ")
            .replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
            .replace("´", "'")
            .rstrip()
        )

        if " -- " not in line:
            continue

        parts = [
            clean_text(part)
            for part in re.split(r"\s{2,}", line.strip())
            if clean_text(part)
        ]
        if len(parts) < 7:
            continue

        raw_players = parts[-5:]
        if len(raw_players) != 5 or any(" -- " not in player for player in raw_players):
            continue

        ballot_count += 1
        for raw_player, points in zip(raw_players, point_values):
            player = normalize_official_player_name(raw_player.split(" -- ", 1)[0])
            player_points[player] = player_points.get(player, 0.0) + float(points)

    if ballot_count == 0:
        raise RuntimeError(f"No MVP ballot rows parsed for season {season}")

    max_points = float(ballot_count * 10)
    rows = [
        {
            "season": season,
            "player": player,
            "player_key": normalize_vote_share_player_key(player),
            "mvp_vote_points": points,
            "mvp_vote_max_points": max_points,
            "mvp_vote_share": points / max_points,
        }
        for player, points in sorted(
            player_points.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return pd.DataFrame(rows)


def parse_mvp_vote_page(season: int) -> pd.DataFrame:
    empty_df = pd.DataFrame(
        columns=[
            "season",
            "player",
            "player_key",
            "mvp_vote_points",
            "mvp_vote_max_points",
            "mvp_vote_share",
        ]
    )

    if season < 2014:
        return empty_df

    if season == 2014:
        max_points = 1250.0
        rows = [
            {
                "season": season,
                "player": player,
                "player_key": normalize_vote_share_player_key(player),
                "mvp_vote_points": float(points),
                "mvp_vote_max_points": max_points,
                "mvp_vote_share": float(points) / max_points,
            }
            for player, points in MANUAL_MVP_2014_POINTS.items()
        ]
        return pd.DataFrame(rows).sort_values(
            ["season", "mvp_vote_share", "player"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    if season == 2020:
        max_points = 1010.0
        rows = [
            {
                "season": season,
                "player": player,
                "player_key": normalize_vote_share_player_key(player),
                "mvp_vote_points": float(points),
                "mvp_vote_max_points": max_points,
                "mvp_vote_share": float(points) / max_points,
            }
            for player, points in MANUAL_MVP_2020_POINTS.items()
        ]
        return pd.DataFrame(rows).sort_values(
            ["season", "mvp_vote_share", "player"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    try:
        pdf_url = get_mvp_pdf_url_from_award_page(season)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(
                f"Skipping MVP vote-share parse for season {season}: "
                "award page is not available yet"
            )
            return empty_df
        raise

    if pdf_url is None:
        raise RuntimeError(f"Could not determine MVP results URL for season {season}")

    if not pdf_url.lower().endswith(".pdf"):
        print(
            f"Skipping MVP vote-share parse for season {season}: "
            f"results link is not a PDF ({pdf_url})"
        )
        return empty_df

    pdf_text = extract_pdf_text(pdf_url)
    if re.search(r"Voter\s+Affiliation", pdf_text):
        result = parse_mvp_ballot_pdf_text(season=season, pdf_text=pdf_text)
    else:
        result = parse_mvp_totals_pdf_text(season=season, pdf_text=pdf_text)

    if result.empty:
        raise RuntimeError(f"No MVP vote rows parsed for season {season}")

    return result.sort_values(
        ["season", "mvp_vote_share", "player"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def ensure_reputation_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season": 0,
        "player": "",
        "player_key": "",
        "is_all_star_this_season": 0,
        "mvp_vote_points": 0.0,
        "mvp_vote_max_points": 0.0,
        "mvp_vote_share": 0.0,
    }

    result = df.copy()
    for col, default in required.items():
        if col not in result.columns:
            result[col] = default

    result["player_key"] = result.apply(
        lambda row: normalize_name(row["player"])
        if not str(row["player_key"]).strip()
        else normalize_name(row["player_key"]),
        axis=1,
    )
    return result


def upsert_reputation_row(
    df: pd.DataFrame,
    season: int,
    player: str,
    is_all_star: int | None = None,
    mvp_points: float | None = None,
    mvp_max_points: float | None = None,
) -> pd.DataFrame:
    result = df.copy()
    player_key = normalize_name(player)

    mask = (
        result["season"].astype(int) == int(season)
    ) & (result["player_key"] == player_key)

    if not mask.any():
        new_row = {col: pd.NA for col in result.columns}
        new_row["season"] = int(season)
        new_row["player"] = player
        new_row["player_key"] = player_key
        new_row["is_all_star_this_season"] = 0
        new_row["mvp_vote_points"] = 0.0
        new_row["mvp_vote_max_points"] = 0.0
        new_row["mvp_vote_share"] = 0.0
        result.loc[len(result)] = new_row
        mask = (
            result["season"].astype(int) == int(season)
        ) & (result["player_key"] == player_key)

    if is_all_star is not None:
        result.loc[mask, "is_all_star_this_season"] = int(is_all_star)

    if mvp_points is not None:
        max_points = 0.0 if mvp_max_points is None else float(mvp_max_points)
        result.loc[mask, "mvp_vote_points"] = float(mvp_points)
        result.loc[mask, "mvp_vote_max_points"] = max_points
        result.loc[mask, "mvp_vote_share"] = 0.0 if max_points == 0 else float(
            mvp_points
        ) / max_points

    return result


def apply_manual_reputation_overrides(df: pd.DataFrame) -> pd.DataFrame:
    result = ensure_reputation_columns(df)

    for season, players in MANUAL_ALL_STAR_OVERRIDES.items():
        for player in players:
            result = upsert_reputation_row(
                df=result,
                season=season,
                player=player,
                is_all_star=1,
            )

    for season, rows in MANUAL_MVP_OVERRIDES.items():
        for player, points, max_points in rows:
            result = upsert_reputation_row(
                df=result,
                season=season,
                player=player,
                mvp_points=points,
                mvp_max_points=max_points,
            )

    result["season"] = result["season"].astype(int)
    result["is_all_star_this_season"] = result["is_all_star_this_season"].fillna(0).astype(int)
    result["mvp_vote_points"] = result["mvp_vote_points"].fillna(0.0).astype(float)
    result["mvp_vote_max_points"] = result["mvp_vote_max_points"].fillna(0.0).astype(float)
    result["mvp_vote_share"] = result["mvp_vote_share"].fillna(0.0).astype(float)

    return (
        result.sort_values(["season", "player_key"])
        .drop_duplicates(["season", "player_key"], keep="last")
        .reset_index(drop=True)
    )


def collect_reputation_signals(min_season: int, max_season: int) -> pd.DataFrame:
    all_star_frames = []
    mvp_frames = []

    for season in range(min_season, max_season + 1):
        print(f"Downloading reputation signals for season {season}")
        all_star_frames.append(parse_all_star_selections_page(season))
        mvp_df = parse_mvp_vote_page(season)
        if not mvp_df.empty:
            mvp_frames.append(mvp_df)
        time.sleep(1.0)

    all_star_df = pd.concat(all_star_frames, ignore_index=True)
    mvp_df = (
        pd.concat(mvp_frames, ignore_index=True)
        if mvp_frames
        else pd.DataFrame(
            columns=[
                "season",
                "player",
                "player_key",
                "mvp_vote_points",
                "mvp_vote_max_points",
                "mvp_vote_share",
            ]
        )
    )

    result = all_star_df[
        ["season", "player", "player_key", "is_all_star_this_season"]
    ].merge(
        mvp_df[
            [
                "season",
                "player",
                "player_key",
                "mvp_vote_points",
                "mvp_vote_max_points",
                "mvp_vote_share",
            ]
        ],
        on=["season", "player_key"],
        how="outer",
        suffixes=("_all_star", "_mvp"),
    )

    result["player"] = (
        result["player_all_star"].fillna(result["player_mvp"]).astype(str).str.strip()
    )
    result = result.drop(columns=["player_all_star", "player_mvp"])
    result["is_all_star_this_season"] = result["is_all_star_this_season"].fillna(0).astype(int)

    for col in ["mvp_vote_points", "mvp_vote_max_points", "mvp_vote_share"]:
        result[col] = result[col].fillna(0.0)

    result = apply_manual_reputation_overrides(result)

    return result.sort_values(
        ["season", "is_all_star_this_season", "mvp_vote_share", "player"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def save_reputation_signals(
    min_season: int,
    max_season: int,
    output_path: Path,
) -> Path:
    df = collect_reputation_signals(min_season=min_season, max_season=max_season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved reputation signals to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Season range: {df['season'].min()}-{df['season'].max()}")
    print()
    print("All-Star positive rows by season:")
    print(
        df.groupby("season")["is_all_star_this_season"]
        .sum()
        .astype(int)
        .to_string()
    )
    print()
    print("MVP vote-share positive rows by season:")
    positive_mvp = df[df["mvp_vote_share"] > 0]
    if positive_mvp.empty:
        print("No positive MVP vote-share rows")
    else:
        print(positive_mvp.groupby("season").size().astype(int).to_string())

    return output_path


def map_team_label(raw_value: str) -> tuple[str, int] | None:
    value = str(raw_value).strip().upper()
    mapping = {
        "1T": ("first all-nba team", 3),
        "2T": ("second all-nba team", 2),
        "3T": ("third all-nba team", 1),
        "1ST": ("first all-nba team", 3),
        "2ND": ("second all-nba team", 2),
        "3RD": ("third all-nba team", 1),
        "ORV": ("others receiving votes", 0),
    }
    return mapping.get(value)


def parse_all_nba_vote_share_page(season: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/awards/awards_{season}.html"
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    table = soup.find("table", id="leading_all_nba")

    if table is None:
        raise RuntimeError(
            f"Could not find table 'leading_all_nba' on page: {url}"
        )

    rows = []
    for row in table.find("tbody").find_all("tr"):
        player_cell = row.find(["th", "td"], {"data-stat": "player"})
        if player_cell is None:
            continue

        player = player_cell.get_text(" ", strip=True)
        if not player:
            continue

        team_cell = row.find(["th", "td"], {"data-stat": "all_nba_team"})
        team_raw = "" if team_cell is None else team_cell.get_text(" ", strip=True)
        mapped_team = map_team_label(team_raw)
        if mapped_team is None:
            continue

        points_won_cell = row.find(["th", "td"], {"data-stat": "points_won"})
        points_max_cell = row.find(["th", "td"], {"data-stat": "points_max"})
        share_cell = row.find(["th", "td"], {"data-stat": "award_share"})
        first_votes_cell = row.find(["th", "td"], {"data-stat": "first_team_votes"})
        second_votes_cell = row.find(["th", "td"], {"data-stat": "second_team_votes"})
        third_votes_cell = row.find(["th", "td"], {"data-stat": "third_team_votes"})
        team_label, target_all_nba = mapped_team

        rows.append(
            {
                "season": season,
                "player": player,
                "player_key": normalize_vote_share_player_key(player),
                "team_label": team_label,
                "target_all_nba": target_all_nba,
                "all_nba_vote_points": float(points_won_cell.get_text(strip=True) or 0),
                "all_nba_vote_max_points": float(
                    points_max_cell.get_text(strip=True) or 0
                ),
                "all_nba_vote_share": float(share_cell.get_text(strip=True) or 0),
                "all_nba_first_team_votes": int(
                    first_votes_cell.get_text(strip=True) or 0
                ),
                "all_nba_second_team_votes": int(
                    second_votes_cell.get_text(strip=True) or 0
                ),
                "all_nba_third_team_votes": int(
                    third_votes_cell.get_text(strip=True) or 0
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No All-NBA vote-share rows parsed for season {season}")

    return result.sort_values(
        ["season", "all_nba_vote_points", "player"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def collect_all_nba_vote_share(min_season: int, max_season: int) -> pd.DataFrame:
    frames = []
    seasons = list(range(min_season, max_season + 1))

    for index, season in enumerate(seasons):
        print(f"Downloading All-NBA vote share for season {season}")
        frames.append(parse_all_nba_vote_share_page(season))

        if index < len(seasons) - 1:
            time.sleep(BREF_SEASON_REQUEST_SLEEP_SECONDS)

    result = pd.concat(frames, ignore_index=True)
    counts = result.groupby("season").size()
    bad_counts = counts[counts < 15]
    if not bad_counts.empty:
        raise ValueError(
            "Unexpectedly low All-NBA vote-share row counts by season: "
            f"{bad_counts.to_dict()}"
        )
    return result


def save_all_nba_vote_share(
    min_season: int,
    max_season: int,
    output_path: Path,
) -> Path:
    df = collect_all_nba_vote_share(min_season=min_season, max_season=max_season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved All-NBA vote share to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Season range: {df['season'].min()}-{df['season'].max()}")
    return output_path


def parse_numeric(value: str) -> float | None:
    parsed = str(value).strip()
    if parsed == "":
        return None
    return float(parsed)


def choose_best_player_row(group_df: pd.DataFrame) -> pd.Series:
    tot_rows = group_df[group_df["TEAM_ABBREVIATION"] == "TOT"]
    if not tot_rows.empty:
        return tot_rows.iloc[0]

    return (
        group_df.sort_values(
            ["MP", "G", "TEAM_ABBREVIATION"],
            ascending=[False, False, True],
        )
        .iloc[0]
    )


def parse_bref_advanced_page(season: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    table = soup.find("table", id="advanced")

    if table is None:
        raise RuntimeError(f"Could not find BRef advanced table for season {season}")

    rows = []
    for row in table.find("tbody").find_all("tr"):
        if "thead" in row.get("class", []):
            continue

        player_cell = row.find(["th", "td"], {"data-stat": "name_display"})
        if player_cell is None:
            continue

        player = player_cell.get_text(" ", strip=True)
        if not player:
            continue

        team_cell = row.find(["th", "td"], {"data-stat": "team_name_abbr"})
        team_abbreviation = "" if team_cell is None else team_cell.get_text(" ", strip=True)

        mp_cell = row.find(["th", "td"], {"data-stat": "mp"})
        g_cell = row.find(["th", "td"], {"data-stat": "games"})

        parsed_row = {
            "SEASON": season,
            "PLAYER_NAME_BREF": player,
            "player_key": normalize_vote_share_player_key(player),
            "TEAM_ABBREVIATION": team_abbreviation,
            "MP": 0.0 if mp_cell is None else float(mp_cell.get_text(strip=True) or 0.0),
            "G": 0.0 if g_cell is None else float(g_cell.get_text(strip=True) or 0.0),
        }

        for data_stat, output_col in BREF_ADVANCED_FEATURE_MAP.items():
            cell = row.find(["th", "td"], {"data-stat": data_stat})
            parsed_row[output_col] = None if cell is None else parse_numeric(
                cell.get_text(strip=True)
            )

        rows.append(parsed_row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No BRef advanced rows parsed for season {season}")

    return (
        result.groupby(["SEASON", "player_key"], sort=False, as_index=False)
        .apply(choose_best_player_row, include_groups=False)
        .reset_index(drop=True)
    )


def collect_bref_advanced(min_season: int, max_season: int) -> pd.DataFrame:
    frames = []
    seasons = list(range(min_season, max_season + 1))

    for index, season in enumerate(seasons):
        print(f"Downloading BRef advanced metrics for season {season}")
        frames.append(parse_bref_advanced_page(season))

        if index < len(seasons) - 1:
            time.sleep(BREF_SEASON_REQUEST_SLEEP_SECONDS)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["SEASON", "PLAYER_NAME_BREF"]).reset_index(drop=True)


def save_bref_advanced(
    min_season: int,
    max_season: int,
    output_path: Path,
) -> Path:
    df = collect_bref_advanced(min_season=min_season, max_season=max_season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved BRef advanced metrics to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Season range: {df['SEASON'].min()}-{df['SEASON'].max()}")
    return output_path


def save_draft_history(output_path: Path) -> Path:
    df = drafthistory.DraftHistory(league_id="00", timeout=60).get_data_frames()[0]
    df = df.rename(
        columns={
            "PERSON_ID": "PLAYER_ID",
            "SEASON": "DRAFT_YEAR",
            "ROUND_NUMBER": "DRAFT_ROUND",
            "ROUND_PICK": "DRAFT_ROUND_PICK",
            "OVERALL_PICK": "DRAFT_OVERALL_PICK",
        }
    )

    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "DRAFT_YEAR",
        "DRAFT_ROUND",
        "DRAFT_ROUND_PICK",
        "DRAFT_OVERALL_PICK",
        "TEAM_ID",
        "TEAM_CITY",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()

    for col in [
        "PLAYER_ID",
        "DRAFT_YEAR",
        "DRAFT_ROUND",
        "DRAFT_ROUND_PICK",
        "DRAFT_OVERALL_PICK",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return output_path


def run_with_existing_file_fallback(
    label: str,
    output_path: Path,
    collector,
) -> Path:
    try:
        return collector()
    except Exception as exc:
        if output_path.exists():
            print(
                f"WARNING: failed to refresh {label}: {exc}. "
                f"Using existing file: {output_path}"
            )
            return output_path
        raise


def run_optional_collector(
    target_name: str,
    label: str,
    output_path: Path,
    collector,
    allow_missing: bool,
) -> Path | None:
    try:
        return run_with_existing_file_fallback(
            label=label,
            output_path=output_path,
            collector=collector,
        )
    except Exception as exc:
        if allow_missing:
            print(
                f"WARNING: skipping optional target '{target_name}' "
                f"because refresh failed and no cached file exists: {exc}"
            )
            return None
        raise


def resolve_targets(selected_targets: list[str]) -> list[str]:
    if not selected_targets or "all" in selected_targets:
        return COLLECTION_TARGETS

    ordered = []
    for target in COLLECTION_TARGETS:
        if target in selected_targets:
            ordered.append(target)
    return ordered


def collect_data(
    start_season: int,
    end_season: int,
    label_max_season: int,
    targets: list[str],
    refresh: bool,
) -> None:
    selected_targets = resolve_targets(targets)
    user_requested_all = (not targets) or ("all" in targets)
    print("Selected targets:", ", ".join(selected_targets))
    print(f"Season range: {start_season}-{end_season}")
    print(f"Label/vote-share max season: {label_max_season}")

    if "stats" in selected_targets:
        collect_or_reuse_stats(
            start_season=start_season,
            end_season=end_season,
            refresh=refresh,
        )

    if "awards" in selected_targets:
        save_award_labels(
            min_season=start_season,
            max_season=label_max_season,
            refresh=refresh,
        )

    if "reputation" in selected_targets:
        if refresh or not DEFAULT_REPUTATION_PATH.exists():
            run_with_existing_file_fallback(
                label="reputation signals",
                output_path=DEFAULT_REPUTATION_PATH,
                collector=lambda: save_reputation_signals(
                    min_season=start_season,
                    max_season=end_season,
                    output_path=DEFAULT_REPUTATION_PATH,
                ),
            )
        else:
            print(f"Using existing reputation file: {DEFAULT_REPUTATION_PATH}")

    if "vote_share" in selected_targets:
        if refresh or not DEFAULT_VOTE_SHARE_PATH.exists():
            run_optional_collector(
                target_name="vote_share",
                label="All-NBA vote share",
                output_path=DEFAULT_VOTE_SHARE_PATH,
                allow_missing=user_requested_all,
                collector=lambda: save_all_nba_vote_share(
                    min_season=start_season,
                    max_season=label_max_season,
                    output_path=DEFAULT_VOTE_SHARE_PATH,
                ),
            )
        else:
            print(f"Using existing vote-share file: {DEFAULT_VOTE_SHARE_PATH}")

    if "bref_advanced" in selected_targets:
        if refresh or not DEFAULT_BREF_ADVANCED_PATH.exists():
            run_optional_collector(
                target_name="bref_advanced",
                label="BRef advanced metrics",
                output_path=DEFAULT_BREF_ADVANCED_PATH,
                allow_missing=user_requested_all,
                collector=lambda: save_bref_advanced(
                    min_season=start_season,
                    max_season=end_season,
                    output_path=DEFAULT_BREF_ADVANCED_PATH,
                ),
            )
        else:
            print(f"Using existing BRef advanced file: {DEFAULT_BREF_ADVANCED_PATH}")

    if "draft_history" in selected_targets:
        if refresh or not DEFAULT_DRAFT_HISTORY_PATH.exists():
            run_with_existing_file_fallback(
                label="draft history",
                output_path=DEFAULT_DRAFT_HISTORY_PATH,
                collector=lambda: save_draft_history(DEFAULT_DRAFT_HISTORY_PATH),
            )
        else:
            print(f"Using existing draft history file: {DEFAULT_DRAFT_HISTORY_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument(
        "--label-max-season",
        type=int,
        default=None,
        help="Last season with final All-NBA / All-Rookie labels and vote-share.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["all"] + COLLECTION_TARGETS,
        default=["all"],
        help="Which sources to collect.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload and overwrite files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_max_season = args.label_max_season
    if label_max_season is None:
        label_max_season = max(args.start, args.end - 1)

    collect_data(
        start_season=args.start,
        end_season=args.end,
        label_max_season=label_max_season,
        targets=args.targets,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
