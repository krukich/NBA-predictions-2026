from __future__ import annotations

from pathlib import Path
import argparse
import unicodedata

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_DATASET_PATH = PROCESSED_DIR / "player_season_dataset.csv"
DEFAULT_REPUTATION_PATH = RAW_DIR / "reputation_signals.csv"

SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


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


def build_fallback_key(player_key: str) -> str:
    key = str(player_key).strip().lower()
    key = key.replace(".", "")
    key = key.replace("'", "")
    tokens = key.split()

    if tokens and tokens[-1] in SUFFIX_TOKENS:
        tokens = tokens[:-1]

    return " ".join(tokens)


def load_raw_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = RAW_DIR / "player_base_stats.csv"
    advanced_path = RAW_DIR / "player_advanced_stats.csv"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing file: {base_path}")
    if not advanced_path.exists():
        raise FileNotFoundError(f"Missing file: {advanced_path}")

    return pd.read_csv(base_path), pd.read_csv(advanced_path)


def load_optional_stat_tables() -> list[tuple[str, pd.DataFrame, str]]:
    optional_files = {
        "player_misc_stats.csv": "",
        "player_usage_stats.csv": "",
        "player_scoring_stats.csv": "",
        "player_defense_stats.csv": "",
        "player_base_last20_stats.csv": "L20_",
        "player_advanced_last20_stats.csv": "L20_ADV_",
    }

    tables = []
    for filename, prefix in optional_files.items():
        path = RAW_DIR / filename
        if path.exists():
            tables.append((filename, pd.read_csv(path), prefix))

    return tables


def merge_stat_tables(
    base: pd.DataFrame,
    stat_tables: list[tuple[str, pd.DataFrame, str]],
) -> pd.DataFrame:
    key_cols = ["SEASON", "PLAYER_ID", "PLAYER_NAME"]

    for col in key_cols:
        if col not in base.columns:
            raise ValueError(f"Missing column in base stats: {col}")

    df = base.copy()

    for table_name, table_df, prefix in stat_tables:
        for col in key_cols:
            if col not in table_df.columns:
                raise ValueError(f"Missing column in {table_name}: {col}")

        if prefix:
            extra_cols = [col for col in table_df.columns if col not in key_cols]
        else:
            extra_cols = [col for col in table_df.columns if col not in df.columns]

        if not extra_cols:
            continue

        table_small = table_df[key_cols + extra_cols].copy()

        if prefix:
            table_small = table_small.rename(
                columns={col: f"{prefix}{col}" for col in extra_cols}
            )

        df = df.merge(table_small, on=key_cols, how="left")

    df["player_key"] = df["PLAYER_NAME"].apply(normalize_name)
    return df


def add_rookie_flag(df: pd.DataFrame) -> pd.DataFrame:
    rookie_path = RAW_DIR / "player_rookie_stats.csv"
    if not rookie_path.exists():
        raise FileNotFoundError(f"Missing file: {rookie_path}")

    rookies = pd.read_csv(rookie_path)
    for col in ["SEASON", "PLAYER_ID"]:
        if col not in rookies.columns:
            raise ValueError(f"Missing column in rookie stats: {col}")

    rookie_keys = rookies[["SEASON", "PLAYER_ID"]].drop_duplicates()
    rookie_keys["IS_ROOKIE"] = 1

    result = df.merge(rookie_keys, on=["SEASON", "PLAYER_ID"], how="left")
    result["IS_ROOKIE"] = result["IS_ROOKIE"].fillna(0).astype(int)
    return result


def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_nba_path = RAW_DIR / "all_nba_labels.csv"
    all_rookie_path = RAW_DIR / "all_rookie_labels.csv"

    if not all_nba_path.exists():
        raise FileNotFoundError(f"Missing file: {all_nba_path}")
    if not all_rookie_path.exists():
        raise FileNotFoundError(f"Missing file: {all_rookie_path}")

    all_nba = pd.read_csv(all_nba_path)
    all_rookie = pd.read_csv(all_rookie_path)

    required_all_nba_cols = ["season", "player", "target_all_nba"]
    required_rookie_cols = ["season", "player", "target_rookie"]

    for col in required_all_nba_cols:
        if col not in all_nba.columns:
            raise ValueError(f"Missing column in all_nba_labels.csv: {col}")

    for col in required_rookie_cols:
        if col not in all_rookie.columns:
            raise ValueError(f"Missing column in all_rookie_labels.csv: {col}")

    all_nba["player_key"] = all_nba["player"].apply(normalize_name)
    all_rookie["player_key"] = all_rookie["player"].apply(normalize_name)

    return all_nba, all_rookie


def add_all_nba_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    result = merge_feature_with_fallback(df, labels, "target_all_nba")
    result["target_all_nba"] = result["target_all_nba"].fillna(0).astype(int)
    return result


def add_rookie_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    result = merge_feature_with_fallback(df, labels, "target_rookie")
    result["target_rookie"] = result["target_rookie"].fillna(0).astype(int)
    return result


def merge_feature_with_fallback(
    dataset_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    small = labels_df[["season", "player_key", value_col]].copy()

    result = dataset_df.merge(
        small,
        left_on=["SEASON", "player_key"],
        right_on=["season", "player_key"],
        how="left",
    ).drop(columns=["season"])

    missing_mask = result[value_col].isna()
    if not missing_mask.any():
        return result

    result["_fallback_key"] = result["player_key"].map(build_fallback_key)
    small["_fallback_key"] = small["player_key"].map(build_fallback_key)

    fallback_counts = (
        small.groupby(["season", "_fallback_key"])
        .size()
        .rename("fallback_count")
        .reset_index()
    )
    unique_fallbacks = fallback_counts[fallback_counts["fallback_count"] == 1][
        ["season", "_fallback_key"]
    ]

    fallback_small = (
        small.merge(unique_fallbacks, on=["season", "_fallback_key"], how="inner")
        .drop(columns=["player_key"])
        .drop_duplicates(["season", "_fallback_key"])
    )

    missing_rows = result.loc[missing_mask, ["SEASON", "_fallback_key"]].copy()
    missing_rows["_row_id"] = missing_rows.index

    fallback_merged = missing_rows.merge(
        fallback_small,
        left_on=["SEASON", "_fallback_key"],
        right_on=["season", "_fallback_key"],
        how="left",
    ).set_index("_row_id")

    result.loc[missing_mask, value_col] = result.loc[missing_mask, value_col].fillna(
        fallback_merged[value_col]
    )

    return result.drop(columns=["_fallback_key"])


def print_unmatched_labels(
    stats_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target_col: str,
) -> None:
    stats_keys = set(zip(stats_df["SEASON"], stats_df["player_key"]))
    fallback_counts = (
        stats_df.assign(_fallback_key=stats_df["player_key"].map(build_fallback_key))
        .groupby(["SEASON", "_fallback_key"])
        .size()
        .rename("fallback_count")
        .reset_index()
    )
    unique_fallbacks = set(
        zip(
            fallback_counts.loc[fallback_counts["fallback_count"] == 1, "SEASON"],
            fallback_counts.loc[fallback_counts["fallback_count"] == 1, "_fallback_key"],
        )
    )

    label_rows = labels_df[["season", "player", "player_key", target_col]].copy()
    label_rows["_fallback_key"] = label_rows["player_key"].map(build_fallback_key)

    unmatched = []
    for _, row in label_rows.iterrows():
        direct_key = (row["season"], row["player_key"])
        fallback_key = (row["season"], row["_fallback_key"])
        if direct_key not in stats_keys and fallback_key not in unique_fallbacks:
            unmatched.append(row.drop(labels="_fallback_key"))

    if not unmatched:
        print(f"No unmatched labels for {target_col}")
        return

    print()
    print(f"WARNING: unmatched labels for {target_col}:")
    print(pd.DataFrame(unmatched).to_string(index=False))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "GP" in result.columns and "MIN" in result.columns:
        result["TOTAL_MIN"] = result["GP"] * result["MIN"]
    if "GP" in result.columns and "PTS" in result.columns:
        result["TOTAL_PTS"] = result["GP"] * result["PTS"]
    if "GP" in result.columns and "REB" in result.columns:
        result["TOTAL_REB"] = result["GP"] * result["REB"]
    if "GP" in result.columns and "AST" in result.columns:
        result["TOTAL_AST"] = result["GP"] * result["AST"]

    if {"PTS", "REB", "AST"}.issubset(result.columns):
        result["PTS_REB_AST"] = result["PTS"] + result["REB"] + result["AST"]

    if {"STL", "BLK"}.issubset(result.columns):
        result["STOCKS"] = result["STL"] + result["BLK"]

    if "AST" in result.columns and "TOV" in result.columns:
        result["AST_TOV_SIMPLE"] = result["AST"] / (result["TOV"] + 0.1)

    recency_specs = [
        ("L20_GP", "L20_MIN", "L20_TOTAL_MIN"),
        ("L20_GP", "L20_PTS", "L20_TOTAL_PTS"),
        ("L20_GP", "L20_REB", "L20_TOTAL_REB"),
        ("L20_GP", "L20_AST", "L20_TOTAL_AST"),
    ]
    for gp_col, stat_col, out_col in recency_specs:
        if gp_col in result.columns and stat_col in result.columns:
            result[out_col] = result[gp_col] * result[stat_col]

    rank_cols = [
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "PLUS_MINUS",
        "W_PCT",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "AST_PCT",
        "REB_PCT",
        "EFG_PCT",
        "TS_PCT",
        "USG_PCT",
        "PIE",
        "TOTAL_MIN",
        "TOTAL_PTS",
        "TOTAL_REB",
        "TOTAL_AST",
        "PTS_REB_AST",
        "STOCKS",
    ]

    for col in rank_cols:
        if col not in result.columns:
            continue

        rank_col = f"{col}_PCT_RANK"
        ascending = col in {"DEF_RATING", "TOV"}
        result[rank_col] = result.groupby("SEASON")[col].rank(
            pct=True,
            ascending=ascending,
        )

    return result


def add_team_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "TEAM_ID" in result.columns:
        team_key_cols = ["SEASON", "TEAM_ID"]
    else:
        team_key_cols = ["SEASON", "TEAM_ABBREVIATION"]

    if "TEAM_COUNT" in result.columns:
        result["IS_MULTI_TEAM_PLAYER"] = (
            result["TEAM_COUNT"].fillna(1).astype(float) > 1
        ).astype(int)
    else:
        result["IS_MULTI_TEAM_PLAYER"] = 0

    result["TEAM_PLAYER_COUNT"] = (
        result.groupby(team_key_cols)["PLAYER_ID"].transform("count").astype(int)
    )

    team_rank_specs = [
        ("PTS", "TEAM_PTS_RANK"),
        ("REB", "TEAM_REB_RANK"),
        ("AST", "TEAM_AST_RANK"),
        ("MIN", "TEAM_MIN_RANK"),
        ("USG_PCT", "TEAM_USG_RANK"),
        ("PIE", "TEAM_PIE_RANK"),
        ("PLUS_MINUS", "TEAM_PLUS_MINUS_RANK"),
        ("TOTAL_PTS", "TEAM_TOTAL_PTS_RANK"),
        ("TOTAL_REB", "TEAM_TOTAL_REB_RANK"),
        ("TOTAL_AST", "TEAM_TOTAL_AST_RANK"),
        ("TOTAL_MIN", "TEAM_TOTAL_MIN_RANK"),
        ("STOCKS", "TEAM_STOCKS_RANK"),
    ]

    for source_col, rank_col in team_rank_specs:
        if source_col in result.columns:
            result[rank_col] = result.groupby(team_key_cols)[source_col].rank(
                method="min",
                ascending=False,
            )

    leader_specs = [
        ("TEAM_PTS_RANK", "IS_TEAM_PTS_LEADER"),
        ("TEAM_MIN_RANK", "IS_TEAM_MIN_LEADER"),
        ("TEAM_PIE_RANK", "IS_TEAM_PIE_LEADER"),
        ("TEAM_TOTAL_PTS_RANK", "IS_TEAM_TOTAL_PTS_LEADER"),
        ("TEAM_TOTAL_MIN_RANK", "IS_TEAM_TOTAL_MIN_LEADER"),
    ]
    for rank_col, flag_col in leader_specs:
        if rank_col in result.columns:
            result[flag_col] = (result[rank_col] == 1).astype(int)

    top3_specs = [
        ("TEAM_PTS_RANK", "TEAM_TOP3_PTS_FLAG"),
        ("TEAM_MIN_RANK", "TEAM_TOP3_MIN_FLAG"),
        ("TEAM_PIE_RANK", "TEAM_TOP3_PIE_FLAG"),
        ("TEAM_TOTAL_PTS_RANK", "TEAM_TOP3_TOTAL_PTS_FLAG"),
        ("TEAM_TOTAL_MIN_RANK", "TEAM_TOP3_TOTAL_MIN_FLAG"),
    ]
    for rank_col, flag_col in top3_specs:
        if rank_col in result.columns:
            result[flag_col] = (result[rank_col] <= 3).astype(int)

    share_specs = [
        ("TOTAL_PTS", "TEAM_TOTAL_PTS_SHARE"),
        ("TOTAL_REB", "TEAM_TOTAL_REB_SHARE"),
        ("TOTAL_AST", "TEAM_TOTAL_AST_SHARE"),
        ("TOTAL_MIN", "TEAM_TOTAL_MIN_SHARE"),
    ]
    for source_col, share_col in share_specs:
        if source_col not in result.columns:
            continue

        team_sum = result.groupby(team_key_cols)[source_col].transform("sum")
        team_sum = team_sum.replace(0, pd.NA)
        result[share_col] = (result[source_col] / team_sum).fillna(0.0)

    team_level_cols = [
        col
        for col in ["W", "W_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING"]
        if col in result.columns
    ]
    if not team_level_cols:
        return result

    team_df = result[team_key_cols + team_level_cols].drop_duplicates(
        subset=team_key_cols
    )

    league_rank_specs = [
        ("W", "LEAGUE_TEAM_W_RANK", False),
        ("W_PCT", "LEAGUE_TEAM_W_PCT_RANK", False),
        ("OFF_RATING", "LEAGUE_TEAM_OFF_RATING_RANK", False),
        ("DEF_RATING", "LEAGUE_TEAM_DEF_RATING_RANK", True),
        ("NET_RATING", "LEAGUE_TEAM_NET_RATING_RANK", False),
    ]

    for source_col, rank_col, ascending in league_rank_specs:
        if source_col in team_df.columns:
            team_df[rank_col] = team_df.groupby("SEASON")[source_col].rank(
                method="min",
                ascending=ascending,
            )

    league_rank_cols = [col for col in team_df.columns if col.startswith("LEAGUE_TEAM_")]
    if league_rank_cols:
        result = result.merge(
            team_df[team_key_cols + league_rank_cols],
            on=team_key_cols,
            how="left",
        )

    return result


def add_previous_season_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["PLAYER_ID", "SEASON"]).copy()

    lag_cols = [
        "target_all_nba",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "GP",
        "MIN",
        "W_PCT",
        "TS_PCT",
        "USG_PCT",
        "PIE",
    ]
    existing_lag_cols = [col for col in lag_cols if col in result.columns]

    for col in existing_lag_cols:
        result[f"PREV_{col}"] = result.groupby("PLAYER_ID")[col].shift(1)

    prev_cols = [f"PREV_{col}" for col in existing_lag_cols]
    result[prev_cols] = result[prev_cols].fillna(0)
    return result


def merge_reputation_with_fallback(
    dataset_df: pd.DataFrame,
    reputation_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = [
        "is_all_star_this_season",
        "mvp_vote_points",
        "mvp_vote_max_points",
        "mvp_vote_share",
    ]
    small = reputation_df[["season", "player_key"] + feature_cols].copy()

    result = dataset_df.merge(
        small,
        left_on=["SEASON", "player_key"],
        right_on=["season", "player_key"],
        how="left",
    ).drop(columns=["season"])

    missing_mask = result[feature_cols].isna().all(axis=1)
    if not missing_mask.any():
        return result

    result["_fallback_key"] = result["player_key"].map(build_fallback_key)
    small["_fallback_key"] = small["player_key"].map(build_fallback_key)

    fallback_counts = (
        small.groupby(["season", "_fallback_key"])
        .size()
        .rename("fallback_count")
        .reset_index()
    )
    unique_fallbacks = fallback_counts[fallback_counts["fallback_count"] == 1][
        ["season", "_fallback_key"]
    ]

    fallback_small = (
        small.merge(unique_fallbacks, on=["season", "_fallback_key"], how="inner")
        .drop(columns=["player_key"])
        .drop_duplicates(["season", "_fallback_key"])
    )

    missing_rows = result.loc[missing_mask, ["SEASON", "_fallback_key"]].copy()
    missing_rows["_row_id"] = missing_rows.index

    fallback_merged = missing_rows.merge(
        fallback_small,
        left_on=["SEASON", "_fallback_key"],
        right_on=["season", "_fallback_key"],
        how="left",
    ).set_index("_row_id")

    for col in feature_cols:
        result.loc[missing_mask, col] = result.loc[missing_mask, col].fillna(
            fallback_merged[col]
        )

    return result.drop(columns=["_fallback_key"])


def add_reputation_features(
    dataset_df: pd.DataFrame,
    reputation_df: pd.DataFrame,
) -> pd.DataFrame:
    result = dataset_df.copy()
    if "player_key" not in result.columns:
        result["player_key"] = result["PLAYER_NAME"].apply(normalize_name)

    reputation = reputation_df.copy()
    if "player_key" not in reputation.columns:
        reputation["player_key"] = reputation["player"].apply(normalize_name)

    required_cols = [
        "season",
        "player_key",
        "is_all_star_this_season",
        "mvp_vote_share",
    ]
    missing = [col for col in required_cols if col not in reputation.columns]
    if missing:
        raise ValueError(f"Missing columns in reputation file: {missing}")

    for col in ["mvp_vote_points", "mvp_vote_max_points"]:
        if col not in reputation.columns:
            reputation[col] = 0.0

    result = merge_reputation_with_fallback(result, reputation)

    result["IS_ALL_STAR_THIS_SEASON"] = (
        result["is_all_star_this_season"].fillna(0).astype(int)
    )
    result["MVP_VOTE_POINTS_THIS_SEASON"] = result["mvp_vote_points"].fillna(0.0)
    result["MVP_VOTE_MAX_POINTS_THIS_SEASON"] = result["mvp_vote_max_points"].fillna(0.0)
    result["MVP_VOTE_SHARE_THIS_SEASON"] = result["mvp_vote_share"].fillna(0.0)
    result["PREV_ALL_NBA_TEAM"] = result["PREV_target_all_nba"].fillna(0.0)

    result = result.sort_values(["PLAYER_ID", "SEASON"]).reset_index(drop=True)
    grouped = result.groupby("PLAYER_ID", sort=False)

    result["PREV_ALL_STAR"] = grouped["IS_ALL_STAR_THIS_SEASON"].shift(1).fillna(0).astype(int)
    result["ALL_STAR_SELECTIONS_BEFORE_SEASON"] = (
        grouped["IS_ALL_STAR_THIS_SEASON"].cumsum() - result["IS_ALL_STAR_THIS_SEASON"]
    ).astype(float)
    result["PREV_MVP_VOTE_SHARE"] = grouped["MVP_VOTE_SHARE_THIS_SEASON"].shift(1).fillna(0.0)
    result["MVP_VOTE_SHARE_BEFORE_SEASON_MAX"] = grouped[
        "MVP_VOTE_SHARE_THIS_SEASON"
    ].transform(lambda s: s.cummax().shift(1).fillna(0.0))
    result["ALL_NBA_SELECTIONS_BEFORE_SEASON"] = grouped["target_all_nba"].transform(
        lambda s: (s.gt(0).astype(int).cumsum() - s.gt(0).astype(int))
    ).astype(float)
    result["MAX_ALL_NBA_TEAM_BEFORE_SEASON"] = grouped["target_all_nba"].transform(
        lambda s: s.cummax().shift(1).fillna(0.0)
    )

    return result.drop(
        columns=[
            "is_all_star_this_season",
            "mvp_vote_points",
            "mvp_vote_max_points",
            "mvp_vote_share",
        ],
        errors="ignore",
    )


def build_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    reputation_path: Path = DEFAULT_REPUTATION_PATH,
) -> pd.DataFrame:
    base, advanced = load_raw_stats()
    optional_tables = load_optional_stat_tables()

    print("Base shape:", base.shape)
    print("Advanced shape:", advanced.shape)
    for table_name, table_df, _ in optional_tables:
        print(f"{table_name} shape:", table_df.shape)

    stat_tables = [("player_advanced_stats.csv", advanced, "")] + optional_tables
    df_raw = merge_stat_tables(base, stat_tables)
    df_raw = add_rookie_flag(df_raw)

    all_nba_labels, rookie_labels = load_labels()

    print()
    print_unmatched_labels(df_raw, all_nba_labels, "target_all_nba")
    print_unmatched_labels(df_raw, rookie_labels, "target_rookie")

    df = add_all_nba_labels(df_raw, all_nba_labels)
    df = add_rookie_labels(df, rookie_labels)
    df = add_engineered_features(df)
    df = add_team_relative_features(df)
    df = add_previous_season_features(df)

    if not reputation_path.exists():
        raise FileNotFoundError(f"Missing reputation file: {reputation_path}")

    reputation_df = pd.read_csv(reputation_path)
    df = add_reputation_features(df, reputation_df)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)

    print()
    print("Saved final dataset:")
    print(dataset_path)
    print()
    print("Final shape:", df.shape)
    print()
    print("All-NBA target counts:")
    print(df["target_all_nba"].value_counts().sort_index())
    print()
    print("Rookie target counts:")
    print(df["target_rookie"].value_counts().sort_index())
    print()
    print("Rookie flag counts:")
    print(df["IS_ROOKIE"].value_counts().sort_index())

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--reputation-path",
        type=Path,
        default=DEFAULT_REPUTATION_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        dataset_path=args.dataset_path,
        reputation_path=args.reputation_path,
    )


if __name__ == "__main__":
    main()
