from pathlib import Path
import argparse
import json
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "processed" / "player_season_dataset.csv"
MODELS_PATH = ROOT_DIR / "models" / "final_pipeline.joblib"
METADATA_PATH = ROOT_DIR / "models" / "final_pipeline_metadata.json"
OUTPUT_PATH = ROOT_DIR / "Kruk_Artsiom.json"

FEATURE_COLUMNS = [
    "AGE",
    "GP",
    "W",
    "L",
    "W_PCT",
    "MIN",
    "PTS",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PLUS_MINUS",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TO",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PACE",
    "PIE",
    "IS_ROOKIE",
    "TOTAL_MIN",
    "TOTAL_PTS",
    "TOTAL_REB",
    "TOTAL_AST",
    "PTS_REB_AST",
    "STOCKS",
    "AST_TOV_SIMPLE",
    "PREV_target_all_nba",
    "PREV_PTS",
    "PREV_REB",
    "PREV_AST",
    "PREV_STL",
    "PREV_BLK",
    "PREV_GP",
    "PREV_MIN",
    "PREV_W_PCT",
    "PREV_TS_PCT",
    "PREV_USG_PCT",
    "PREV_PIE",
    "IS_MULTI_TEAM_PLAYER",
    "TEAM_PLAYER_COUNT",
    "TEAM_TOTAL_PTS_RANK",
    "TEAM_TOTAL_AST_RANK",
    "TEAM_TOTAL_MIN_RANK",
    "TEAM_TOTAL_PTS_SHARE",
    "TEAM_TOTAL_AST_SHARE",
    "TEAM_TOTAL_MIN_SHARE",
    "IS_TEAM_TOTAL_PTS_LEADER",
    "IS_TEAM_TOTAL_MIN_LEADER",
    "TEAM_TOP3_TOTAL_PTS_FLAG",
    "TEAM_TOP3_TOTAL_MIN_FLAG",
    "LEAGUE_TEAM_W_PCT_RANK",
    "LEAGUE_TEAM_NET_RATING_RANK",
]

FEATURE_SET = "previous_team_share"
ALL_NBA_STAGE1_MODEL_KEY = "random_forest"
ALL_NBA_STAGE2_KIND = "slot_binary_ml"
ROOKIE_STAGE1_MODEL_KEY = "hgb_classifier"
ROOKIE_STAGE2_KIND = "stage1_only"

ALL_NBA_STAGE1_POOL_SIZE = 30
ROOKIE_STAGE1_POOL_SIZE = 20
ALL_NBA_GAMES_MINIMUM = 65
ALL_NBA_ELIGIBILITY_EXCEPTIONS_BY_SEASON = {
    2026: {
        "luka doncic",
        "cade cunningham",
        "victor wembanyama",
    },
}
ALL_NBA_SLOT_SPECS = [
    ("first", 3),
    ("second", 2),
    ("third", 1),
]


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def make_model(model_type: str) -> Pipeline:
    if model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "hgb":
        estimator = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def get_all_nba_features(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def get_rookie_features(base_features: list[str]) -> list[str]:
    return [col for col in base_features if col != "IS_ROOKIE"]


def fit_model(train_df: pd.DataFrame, features: list[str], target_col: str, model_key: str) -> Pipeline:
    if model_key == "random_forest":
        model = make_model("random_forest")
    elif model_key == "hgb_classifier":
        model = make_model("hgb")
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    model.fit(train_df[features], train_df[target_col])
    return model


def add_class_probabilities(
    df: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    target_classes: list[int],
    prefix: str,
) -> pd.DataFrame:
    result = df.copy()
    proba = model.predict_proba(result[features])
    classes = list(model.named_steps["model"].classes_)

    for cls in target_classes:
        col = f"{prefix}_p{cls}"
        if cls in classes:
            result[col] = proba[:, classes.index(cls)]
        else:
            result[col] = 0.0

    return result


def add_stage_score_columns(
    df: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    category: str,
    prefix: str,
) -> pd.DataFrame:
    result = df.copy()

    if category == "all_nba":
        result = add_class_probabilities(result, model, features, [1, 2, 3], prefix)
        result[f"{prefix}_score"] = (
            3.0 * result[f"{prefix}_p3"]
            + 2.0 * result[f"{prefix}_p2"]
            + 1.0 * result[f"{prefix}_p1"]
        )
        result[f"{prefix}_select_prob"] = (
            result[f"{prefix}_p1"]
            + result[f"{prefix}_p2"]
            + result[f"{prefix}_p3"]
        )
    else:
        result = add_class_probabilities(result, model, features, [1, 2], prefix)
        result[f"{prefix}_score"] = (
            2.0 * result[f"{prefix}_p2"]
            + 1.0 * result[f"{prefix}_p1"]
        )
        result[f"{prefix}_select_prob"] = (
            result[f"{prefix}_p1"]
            + result[f"{prefix}_p2"]
        )

    return result


def normalize_player_name_for_eligibility(name: str) -> str:
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


def apply_all_nba_games_filter(candidates: pd.DataFrame, season: int, min_count: int) -> pd.DataFrame:
    if season < 2024:
        return candidates
    if "GP" not in candidates.columns:
        return candidates

    result = candidates.copy()
    exceptions = ALL_NBA_ELIGIBILITY_EXCEPTIONS_BY_SEASON.get(season, set())

    if exceptions and "PLAYER_NAME" in result.columns:
        player_keys = result["PLAYER_NAME"].apply(normalize_player_name_for_eligibility)
        is_exception = player_keys.isin(exceptions)
    else:
        is_exception = pd.Series(False, index=result.index)

    eligible = result[(result["GP"] >= ALL_NBA_GAMES_MINIMUM) | is_exception].copy()
    if len(eligible) >= min_count:
        return eligible
    return candidates


def enrich_pool_features(pool_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = pool_df.copy().reset_index(drop=True)
    result[f"{prefix}_pool_rank"] = np.arange(1, len(result) + 1)
    result[f"{prefix}_pool_pct_rank"] = result[f"{prefix}_pool_rank"] / float(len(result))
    result[f"{prefix}_pool_gap_from_top"] = result[f"{prefix}_score"].max() - result[f"{prefix}_score"]

    for col in [
        "PTS",
        "REB",
        "AST",
        "MIN",
        "GP",
        "PIE",
        "W_PCT",
        "TOTAL_MIN",
        "TOTAL_PTS",
        "TOTAL_REB",
        "TOTAL_AST",
    ]:
        if col not in result.columns:
            continue
        result[f"{prefix}_pool_{col}_pct"] = result[col].rank(
            method="average",
            ascending=True,
            pct=True,
        )

    return result


def build_stage1_pool(
    season_df: pd.DataFrame,
    season: int,
    category: str,
    model: Pipeline,
    features: list[str],
    pool_size: int,
) -> pd.DataFrame:
    if category == "all_nba":
        candidates = apply_all_nba_games_filter(season_df.copy(), season, pool_size)
    else:
        candidates = season_df[season_df["IS_ROOKIE"] == 1].copy()

    if candidates.empty:
        raise ValueError(f"No candidates found for category={category}, season={season}")

    candidates = add_stage_score_columns(
        df=candidates,
        model=model,
        features=features,
        category=category,
        prefix="stage1",
    )

    pool = (
        candidates.sort_values("stage1_score", ascending=False)
        .head(pool_size)
        .copy()
        .reset_index(drop=True)
    )

    return enrich_pool_features(pool, "stage1")


def get_stage2_features(pool_df: pd.DataFrame, base_features: list[str]) -> list[str]:
    features = list(base_features)

    if "SEASON" in pool_df.columns and "SEASON" not in features:
        features.append("SEASON")

    for col in pool_df.columns:
        if col.startswith("stage1_") and col not in features:
            features.append(col)

    return features


def add_binary_slot_score(df: pd.DataFrame, model: Pipeline, features: list[str], score_col: str) -> pd.DataFrame:
    result = df.copy()
    proba = model.predict_proba(result[features])
    classes = list(model.named_steps["model"].classes_)

    if 1 in classes:
        result[score_col] = proba[:, classes.index(1)]
    else:
        result[score_col] = 0.0

    return result


def train_all_nba_slot_models(
    historical_pools: pd.DataFrame,
    base_features: list[str],
) -> tuple[dict[str, Pipeline], dict[str, list[str]]]:
    slot_models: dict[str, Pipeline] = {}
    slot_features: dict[str, list[str]] = {}
    residual_by_season = {
        int(season): historical_pools[historical_pools["SEASON"] == season].copy()
        for season in sorted(historical_pools["SEASON"].unique())
    }

    for slot_name, target_value in ALL_NBA_SLOT_SPECS:
        train_frames = []

        for season in sorted(residual_by_season):
            residual_df = residual_by_season[season].copy()
            residual_df["slot_target"] = (residual_df["target_all_nba"] == target_value).astype(int)
            train_frames.append(residual_df)

        train_df = pd.concat(train_frames, ignore_index=True)
        features = get_stage2_features(train_df, base_features)
        model = make_model("hgb")
        model.fit(train_df[features], train_df["slot_target"])

        slot_models[slot_name] = model
        slot_features[slot_name] = features

        updated_residuals: dict[int, pd.DataFrame] = {}

        for season in sorted(residual_by_season):
            scored_df = add_binary_slot_score(
                df=residual_by_season[season],
                model=model,
                features=features,
                score_col=f"{slot_name}_slot_score",
            )
            selected = (
                scored_df.sort_values([f"{slot_name}_slot_score", "stage1_score"], ascending=False)
                .head(5)
                .copy()
            )
            updated_residuals[season] = scored_df.drop(index=selected.index).copy()

        residual_by_season = updated_residuals

    return slot_models, slot_features


def apply_all_nba_slot_models(
    pool_df: pd.DataFrame,
    slot_models: dict[str, Pipeline],
    slot_features: dict[str, list[str]],
) -> pd.DataFrame:
    residual_df = pool_df.copy()
    selected_frames = []

    for slot_name, _ in ALL_NBA_SLOT_SPECS:
        scored_df = add_binary_slot_score(
            df=residual_df,
            model=slot_models[slot_name],
            features=slot_features[slot_name],
            score_col="stage2_score",
        )
        selected = (
            scored_df.sort_values(["stage2_score", "stage1_score"], ascending=False)
            .head(5)
            .copy()
        )
        selected_frames.append(selected)
        residual_df = scored_df.drop(index=selected.index).copy()

    return pd.concat(selected_frames, ignore_index=True)


def fit_bundle(dataset_df: pd.DataFrame, target_season: int) -> dict:
    train_end = target_season - 1
    all_nba_features = get_all_nba_features(dataset_df)
    rookie_features = get_rookie_features(all_nba_features)

    all_nba_train_df = dataset_df[dataset_df["SEASON"] <= train_end].copy()
    rookie_train_df = dataset_df[
        (dataset_df["SEASON"] <= train_end)
        & (dataset_df["IS_ROOKIE"] == 1)
    ].copy()

    all_nba_stage1_model = fit_model(
        train_df=all_nba_train_df,
        features=all_nba_features,
        target_col="target_all_nba",
        model_key=ALL_NBA_STAGE1_MODEL_KEY,
    )
    rookie_stage1_model = fit_model(
        train_df=rookie_train_df,
        features=rookie_features,
        target_col="target_rookie",
        model_key=ROOKIE_STAGE1_MODEL_KEY,
    )

    historical_pools = []
    for season in sorted(int(value) for value in dataset_df.loc[dataset_df["SEASON"] <= train_end, "SEASON"].unique()):
        season_df = dataset_df[dataset_df["SEASON"] == season].copy()
        historical_pools.append(
            build_stage1_pool(
                season_df=season_df,
                season=season,
                category="all_nba",
                model=all_nba_stage1_model,
                features=all_nba_features,
                pool_size=ALL_NBA_STAGE1_POOL_SIZE,
            )
        )

    historical_pools_df = pd.concat(historical_pools, ignore_index=True)
    all_nba_slot_models, all_nba_slot_features = train_all_nba_slot_models(
        historical_pools=historical_pools_df,
        base_features=all_nba_features,
    )

    return {
        "target_season": target_season,
        "feature_set": FEATURE_SET,
        "all_nba_stage1_model_key": ALL_NBA_STAGE1_MODEL_KEY,
        "all_nba_stage2_kind": ALL_NBA_STAGE2_KIND,
        "rookie_stage1_model_key": ROOKIE_STAGE1_MODEL_KEY,
        "rookie_stage2_kind": ROOKIE_STAGE2_KIND,
        "all_nba_features": all_nba_features,
        "rookie_features": rookie_features,
        "all_nba_stage1_model": all_nba_stage1_model,
        "rookie_stage1_model": rookie_stage1_model,
        "all_nba_slot_models": all_nba_slot_models,
        "all_nba_slot_features": all_nba_slot_features,
    }


def build_output(all_nba_selected: pd.DataFrame, rookie_selected: pd.DataFrame) -> dict:
    all_nba_names = all_nba_selected["PLAYER_NAME"].tolist()
    rookie_names = rookie_selected["PLAYER_NAME"].tolist()

    result = {
        "first all-nba team": all_nba_names[0:5],
        "second all-nba team": all_nba_names[5:10],
        "third all-nba team": all_nba_names[10:15],
        "first rookie all-nba team": rookie_names[0:5],
        "second rookie all-nba team": rookie_names[5:10],
    }

    for key, value in result.items():
        if len(value) != 5:
            raise ValueError(f"Output key '{key}' should contain 5 players.")

    return result


def predict_from_bundle(dataset_df: pd.DataFrame, season: int, bundle: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    season_df = dataset_df[dataset_df["SEASON"] == season].copy()
    if season_df.empty:
        raise ValueError(f"No rows found for season {season}")

    all_nba_selected = apply_all_nba_slot_models(
        pool_df=build_stage1_pool(
            season_df=season_df,
            season=season,
            category="all_nba",
            model=bundle["all_nba_stage1_model"],
            features=bundle["all_nba_features"],
            pool_size=ALL_NBA_STAGE1_POOL_SIZE,
        ),
        slot_models=bundle["all_nba_slot_models"],
        slot_features=bundle["all_nba_slot_features"],
    )

    rookie_selected = build_stage1_pool(
        season_df=season_df,
        season=season,
        category="rookie",
        model=bundle["rookie_stage1_model"],
        features=bundle["rookie_features"],
        pool_size=10,
    )

    prediction = build_output(all_nba_selected=all_nba_selected, rookie_selected=rookie_selected)
    return prediction, all_nba_selected, rookie_selected


def save_bundle(bundle: dict, path: Path = MODELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    metadata = {
        "target_season": bundle["target_season"],
        "feature_set": bundle["feature_set"],
        "all_nba_stage1_model_key": bundle["all_nba_stage1_model_key"],
        "all_nba_stage2_kind": bundle["all_nba_stage2_kind"],
        "rookie_stage1_model_key": bundle["rookie_stage1_model_key"],
        "rookie_stage2_kind": bundle["rookie_stage2_kind"],
        "all_nba_features": bundle["all_nba_features"],
        "rookie_features": bundle["rookie_features"],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def load_bundle(path: Path = MODELS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing models: {path}")
    return joblib.load(path)


def save_prediction(prediction: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(prediction, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--models-path", type=Path, default=MODELS_PATH)
    parser.add_argument("--season", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_path)
    bundle = fit_bundle(dataset_df=dataset, target_season=args.season)
    save_bundle(bundle=bundle, path=args.models_path)
    print(f"Saved models to {args.models_path}")
    print(f"Target season: {args.season}")
    print("All-NBA config: random_forest + slot_binary_ml")
    print("Rookie config: hgb_classifier")


if __name__ == "__main__":
    main()
