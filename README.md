# NBA Awards Prediction Project

Machine learning project for predicting **All-NBA** and **All-Rookie** teams for the 2026 NBA season.

The project builds a season-level player dataset, trains separate models for All-NBA and All-Rookie prediction, and generates the final JSON file required by the project scoring format.

## Project Goal

The task is not a simple binary classification problem. The model has to:

- select the correct players,
- rank candidates,
- assign All-NBA players to the 1st, 2nd, and 3rd teams,
- assign rookies to the 1st and 2nd All-Rookie teams,
- save the final prediction in the required JSON structure.

## Model Overview

The final solution uses the `previous_team_share` feature set. This feature set includes regular season statistics, advanced NBA API statistics, total production features, previous-season features, and team-relative features.

The final model is split into two independent parts.

### All-NBA Model

The All-NBA prediction is handled by a two-stage model:

1. **Stage 1:** `RandomForestClassifier`
   - ranks players by their probability of belonging to one of the All-NBA teams,
   - creates a candidate pool for the final team assignment.

2. **Stage 2:** `slot_binary_ml`
   - uses separate binary models for the 1st, 2nd, and 3rd All-NBA teams,
   - assigns players to the correct team slots,
   - improves control over team placement instead of relying only on a single global ranking.

This structure was chosen because the All-NBA task requires not only finding the top 15 players, but also assigning them to the correct team level.

### All-Rookie Model

The All-Rookie prediction uses a simpler one-stage model:

- `HistGradientBoostingClassifier`

The model is trained only on rookie-eligible players. It ranks rookie candidates and assigns the top 5 to the 1st All-Rookie team and the next 5 to the 2nd All-Rookie team.

A second-stage model was also tested for All-Rookie, but it did not improve the result, so the final version keeps the simpler one-stage approach.

## Data

The project uses data collected from NBA-related sources and prepared into a single modelling dataset.

Main data groups:

- NBA API player statistics:
  - `Base`
  - `Advanced`
  - `Misc`
  - `Usage`
  - `Scoring`
  - `Defense`
- last 20 games statistics,
- rookie status data,
- historical All-NBA and All-Rookie labels,
- additional experimental data such as reputation signals, draft history, Basketball Reference metrics, and All-NBA vote data.

The final model does not use all prepared experimental features. Some additional data sources were tested during experimentation but were not included in the final feature set because they did not give a stable improvement.

## Structure

- `src/collect_data.py` — downloads source data
- `src/build_dataset.py` — builds the modelling dataset
- `src/train.py` — trains the final model
- `src/predict.py` — generates the final prediction JSON
- `data/` — data files
- `models/` — saved trained model
- `Kruk_Artsiom.json` — final prediction file

## Run

```bash
python src/collect_data.py
python src/build_dataset.py
python src/train.py
python src/predict.py
```

If the data files are already present, it is usually enough to rebuild the dataset, train the model, and generate the prediction:

```bash
python src/build_dataset.py
python src/train.py
python src/predict.py
```

## Output

The final prediction is saved in:

```text
Kruk_Artsiom.json
```

This JSON file contains five lists required by the project scoring format:

- `first all-nba team`
- `second all-nba team`
- `third all-nba team`
- `first rookie all-nba team`
- `second rookie all-nba team`

## Final 2026 Prediction

### All-NBA

| Team | Players |
|---|---|
| 1st All-NBA Team | Nikola Jokić, Luka Dončić, Shai Gilgeous-Alexander, Victor Wembanyama, Chet Holmgren |
| 2nd All-NBA Team | Jaylen Brown, Donovan Mitchell, Kevin Durant, Kawhi Leonard, Jamal Murray |
| 3rd All-NBA Team | Karl-Anthony Towns, Cade Cunningham, Tyrese Maxey, Jalen Johnson, Jalen Brunson |

### All-Rookie

| Team | Players |
|---|---|
| 1st All-Rookie Team | Cooper Flagg, VJ Edgecombe, Kon Knueppel, Derik Queen, Cedric Coward |
| 2nd All-Rookie Team | Maxime Raynaud, Jeremiah Fears, Dylan Harper, Ace Bailey, Collin Murray-Boyles |

## Validation Result Example

For the 2025 validation season, the model produced strong player-selection results without relying on manual corrections of the final prediction:

| Category | Correct players | Correct team placements |
|---|---:|---:|
| All-NBA | 14 / 15 | 10 / 15 |
| All-Rookie | 9 / 10 | 9 / 10 |

These values describe hits and exact team placement, not the project point score.

## Notes

Reputation features, draft-based features, Basketball Reference metrics, and vote-related features were tested during experimentation. They were not included in the final model because they did not provide a stable improvement over the selected `previous_team_share` feature set.

The final approach keeps the model relatively simple while preserving separate modelling strategies for All-NBA and All-Rookie.
