# NBA Awards Prediction Project

Project for predicting All-NBA and All-Rookie teams for the 2026 season.

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
## Output

The final prediction is saved in `Kruk_Artsiom.json`.

This JSON file contains five lists required by the project scoring format:

- `first all-nba team`
- `second all-nba team`
- `third all-nba team`
- `first rookie all-nba team`
- `second rookie all-nba team`
