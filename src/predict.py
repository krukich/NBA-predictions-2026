from pathlib import Path
import argparse

import train as pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=pipeline.DATASET_PATH)
    parser.add_argument("--models-path", type=Path, default=pipeline.MODELS_PATH)
    parser.add_argument("--output-path", type=Path, default=pipeline.OUTPUT_PATH)
    parser.add_argument("--season", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pipeline.load_dataset(args.dataset_path)
    bundle = pipeline.load_bundle(args.models_path)
    prediction, all_nba_selected, rookie_selected = pipeline.predict_from_bundle(dataset, args.season, bundle)
    pipeline.save_prediction(prediction, args.output_path)

    print(f"Saved prediction to {args.output_path}")
    print()
    print("All-NBA")
    print(all_nba_selected[["PLAYER_NAME", "stage1_score"]].to_string(index=False))
    print()
    print("Rookie")
    print(rookie_selected[["PLAYER_NAME", "stage1_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
