# Person A pipeline runner

This script runs the Person-A downstream pipeline starting from the provided XGBoost-ready CSV (xgboost_ready_for_personA.csv.gz).

It performs:
- basic data inspection / quality report
- chronological train/validation/test split
- simple persistence and Ridge baselines across selected horizons
- saves train/val/test CSVs and baseline results

Usage:

python run_personA_pipeline.py --input path/to/xgboost_ready_for_personA.csv.gz --outdir personA/PersonA_COMPLETE

The repository already contains PERSON_A_MODEL_DATA_UNDER_25MB.zip which includes xgboost_ready_for_personA.csv.gz. Extract that ZIP and point --input to the .csv or .csv.gz inside.
