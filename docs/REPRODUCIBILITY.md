# Reproducibility notes

## Environment

The experiments were developed primarily in Kaggle and Google Colab with Python and the dependencies listed in `requirements.txt`.

## Data

BirdCLEF audio is not committed to this repository. Configure the dataset root in the experiment configuration cells/scripts before execution.

The original academic submission also used intermediate Parquet caches for expensive feature extraction. They are excluded from GitHub to keep the repository lightweight. Re-running from raw audio therefore requires recomputing those features.

## Execution order

1. `01_eda_birdclef.py`
2. `02_multilabel_cooccurrence.py`
3. `03_soundscape_subwindows.py`
4. `04_grouped_validation_no_leakage.py`
5. `05_kaggle_submission_0783.py`

The scripts are exports of the original notebooks, not a packaged production library. They preserve the experimental code and cell structure so the methodology remains auditable.