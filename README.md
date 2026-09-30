# BirdCLEF+ 2026 — Multi-Label Bioacoustic Classification

Classical machine-learning pipeline for **multi-label species recognition in 5-second soundscape windows**, developed for the BirdCLEF+ 2026 setting as part of the M1 MIND Machine Learning course at Sorbonne Université.

The project focuses on a difficult combination of **234 target classes**, severe class imbalance, overlapping species and distribution shift between isolated training clips and deployment soundscapes. Rather than training a deep audio network, we studied how far a carefully validated classical pipeline could go with explicit audio features and LightGBM.

## Highlights

- Handcrafted spectral and temporal audio features, including MFCC-oriented descriptors.
- Complementary log-mel / PCEN and prototype signals.
- One-vs-rest **LightGBM** classifiers for multi-label prediction.
- Train-fold-only species co-occurrence correction.
- Soundscape-specific **sub-window temporal analysis**.
- Leakage-safe validation with **GroupKFold by soundscape file**.
- Best confirmed **public Kaggle macro-AUC: 0.783**.

## Experimental progression

| Stage | Grouped soundscape validation AUC | Public Kaggle AUC |
| --- | ---: | ---: |
| Conservative classical enrichment | 0.785 | 0.720 |
| Multi-label co-occurrence correction | 0.834 | 0.766 |
| Soundscape sub-window branch | **0.881** | **0.783** |

These values come from the final academic report and the retained experimental notebooks. Validation and public-test scores are deliberately reported separately.

## Repository structure

```text
.
├── experiments/
│   ├── 01_eda_birdclef.py
│   ├── 02_multilabel_cooccurrence.py
│   ├── 03_soundscape_subwindows.py
│   ├── 04_grouped_validation_no_leakage.py
│   └── 05_kaggle_submission_0783.py
├── docs/
│   ├── RESULTS.md
│   └── REPRODUCIBILITY.md
├── requirements.txt
└── README.md
```

The files in `experiments/` are clean text exports of the original Jupyter/Colab notebooks. Cell boundaries are preserved with `# %%` markers; notebook outputs were intentionally removed to keep the repository reviewable and avoid committing generated artifacts.

## Methodology

The final pipeline follows this high-level flow:

```text
raw audio
   ↓
5 s windows / soundscape sub-windows
   ↓
explicit audio features + log-mel / PCEN signals
   ↓
LightGBM one-vs-rest models
   ↓
co-occurrence correction + temporal smoothing
   ↓
soundscape-specific branch
   ↓
234-class probability vector
```

The most important methodological choice was the validation protocol: neighboring windows from the same soundscape must not be split across train and validation folds. The final tuning therefore groups by soundscape file to reduce leakage.

## Data

The repository does **not** redistribute the BirdCLEF+ 2026 audio dataset or the large intermediate Parquet feature caches used during the project. Those artifacts are intentionally excluded because of size and reproducibility concerns.

See `docs/REPRODUCIBILITY.md` for the expected execution environment and path configuration.

## Limitations

- The system is a classical feature-based approach; no end-to-end neural audio model was trained.
- Public leaderboard performance is not a substitute for evaluation on the hidden/private competition test set.
- Rare classes remain difficult under severe imbalance.
- The gap between grouped validation (0.881) and public test (0.783) shows that domain shift remains substantial.
- Several notebooks were originally executed in Kaggle/Colab and therefore contain environment-specific path configuration that must be adapted locally.

## Academic context

**Sorbonne Université — Master 1 MIND, Machine Learning, 2025–2026**

Authors:
- **Zineddine Mohammedi**
- **Wafaa Berrais**

This repository presents the academic work as a technical portfolio project; it does not claim an individual contribution split that was not documented in the original deliverables.