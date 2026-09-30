# Verified results

The project used **macro ROC-AUC** as the main competition metric.

| Experiment | Grouped validation | Public Kaggle test | Interpretation |
| --- | ---: | ---: | --- |
| Classical baseline enriched with log-mel / prototype signals | 0.785 | 0.720 | Conservative feature enrichment |
| Multi-label co-occurrence correction | 0.834 | 0.766 | Better use of label dependencies |
| Soundscape sub-window branch | **0.881** | **0.783** | Best retained public submission |

## What mattered most

The experiments indicated that improving **domain alignment with real soundscapes** and using a **leakage-safe grouped validation protocol** were more reliable than aggressively increasing feature complexity.

## What did not solve the task

The final report documents limitations of one-vs-rest learning, rare-class performance, noise/overlap and synthetic mixing experiments. Some candidate improvements increased complexity without producing robust validation gains and were not retained in the final submission.

## Reporting rule

The repository does not present training-set performance as final evidence. Validation scores and public Kaggle scores are kept separate because they answer different questions.
