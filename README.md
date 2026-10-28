# News Article Classification

Project for the **Data Science and Machine Learning Lab** course, Politecnico di Torino (Winter call, A.Y. 2025/2026). Developed as a team of two with Riccardo Polazzi.

The goal is to classify short news snippets into seven topical categories using only the dataset provided by the course. The final model is a class-balanced linear SVM on TF-IDF text features combined with publisher and time metadata, reaching a **Macro F1 of 0.7148** on a held-out validation set.

## Task

Each record is a news article with metadata (`source`, `page_rank`, `timestamp`) and text (`title`, `article`). The label is one of seven categories:

| Label | Category |
|---|---|
| 0 | International News |
| 1 | Business |
| 2 | Technology |
| 3 | Entertainment |
| 4 | Sports |
| 5 | General News |
| 6 | Health |

Predictions are evaluated with **Macro F1**, the unweighted mean of the per-class F1 scores. This gives the smallest class (Health, 3.9% of the data) the same weight as the largest (International, 29.4%).

## Data challenges

Exploratory analysis (`analysis.ipynb`) showed several issues that shaped the pipeline:

- **Class imbalance**: from 29.4% (International) down to 3.9% (Health).
- **Short texts**: about 35 words on average, with a tail of snippets under 5 words.
- **Label noise**: 2,954 duplicated (title, article) rows and 1,516 texts appearing with conflicting labels.
- **Invalid timestamps**: about 35% of rows contain the placeholder `0000-00-00`.
- **Source skew**: 1,359 distinct sources following a power law; Yahoo, Reuters and BBC account for about 45% of the rows, and 64% of sources appear fewer than 5 times.

## Approach

1. **Cleaning.** Exact duplicates are collapsed. Texts with conflicting labels are resolved by majority vote, and unresolved ties are discarded. Snippets shorter than 20 characters are dropped. This is done before the train/validation split, so no exact copy of a text appears on both sides.
2. **Text construction.** `source` and `title` are repeated three times before the article body, to up-weight the most informative fields. Text is lowercased, HTML-decoded, stripped of tags and non-alphanumeric characters, and stemmed with the Snowball stemmer.
3. **Time features.** `has_date`, `hour` and `day_of_week`, with `-1` used for invalid timestamps so that "missing" stays distinct from midnight.
4. **Text features.** TF-IDF with unigrams and bigrams, `min_df=3`, sublinear term frequency, followed by χ² feature selection (top 40,000 features).
5. **Metadata features.** One-hot encoding of `source` and `day_of_week` (rare values grouped into one category); min-max scaling of `page_rank`, `hour` and `has_date`.
6. **Classifier.** `LinearSVC` with `class_weight='balanced'` and `C=0.2`. A low `C` means stronger regularization, which helps tolerate noisy labels. Class weights penalize errors on minority classes more heavily.
7. **Model selection.** Stratified 80/20 hold-out split. A 3-fold `GridSearchCV` (Macro F1) on the training part over 36 configurations: n-gram range, `min_df`, number of selected features `k`, and `C`. The final model is retrained on the full development set.

## Results

Best configuration: n-gram range (1, 2), `min_df=3`, `k=40000`, `C=0.2`.

- Cross-validation Macro F1: **0.7072**
- Validation Macro F1: **0.7148** (accuracy 0.73)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 International | 0.77 | 0.75 | 0.76 |
| 1 Business | 0.74 | 0.82 | 0.78 |
| 2 Technology | 0.81 | 0.84 | 0.82 |
| 3 Entertainment | 0.65 | 0.53 | 0.59 |
| 4 Sports | 0.79 | 0.94 | 0.86 |
| 5 General News | 0.60 | 0.47 | 0.52 |
| 6 Health | 0.56 | 0.85 | 0.67 |

Sports and Technology are the easiest classes thanks to a distinctive vocabulary. General News and Entertainment are the hardest: short snippets with overlapping vocabulary are frequently confused with International News. Class weighting gives Health a high recall (0.85) at the cost of some precision.

## Repository structure

| File | Description |
|---|---|
| `main.py` | Full pipeline: preprocessing, training, prediction on the evaluation set and generation of `submission.csv`. |
| `analysis.ipynb` | Exploratory data analysis, hyperparameter search, validation report and confusion matrix. |
| `Report_exam_winter_2026.pdf` | Project report. |

The dataset (`development.csv`, `evaluation.csv`) is provided by the course and is not included in this repository.

## How to run

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk torch
```

Place `development.csv` and `evaluation.csv` in the same folder as `main.py`, then run:

```bash
python main.py
```

This writes `submission.csv` with columns `Id,Predicted`, in the format required by the course submission platform. To explore the data and reproduce the validation results, open `analysis.ipynb` with Jupyter or VS Code.
