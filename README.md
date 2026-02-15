# CW1 Tabular Regression

Regression pipeline for predicting outcomes on the CW1 test set. The script trains an XGBoost model on `CW1_train.csv` and writes predictions for `CW1_test.csv` to a submission CSV.

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt` (pandas, numpy, scikit-learn, xgboost, etc.)

## Setup

Work from the **project root** directory for all commands below.

### 1. Create a virtual environment (optional)

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Windows (PowerShell):**

```bash
.\.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to run

Run the evaluation script from the project root so that paths like `./data/` resolve correctly:

```bash
python -m src.evaluation
```

This will:

1. Load `data/CW1_train.csv` and `data/CW1_test.csv`
2. Train the pipeline (feature engineering, preprocessor, and XGBoost) on the training set
3. Predict on the test set and save predictions to `data/CW1_submission_K23169225.csv` (one column: `yhat`)

## Project structure

| Path | Description |
|------|-------------|
| `src/evaluation.py` | Trains the model and generates the submission file |
| `src/pipeline.py` | Feature engineering and ML pipeline utilities |
| `data/` | Training data (`CW1_train.csv`), test data (`CW1_test.csv`), and output (`CW1_submission_*.csv`) |