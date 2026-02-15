# CW1 Tabular Regression

Regression pipeline that trains an XGBoost model on `CW1_train.csv` and writes predictions for `CW1_test.csv` to a submission CSV.

## Requirements

- Python 3.x
- pip (for installing dependencies)
- Dependencies in `requirements.txt` (pandas, numpy, scikit-learn, xgboost, etc.)

## Setup

Run all commands from the **project root**.

### 1. Virtual environment (optional)

```bash
python -m venv .venv
```

**Activate it:**

- **Windows (PowerShell):** `.\.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Data

Place `CW1_train.csv` and `CW1_test.csv` in the `data/` folder.

## How to run

From the project root:

```bash
python -m src.evaluation
```

This loads the data, trains the pipeline, predicts on the test set, and saves predictions to `data/CW1_submission_K23169225.csv` (column: `yhat`).

## Project structure

| Path | Description |
|------|-------------|
| `src/evaluation.py` | Trains the model and generates the submission file |
| `src/pipeline.py` | Feature engineering and ML pipeline utilities |
| `data/` | Training/test data and submission output (`CW1_submission_K23169225.csv`) |
