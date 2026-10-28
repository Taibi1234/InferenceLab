\# InferenceLab



A reproducible machine-learning training and model-serving platform built with Python.



\## Current features



\- Compares logistic regression and random forest classifiers

\- Evaluates models using accuracy and macro F1

\- Selects the best model using macro F1

\- Saves the selected model and experiment metrics



\## Run locally



```powershell

py -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt

python scripts/train.py

```



\## Project status



This project is under active development. API serving, automated tests,

containerization, and performance benchmarks will be added incrementally.

## API

Train the model before starting the API:

```powershell
python scripts/train.py
uvicorn app.main:app --reload
```

Interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

- `GET /health`: check API and model status
- `POST /predict`: submit 13 features for classification

## Tests

```powershell
pytest -q

## Model evaluation

Model selection uses five-fold stratified cross-validation on the training
data only. The held-out test set is not used to select the model.

Each experiment records:

- Fold-level accuracy and macro F1 scores
- Mean and standard deviation for each metric
- Final held-out test results
- Random seed and dataset information
- Experiment ID and UTC timestamp

Current metrics are saved in `artifacts/metrics.json`. Experiment runs are
appended to `artifacts/experiment_history.jsonl`.

