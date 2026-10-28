\# InferenceLab

[![CI](https://github.com/Taibi1234/InferenceLab/actions/workflows/ci.yml/badge.svg)](https://github.com/Taibi1234/InferenceLab/actions/workflows/ci.yml)

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

## Continuous integration

GitHub Actions automatically trains the model and runs the test suite on
pushes and pull requests targeting `main`.

To run the same checks locally:

```powershell
python scripts/train.py
python -m pytest -q

## Docker

Build the image:

```powershell
docker build -t inferencelab:latest .
```

Run the API:

```powershell
docker run --rm -p 8000:8000 --name inferencelab inferencelab:latest
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The model is trained automatically while the image is being built.


- `POST /predict/batch`: classify between 1 and 1,000 rows in one request

## Performance benchmark

Start the API, then run the benchmark from another terminal:

```powershell
python scripts/benchmark.py --requests 100 --batch-size 50 --batches 20
```

The benchmark measures individual and batch request latency, throughput,
and p50/p95/p99 latency after a warm-up period.

Results are saved to `artifacts/benchmark.json`. These measurements come
from one sequential local client and are environment-dependent.

## Observability

The API exposes Prometheus-compatible metrics at `/metrics`:

- HTTP request counts by method, route, and status code
- HTTP request duration histograms
- Individual and batch prediction totals

Application metrics reset when the process restarts.
