import argparse
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sklearn.datasets import load_wine


def percentile(values, percentage):
    ordered_values = sorted(values)
    rank = math.ceil((percentage / 100) * len(ordered_values))
    return ordered_values[max(rank - 1, 0)]


def select_rows(rows, start, count):
    return [rows[(start + index) % len(rows)] for index in range(count)]


def summarize_results(request_count, rows_per_request, latencies_ms, elapsed):
    total_predictions = request_count * rows_per_request

    return {
        "request_count": request_count,
        "rows_per_request": rows_per_request,
        "total_predictions": total_predictions,
        "duration_seconds": round(elapsed, 4),
        "requests_per_second": round(request_count / elapsed, 2),
        "predictions_per_second": round(total_predictions / elapsed, 2),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 3),
            "p50": round(percentile(latencies_ms, 50), 3),
            "p95": round(percentile(latencies_ms, 95), 3),
            "p99": round(percentile(latencies_ms, 99), 3),
        },
    }


def run_benchmark(args):
    rows = load_wine().data.tolist()

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()

        # Warm up the server and model before collecting measurements.
        for index in range(10):
            response = client.post(
                "/predict",
                json={"features": rows[index]},
            )
            response.raise_for_status()

        warmup_batch = select_rows(rows, 0, args.batch_size)
        response = client.post(
            "/predict/batch",
            json={"rows": warmup_batch},
        )
        response.raise_for_status()

        single_latencies_ms = []
        single_start = time.perf_counter()

        for index in range(args.requests):
            request_start = time.perf_counter()
            response = client.post(
                "/predict",
                json={"features": rows[index % len(rows)]},
            )
            response.raise_for_status()
            elapsed_ms = (time.perf_counter() - request_start) * 1000
            single_latencies_ms.append(elapsed_ms)

        single_elapsed = time.perf_counter() - single_start

        batch_latencies_ms = []
        batch_start = time.perf_counter()

        for batch_index in range(args.batches):
            start = batch_index * args.batch_size
            batch_rows = select_rows(rows, start, args.batch_size)

            request_start = time.perf_counter()
            response = client.post(
                "/predict/batch",
                json={"rows": batch_rows},
            )
            response.raise_for_status()
            elapsed_ms = (time.perf_counter() - request_start) * 1000
            batch_latencies_ms.append(elapsed_ms)

        batch_elapsed = time.perf_counter() - batch_start

    single_results = summarize_results(
        args.requests,
        1,
        single_latencies_ms,
        single_elapsed,
    )
    batch_results = summarize_results(
        args.batches,
        args.batch_size,
        batch_latencies_ms,
        batch_elapsed,
    )

    speedup = (
        batch_results["predictions_per_second"]
        / single_results["predictions_per_second"]
    )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "method": "Sequential requests from one client after a warm-up",
        "single_prediction": single_results,
        "batch_prediction": batch_results,
        "batch_throughput_speedup": round(speedup, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark InferenceLab")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmark.json"),
    )
    args = parser.parse_args()

    if min(args.requests, args.batch_size, args.batches) <= 0:
        parser.error("Benchmark counts and batch size must be positive.")

    if args.batch_size > 1000:
        parser.error("Batch size cannot exceed the API limit of 1,000.")

    results = run_benchmark(args)

    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved benchmark results to {args.output}")


if __name__ == "__main__":
    main()