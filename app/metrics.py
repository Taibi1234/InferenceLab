from prometheus_client import Counter, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "inferencelab_http_requests_total",
    "Total HTTP requests processed by the API.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "inferencelab_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)

PREDICTIONS_TOTAL = Counter(
    "inferencelab_predictions_total",
    "Total model predictions served.",
    ["endpoint"],
)

DRIFT_CHECKS_TOTAL = Counter(
    "inferencelab_drift_checks_total",
    "Total drift checks by overall status.",
    ["status"],
)